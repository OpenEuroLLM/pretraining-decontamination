
import re
import json
import yaml
import zstandard as zstd
from pathlib import Path
from collections import defaultdict
import argparse
from loguru import logger

with open('env_variables.yaml', 'r') as file:
    variables = yaml.safe_load(file)
    WORK_DIRECTORY = variables["DECONTAMINATION_DIR"]
    Path(WORK_DIRECTORY).mkdir(exist_ok=True, parents=True)

parser = argparse.ArgumentParser(description="Combine per-worker removed JSONL files into a single file per shard.")
parser.add_argument("--dataset", required=True)
parser.add_argument("--lang", required=True)
parser.add_argument("--compress", action=argparse.BooleanOptionalAction, default=False,
                    help="Compress output as .jsonl.zst (default: plain .jsonl).")
parser.add_argument("--compression-level", type=int, default=9,
                    help="zstd compression level (1=fastest, 22=max). Only used with --compress. Default: 9.")
args = parser.parse_args()

INPUT_DIR  = f"{WORK_DIRECTORY}/{args.dataset}/{args.lang}/removed_data"
OUTPUT_DIR = f"{WORK_DIRECTORY}/{args.dataset}/{args.lang}/final_removed_data"
Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

# Files that NemoCurator writes for a shard nested under a "local-shard_X_of_Y"
# directory get flattened into a filename like "local-shard_0_of_10_shard_00000000_processed.jsonl"
# (the nesting info survives only in the filename). Recover it so it can become
# a real output subdirectory again instead of getting silently discarded.
LOCAL_SHARD_RE = re.compile(r"^(local-shard_\d+_of_\d+)_(.+)$")


def load_source_directories(dataset, lang):
    """Read the dataset's `directories:` config, {lang}-expanded, as used by 2_generate_splits.py."""
    config_path = Path("2_datasets", f"{dataset}.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return [d.replace("{lang}", lang) for d in config[dataset]["directories"]]


def match_source_dir(dir_name, name_for_splits_to_real_dir):
    """Map a removed_data subdir (e.g. 'source_global-shard_01_of_10_0') back to its
    real relative source directory (e.g. 'source/global-shard_01_of_10'), stripping
    the trailing chunk index that 2_generate_splits.py appends."""
    for name_for_splits, real_dir in name_for_splits_to_real_dir.items():
        prefix = f"{name_for_splits}_"
        if dir_name.startswith(prefix) and dir_name[len(prefix):].isdigit():
            return real_dir
    return None


def real_dir_to_output_components(real_dir, lang):
    components = real_dir.split("/")
    if components and components[0] == "source":
        components = components[1:]
    # Drop components equal to the lang code: it's redundant with the outer
    # {lang}/ output folder that's already used for every dataset.
    return [c for c in components if c != lang]


def unflatten_dotted_keys(obj: dict) -> dict:
    """Turn a flat key like 'metadata.WARC-Record-ID' into a nested {'metadata': {'WARC-Record-ID': ...}}.

    NemoCurator needs a flat dotted id_field internally (see utils_nemo.py), so upstream
    stages intentionally keep it flat. This is only un-flattened here, at final output time.
    """
    if not any("." in key for key in obj):
        return obj
    result = {}
    for key, value in obj.items():
        if "." in key:
            *parents, leaf = key.split(".")
            d = result
            for part in parents:
                d = d.setdefault(part, {})
            d[leaf] = value
        else:
            result[key] = value
    return result


def file_has_dotted_keys(path: Path) -> bool:
    with path.open("r", encoding="utf-8") as fh:
        first_line = fh.readline().strip()
    if not first_line:
        return False
    return any("." in key for key in json.loads(first_line))


def rewrite_unflattened_lines(infile):
    for line in infile:
        line = line.strip()
        if not line:
            continue
        obj = unflatten_dotted_keys(json.loads(line))
        yield json.dumps(obj, ensure_ascii=False) + "\n"


try:
    out_lang_dir = Path(OUTPUT_DIR, args.lang)
    out_lang_dir.mkdir(parents=True, exist_ok=True)

    source_directories = load_source_directories(args.dataset, args.lang)
    name_for_splits_to_real_dir = {d.replace("/", "_"): d for d in source_directories}

    groups = defaultdict(list)
    unmatched_dirs = []
    for src_dir in Path(INPUT_DIR).iterdir():
        if not src_dir.is_dir():
            continue

        real_dir = match_source_dir(src_dir.name, name_for_splits_to_real_dir)
        if real_dir is None:
            unmatched_dirs.append(src_dir.name)
            continue
        output_components = real_dir_to_output_components(real_dir, args.lang)
        # decompress_files() in utils_nemo.py prepends each source file's immediate
        # parent directory name to its basename (f"{shard_dir}_{base_name}") to avoid
        # collisions when NemoCurator flattens nested directories. Recover and strip it
        # so it doesn't linger as a redundant filename prefix (e.g. finepdfs's
        # "fra_Latn_" or nemotron-cc's "actual_"). For datasets nested deeper than the
        # configured source directory (e.g. dclm's "local-shard_X_of_10"), this won't
        # match and the LOCAL_SHARD_RE fallback below handles it instead.
        parent_prefix = f"{real_dir.rsplit('/', 1)[-1]}_"

        for f in src_dir.glob("*.jsonl"):
            stem = f.stem
            # Strip UUID suffix: last _-separated token containing hyphens is a UUID
            parts = stem.rsplit("_", 1)
            no_uuid = parts[0] if len(parts) == 2 and "-" in parts[1] else stem
            base = no_uuid[len(parent_prefix):] if no_uuid.startswith(parent_prefix) else no_uuid

            local_shard_match = LOCAL_SHARD_RE.match(base)
            if local_shard_match:
                file_components = (*output_components, local_shard_match.group(1))
                shard_base = local_shard_match.group(2)
            else:
                file_components = tuple(output_components)
                shard_base = base

            groups[(file_components, shard_base)].append(f)

    if unmatched_dirs:
        logger.error(
            "{} removed_data subdirectory(ies) did not match any configured source "
            "directory for dataset {}: {}. Refusing to guess — fix the dataset config "
            "or investigate these directories.",
            len(unmatched_dirs), args.dataset, sorted(unmatched_dirs),
        )
        raise SystemExit(1)

    if not groups:
        logger.warning("No .jsonl files found under {}", INPUT_DIR)
    else:
        logger.info("Found {} file group(s) to combine", len(groups))

    # Sniff once: dotted keys (e.g. dclm's "metadata.WARC-Record-ID") are a property of
    # the whole dataset/lang, not of an individual file, so one sample is enough. When
    # present, every line must be parsed and un-flattened; otherwise a raw byte copy is
    # fine and much faster.
    needs_unflatten = False
    if groups:
        sample_file = sorted(next(iter(groups.values())))[0]
        needs_unflatten = file_has_dotted_keys(sample_file)
        if needs_unflatten:
            logger.info("Detected dotted keys (e.g. 'metadata.WARC-Record-ID') — un-flattening into nested objects on output.")

    if args.compress:
        cctx = zstd.ZstdCompressor(level=args.compression_level)
        for (file_components, shard_base), files in sorted(groups.items()):
            out_subdir = out_lang_dir.joinpath(*file_components)
            out_subdir.mkdir(parents=True, exist_ok=True)
            out_file = out_subdir / f"{shard_base}.jsonl.zst"
            logger.info("Writing {} ({} part(s)) -> {}", shard_base, len(files), out_file)
            with out_file.open("wb") as raw_out:
                with cctx.stream_writer(raw_out) as compressor:
                    for f in sorted(files):
                        if needs_unflatten:
                            with f.open("r", encoding="utf-8") as infile:
                                for line in rewrite_unflattened_lines(infile):
                                    compressor.write(line.encode("utf-8"))
                        else:
                            with f.open("rb") as infile:
                                compressor.write(infile.read())
    else:
        for (file_components, shard_base), files in sorted(groups.items()):
            out_subdir = out_lang_dir.joinpath(*file_components)
            out_subdir.mkdir(parents=True, exist_ok=True)
            out_file = out_subdir / f"{shard_base}.jsonl"
            logger.info("Writing {} ({} part(s)) -> {}", shard_base, len(files), out_file)
            with out_file.open("w", encoding="utf-8") as outfile:
                for f in sorted(files):
                    if needs_unflatten:
                        with f.open("r", encoding="utf-8") as infile:
                            for line in rewrite_unflattened_lines(infile):
                                outfile.write(line)
                    else:
                        with f.open("r", encoding="utf-8") as infile:
                            outfile.write(infile.read())

    logger.info("Done — {} group(s) written to {}", len(groups), out_lang_dir)

except SystemExit:
    raise
except Exception:
    logger.exception("combine_jsonl_files failed")
    raise SystemExit(1)