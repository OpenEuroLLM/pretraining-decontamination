from typing import List
import zstandard as zstd
from pathlib import Path
import gzip
import os
import io
import json
import glob
import hashlib
from collections import Counter, defaultdict
import pickle
from loguru import logger
import subprocess
import yaml


def get_all_files_from_text(file_path: str) -> List[str]:
    """Reads a file containing a list of file paths and returns them as a list."""
    with open(file_path, "r") as f:
        files = [line.strip() for line in f if line.strip()]
    return files


def _read_dataset_metadata(metadata_path: str) -> dict:
    """Reads a dataset metadata.yaml and returns its contents. Returns {} if path is empty or file not found."""
    if not metadata_path:
        return {}
    path = Path(metadata_path)
    if not path.exists():
        logger.warning(f"metadata.yaml not found at {metadata_path}, using default id/text field names")
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    logger.info(f"Loaded dataset metadata from {metadata_path}: id={data.get('id', 'id')!r}, text={data.get('text', 'text')!r}")
    return data


def _get_nested(obj, field_path: str):
    """Traverse a dotted field path like 'metadata.WARC-Record-ID' into a nested dict."""
    val = obj
    for key in field_path.split("."):
        if not isinstance(val, dict):
            return None
        val = val.get(key)
    return val


def _read_parallel_dataset_side(parallel_dataset_config_path) -> str | None:
    """Reads a 2_datasets/*.yaml dataset config and returns its 'parallel' side ('source' or 'target'), or None."""
    if not parallel_dataset_config_path:
        return None
    path = Path(parallel_dataset_config_path)
    if not path.exists():
        logger.warning(f"parallel dataset config not found at {parallel_dataset_config_path}")
        return None
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    for dataset_info in data.values():
        if "parallel" in dataset_info:
            return dataset_info["parallel"]
    return None


def resolve_dataset_fields(metadata_path, parallel_dataset_config=None):
    """
    Resolves the id/text field names to use for a dataset from its metadata.yaml,
    accounting for parallel datasets via parallel_dataset_config (which declares
    the "source"/"target" side being processed).
    Returns (id_field, text_field, source_text_field, target_text_field).
    id_field is None when the metadata has no "id" field for a parallel dataset -
    callers must then compute the id from source_text_field/target_text_field.
    """
    meta = _read_dataset_metadata(metadata_path)
    parallel_side = _read_parallel_dataset_side(parallel_dataset_config)

    if parallel_side and "parallel" in meta:
        source_text_field = meta["parallel"]["source"]["text"]
        target_text_field = meta["parallel"]["target"]["text"]
        text_field = meta["parallel"][parallel_side]["text"]
        id_field = meta.get("id")
        return id_field, text_field, source_text_field, target_text_field

    id_field = meta.get("id", "id")
    text_field = meta.get("text", "text")
    return id_field, text_field, None, None


def resolve_text_field(metadata_path, parallel_dataset_config=None) -> str:
    """Returns just the text field name to use, e.g. for TaskDecontamination's text_field."""
    _, text_field, _, _ = resolve_dataset_fields(metadata_path, parallel_dataset_config)
    return text_field


def _filter_jsonl_stream(
    f_in,
    f_out,
    id_field: str | None = "id",
    text_field: str = "text",
    source_text_field: str | None = None,
    target_text_field: str | None = None,
) -> int:
    """
    Streams JSONL line by line, strips each record to just {id_field, text_field},
    and writes the result to f_out. Eliminates pandas type inference crashes caused
    by other fields (e.g. dates) having inconsistent types across records.
    Supports dotted-path field names (e.g. 'metadata.WARC-Record-ID').
    Malformed JSON lines are logged and skipped.

    If id_field is None (parallel dataset metadata has no "id" field), the id is
    computed from the source/target text fields instead.
    Returns the number of records written.
    """
    written = 0
    for line_num, line in enumerate(f_in, 1):
        try:
            obj = json.loads(line)
            text_value = _get_nested(obj, text_field)
            if id_field is None:
                src_text = _get_nested(obj, source_text_field)
                tgt_text = _get_nested(obj, target_text_field)
                id_value = hashlib.sha256(f"{src_text}{tgt_text}".encode("utf-8")).hexdigest()
                record = {"id": id_value, text_field: text_value}
            else:
                record = {id_field: _get_nested(obj, id_field), text_field: text_value}
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
        except json.JSONDecodeError:
            logger.exception(f"Invalid JSON at line {line_num}")
    return written


def decompress_files(file_paths, decompressed_dir, parse_jsonl=False, metadata_path=None, parallel_dataset_config=None):
    decompressed_files = []
    decompressed_dir = Path(decompressed_dir)
    decompressed_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Decompressing files to {decompressed_dir}")
    zstd_decompressor = zstd.ZstdDecompressor()

    if not parse_jsonl:
        for path in file_paths:
            path = Path(path)

            # --- GZIP ---
            if path.suffix == ".gz":
                decompressed_path = decompressed_dir / path.with_suffix("").name
                with gzip.open(path, "rb") as f_in, open(decompressed_path, "wb") as f_out:
                    f_out.write(f_in.read())
                decompressed_files.append(decompressed_path)

            # --- ZSTD ---
            elif path.suffix in (".zst", ".zstd"):
                shard_dir = path.parent.name  # local-shard_0_of_10
                base_name = path.with_suffix("").name  # shard_00000079_processed.jsonl
                decompressed_name = f"{shard_dir}_{base_name}"
                decompressed_path = decompressed_dir / decompressed_name
                cmd = [
                        "/usr/bin/zstd",
                        "--decompress",
                        "--output",
                        str(decompressed_path),
                        str(path),
                    ]

                subprocess.run(cmd, check=True)
                decompressed_files.append(decompressed_path)

            # --- Uncompressed ---
            else:
                decompressed_files.append(path)
    else:
        id_field, text_field, source_text_field, target_text_field = resolve_dataset_fields(
            metadata_path, parallel_dataset_config
        )

        for path in file_paths:
            path = Path(path)

            # --- GZIP ---
            if path.suffix == ".gz":
                decompressed_path = decompressed_dir / path.with_suffix("").name
                with gzip.open(path, "rt", encoding="utf-8") as f_in, \
                    open(decompressed_path, "w", encoding="utf-8") as f_out:
                    written = _filter_jsonl_stream(
                        f_in, f_out, id_field=id_field, text_field=text_field,
                        source_text_field=source_text_field, target_text_field=target_text_field,
                    )

            # --- ZSTD ---
            elif path.suffix in (".zst", ".zstd"):
                shard_dir = path.parent.name  # local-shard_0_of_10
                base_name = path.with_suffix("").name  # shard_00000079_processed.jsonl
                decompressed_name = f"{shard_dir}_{base_name}"
                decompressed_path = decompressed_dir / decompressed_name
                with path.open("rb") as f_in, \
                    zstd_decompressor.stream_reader(f_in) as reader, \
                    open(decompressed_path, "w", encoding="utf-8") as f_out:

                    text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                    written = _filter_jsonl_stream(
                        text_stream, f_out, id_field=id_field, text_field=text_field,
                        source_text_field=source_text_field, target_text_field=target_text_field,
                    )

            # --- Uncompressed ---
            else:
                decompressed_path = decompressed_dir / path.name
                with path.open("r", encoding="utf-8") as f_in, \
                    open(decompressed_path, "w", encoding="utf-8") as f_out:
                    written = _filter_jsonl_stream(
                        f_in, f_out, id_field=id_field, text_field=text_field,
                        source_text_field=source_text_field, target_text_field=target_text_field,
                    )

            if written == 0:
                # An empty (zero-record) file has no columns once parsed by pandas.
                # If dask picks it to infer the dataframe's meta, every other partition's
                # id/text columns will look like a "Metadata mismatch" against it, so drop it here.
                logger.warning(f"{path} produced 0 valid records after filtering, skipping it: {decompressed_path}")
                decompressed_path.unlink()
                continue

            decompressed_files.append(decompressed_path)
    logger.info(f"Finished decompressed {len(decompressed_files)} files.")
    return decompressed_files

def delete_decompressed_files(decompressed_dir):
    """Deletes all files in the specified decompressed directory."""
    decompressed_dir = Path(decompressed_dir)
    if decompressed_dir.exists() and decompressed_dir.is_dir():
        for file in decompressed_dir.iterdir():
            if file.is_file() and str(file).endswith(('.jsonl', '.parquet', '.json')):
                os.remove(file)
        decompressed_dir.rmdir()
        
        
def merge_level1(total, data):
    for field, value in data.items():
        if isinstance(value, defaultdict):
            if field not in total:
                total[field] = defaultdict(int)
            for k, v in value.items():
                total[field][k] += v
    return total

def combine_jsonl_files(directory):
    files = Path().glob(f"{directory}/*.jsonl")
    print(files)
    groups = {}
    for file in files:
        match = Path(file).name.split("_block_")[0]
        if match not in groups:
            groups[match] = []
        groups[match].append(file)

    for prefix, file_list in groups.items():
        output_file = f"{directory}/full_results/{prefix}.jsonl"
        print(f"Combining into {output_file} ...")

        with open(output_file, "w", encoding="utf-8") as outfile:
            for fname in sorted(file_list):
                with open(fname, "r", encoding="utf-8") as infile:
                    for line in infile:
                        outfile.write(line)

        # Remove original split files
        # for fname in file_list:
        #     Path(fname).unlink()
        #     print(f"Removed {fname}")

    print("Done!")