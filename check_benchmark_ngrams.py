#!/usr/bin/env python3
"""
External diagnostic for the NeMo Curator task-decontamination IndexError:

    File ".../nemo_curator/modules/task.py", line 232, in _find_ngrams
        positions[last_seq_start_position + i],
    IndexError: list index out of range

It does NOT touch NeMo Curator internals. It only instantiates your benchmark
task classes (exactly like your YAML loader does), calls generate_ngrams(), and
re-parses every produced n-gram key with the SAME get_words() the matcher uses.

The crash is governed by one number per benchmark:
    reparsed_len(key) = len(get_words(key)[0])
i.e. how many words the matching side (`_compute_ngram_freq_sorted`) sees in a
key. A document crashes whenever some key's reparsed_len <= that document's
whitespace "desync" D. Concretely:

    reparsed_len == 0  -> crashes EVERY sufficiently long document (even clean text)
    reparsed_len == 1  -> crashes any document with >=1 non-ASCII-whitespace token
    reparsed_len <  min_ngram_size -> generation/matcher counts disagree (the bug)

Your existing check (len(generated_ngrams) < 10) cannot find this: it counts the
NUMBER of keys, not the per-key reparsed length. A benchmark can emit thousands
of keys and still contain one length-1 key.

Usage:
    python check_benchmark_ngrams.py path/to/english_benchmarks.yaml
    python check_benchmark_ngrams.py path/to/file.yaml --show 12 --default-min 8

Requires NeMo Curator (and your `benchmarks...` package) importable, since it
loads the actual datasets.
"""

import argparse
import importlib
import string
import sys
import unicodedata
from collections import Counter

import yaml


# ---------------------------------------------------------------------------
# Use the CONTAINER's real get_words if available, so the diagnostic exactly
# matches what the matcher (_compute_ngram_freq_sorted / _find_ngrams) runs,
# regardless of NeMo Curator version. Fall back to a local mirror otherwise.
# ---------------------------------------------------------------------------
try:
    from nemo_curator.utils.text_utils import get_words  # noqa: F401
    _GET_WORDS_SOURCE = "nemo_curator.utils.text_utils.get_words (container)"
except Exception:  # noqa: BLE001
    def get_words(text):  # mirror of v0.6/v0.7
        word_start_char_positions = []
        prev = 0
        words = []
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        if len(text) > 0:
            for i in range(len(text)):
                if text[i] != " ":
                    if i == 0 or text[i - 1] == " ":
                        word_start_char_positions.append(i)
                        if i != 0:
                            words.append(text[prev:i].strip())
                        prev = i
            words.append(text[prev : i + 1].strip())
            if words[0] == "":
                words = words[1:]
        return words, word_start_char_positions
    _GET_WORDS_SOURCE = "local mirror (nemo_curator not importable)"


def has_nonascii_whitespace(s):
    """True if the string contains whitespace that get_words does NOT treat as
    a separator (anything that is whitespace but not the ASCII space ' ')."""
    for ch in s:
        if ch != " " and (ch.isspace() or unicodedata.category(ch) in ("Zs", "Zl", "Zp")):
            return True
    return False


def nonascii_ws_codepoints(s):
    cps = set()
    for ch in s:
        if ch != " " and (ch.isspace() or unicodedata.category(ch) in ("Zs", "Zl", "Zp")):
            cps.add("U+%04X" % ord(ch))
    return sorted(cps)


def load_class(path: str):
    """'package.module.ClassName' -> class object."""
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def ngram_keys(generated):
    """generate_ngrams() returns a dict {ngram: 0} in the base class, but be
    tolerant of a plain iterable of strings too."""
    if isinstance(generated, dict):
        return list(generated.keys())
    return list(generated)


def resolve_min(obj, params, default_min):
    """Best-effort recovery of the benchmark's min_ngram_size."""
    if "min_ngram_size" in params:
        return params["min_ngram_size"]
    for attr in ("_min_ngram_size", "min_ngram_size"):
        if hasattr(obj, attr):
            return getattr(obj, attr)
    return default_min


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("yaml_path", help="YAML file listing tasks (same format as your loader)")
    ap.add_argument("--show", type=int, default=8,
                    help="How many shortest keys to print per flagged benchmark (default 8)")
    ap.add_argument("--default-min", type=int, default=8,
                    help="Fallback min_ngram_size if a task doesn't expose one (default 8)")
    args = ap.parse_args()

    with open(args.yaml_path, "r") as f:
        data = yaml.safe_load(f)
    tasks = data["tasks"]

    print(f"get_words source: {_GET_WORDS_SOURCE}")
    print(f"Loaded {len(tasks)} task(s) from {args.yaml_path}\n")
    print(f"{'benchmark':<22}{'keys':>9}{'min_len':>9}{'<min':>8}{'len0':>7}{'len1':>7}  verdict")
    print("-" * 88)

    flagged = []          # (name, list of (reparsed_len, key))
    global_min = None     # smallest reparsed length seen across everything

    for item in tasks:
        dotted = item["name"]
        short_name = dotted.split(".")[-1]
        params = item.get("params", {}) or {}

        try:
            cls = load_class(dotted)
            obj = cls(**params)
            generated = obj.generate_ngrams()
        except Exception as e:  # noqa: BLE001 - we want to keep going
            print(f"{short_name:<22}{'ERROR':>9}   {type(e).__name__}: {e}")
            continue

        keys = ngram_keys(generated)
        min_n = resolve_min(obj, params, args.default_min)

        reparsed = [(len(get_words(k)[0]), k) for k in keys]
        lens = [L for L, _ in reparsed]
        dist = Counter(lens)
        mn = min(lens) if lens else None
        n_below = sum(1 for L in lens if L < min_n)
        n_zero = dist.get(0, 0)
        n_one = dist.get(1, 0)

        if mn is not None:
            global_min = mn if global_min is None else min(global_min, mn)

        if not keys:
            verdict = "NO KEYS GENERATED"
        elif n_zero:
            verdict = "DANGEROUS: len-0 key crashes even CLEAN docs"
        elif n_below:
            verdict = f"DANGEROUS: {n_below} key(s) re-parse < min={min_n}"
        else:
            verdict = "ok"

        print(f"{short_name:<22}{len(keys):>9}{str(mn):>9}{n_below:>8}{n_zero:>7}{n_one:>7}  {verdict}")

        if n_below or n_zero:
            offenders = sorted(reparsed, key=lambda t: t[0])[: args.show]
            flagged.append((short_name, offenders))

    print("-" * 88)
    if global_min is not None:
        print(f"\nSmallest re-parsed n-gram length across all benchmarks: {global_min}")
        print("A document crashes whenever its whitespace desync D >= this value.")
        if global_min == 0:
            print("  -> 0 means *every* long document crashes, regardless of content.")
        elif global_min == 1:
            print("  -> 1 means any document containing a single NBSP/tab/etc. token crashes.")

    if not flagged:
        print("\nNo benchmark produced a short-re-parsing key. If you still crash, the")
        print("trigger is document-side: measure D on the dataset's text instead.")
        return

    print("\n" + "=" * 88)
    print("OFFENDING KEYS (repr shown so non-ASCII whitespace like \\xa0 is visible)")
    print("=" * 88)
    for name, offenders in flagged:
        print(f"\n### {name}")
        for L, k in offenders:
            ws = nonascii_ws_codepoints(k)
            ws_note = f"   [non-ASCII ws: {', '.join(ws)}]" if ws else ""
            print(f"  reparsed_len={L:<3} {k!r}{ws_note}")

    print("\nFix at the source: in each flagged task's generate_ngrams, normalize")
    print("whitespace before _update_ngrams, e.g.")
    print('    text = " ".join(text.split())')
    print("which splits on ALL Unicode whitespace and rejoins with single ASCII")
    print("spaces, so the generation count and the matcher count agree again.")


if __name__ == "__main__":
    sys.exit(main())
