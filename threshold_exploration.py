import pickle
import sys
from collections import Counter
from pathlib import Path

import yaml
from loguru import logger


def _load_base_path(env_file: str = "env_variables.yaml") -> str:
    with open(env_file, "r") as f:
        env = yaml.safe_load(f)
    project_id = env["PROJECT_ID"]
    user = env["USERNAME"]
    return f"/scratch/{project_id}/users/{user}/decontamination"


LANGS = [
    "bul_Cyrl", "cat_Latn", "ces_Latn", "dan_Latn", "deu_Latn",
    "ell_Grek", "eng_Latn", "est_Latn", "eus_Latn", "ekk_Latn",
    "fin_Latn", "fra_Latn", "gle_Latn", "glg_Latn", "hrv_Latn",
    "hun_Latn", "ita_Latn", "lit_Latn", "lvs_Latn", "mlt_Latn",
    "nld_Latn", "pol_Latn", "por_Latn", "ron_Latn", "slk_Latn",
    "slv_Latn", "spa_Latn", "swe_Latn", "bos_Cyrl", "bos_Latn",
    "kat_Geor", "mkd_Cyrl", "sqi_Latn", "als_Latn", "srp_Cyrl",
    "srp_Latn", "hbs_Cyrl", "tur_Latn", "ukr_Cyrl", "isl_Latn",
    "nor_Latn", "nno_Latn", "nob_Latn",
]

LOG_DIR = Path("threshold_logs")


def setup_logger(dataset: str) -> None:
    """Configure loguru: stderr + a per-dataset log file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{dataset}.log"

    logger.remove()  # drop the default handler
    logger.add(sys.stderr, level="DEBUG",
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
    logger.add(log_file, level="DEBUG", mode="w",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}")

    logger.info(f"Logging to {log_file}")


def print_ngram_frequencies(dataset: str, langs: list[str] = LANGS) -> None:
    """
    For each language, load matched n-gram counts and log a frequency-of-frequency table.

    Parameters
    ----------
    dataset : str
        Name of the decontaminated dataset (e.g. 'dclm-baseline', 'finepdfs').
    langs : list[str]
        Language codes to iterate over. Defaults to the full LANGS list.
    """
    setup_logger(dataset)

    base_path = _load_base_path()
    logger.info(f"Base path: {base_path}")
    logger.info(f"Dataset : {dataset}")
    logger.info(f"Languages: {len(langs)}")

    found, skipped_langs = 0, []

    for lang in langs:
        pkl_path = Path(base_path) / dataset / lang / "matched_ngrams" / "all_matched_ngrams.pkl"

        # Check pickle existance
        if not pkl_path.exists():
            logger.warning(f"[{lang}] path not found - skipping")
            skipped_langs.append(lang)
            continue

        found += 1
        logger.info(f"-- {lang} {'-' * max(0, 30 - len(lang))}")

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        matched: dict = data["matched-ngrams"]
        freq_of_freq = Counter(matched.values())

        if not freq_of_freq:
            logger.info(f"  (no matched n-grams)")
            continue

        # Log ngram match frequency
        max_count = max(freq_of_freq.values())
        bar_scale = 40 / max_count
        for count, n_grams in sorted(freq_of_freq.items()):
            bar = "█" * max(1, round(n_grams * bar_scale))
            logger.info(f"  count={count:>4}  n-grams={n_grams:>6}  {bar}")

    # Show summary
    skipped = len(skipped_langs)
    skipped_str = ", ".join(skipped_langs) if skipped_langs else "none"
    logger.info("-" * 50)
    logger.info(f"Done. Found: {found}/{len(langs)}  |  Skipped (missing): {skipped}  |  [{skipped_str}]")


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "olmo-mix_pes2o"
    print_ngram_frequencies(dataset)
