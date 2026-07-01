#!/bin/bash

DATASET="finepdfs-1.0.0"

LANGS_THRESHOLDS=(
  "bul_Cyrl:8"
  "cat_Latn:10"
  "ces_Latn:10"
  "dan_Latn:10"
  "deu_Latn:10"
  "ell_Grek:11"
  "ekk_Latn:10"
  "eus_Latn:5"
  "fin_Latn:5"
  "fra_Latn:20"
  "gle_Latn:5"
  "glg_Latn:20"
  "hrv_Latn:5"
  "hun_Latn:10"
  "ita_Latn:15"
  "lit_Latn:10"
  "lvs_Latn:5"
  "nld_Latn:8"
  "pol_Latn:10"
  "por_Latn:20"
  "ron_Latn:10"
  "slk_Latn:5"
  "slv_Latn:10"
  "spa_Latn:30"
  "swe_Latn:5"
  "kat_Geor:5"
  "als_Latn:3"
  "srp_Cyrl:10"
  "tur_Latn:10"
  "ukr_Cyrl:15"
  "isl_Latn:5"
  "nno_Latn:5"
  "nob_Latn:15"
)

for entry in "${LANGS_THRESHOLDS[@]}"; do
  lang="${entry%%:*}"
  threshold="${entry##*:}"
  python3 submitter_remove_matches_array.py \
    --shards-jsonl "4_jobs/${DATASET}/${DATASET}_${lang}.jsonl" \
    --node-limit 110 \
    --partition small \
    --time-limit 8:00:00 \
    --n-workers 15 \
    --mem 220G \
    --account project_465002530 \
    --match-threshold "$threshold"
done
