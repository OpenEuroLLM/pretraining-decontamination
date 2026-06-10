#!/bin/bash


# Change dataset name and uncomment only languages supported by dataset
DATASET="finepdfs-1.0.0"
LANGS=(
  # "bul_Cyrl"
  "cat_Latn"
  # "ces_Latn"
  # "dan_Latn"
  "deu_Latn"
  # "ell_Grek"
  "eng_Latn"
  # "est_Latn"
  # "eus_Latn"
  # "ekk_Latn"
  # "fin_Latn"
  # "fra_Latn"
  # "gle_Latn"
  # "glg_Latn"
  # "hrv_Latn"
  # "hun_Latn"
  # "ita_Latn"
  # "lit_Latn"
  # "lvs_Latn"
  # "mlt_Latn"
  # "nld_Latn"
  # "pol_Latn"
  # "por_Latn"
  # "ron_Latn"
  # "slk_Latn"
  # "slv_Latn"
  "spa_Latn"
  # "swe_Latn"
  # "bos_Cyrl"
  # "bos_Latn"
  # "kat_Geor"
  # "mkd_Cyrl"
  # "sqi_Latn"
  # "als_Latn"
  # "srp_Cyrl"
  # "srp_Latn"
  # "hbs_Cyrl"
  # "tur_Latn"
  # "ukr_Cyrl"
  # "isl_Latn"
  # "nor_Latn"
  # "nno_Latn"
  # "nob_Latn"
)

TOTAL=${#LANGS[@]}
IDX=0

for LANG in "${LANGS[@]}"; do
    IDX=$((IDX + 1))
    echo "[${IDX}/${TOTAL}] Processing LANG=${LANG} DATASET=${DATASET}"

    ### Uncomment this line to update the status of all types of jobs per language
    # python3 utils/status.py --shards-jsonl "4_jobs/$DATASET/${DATASET}_${LANG}.jsonl"

    ### Uncomment this line to run the concatenation of matched n-grams after ALL matching jobs have completed
    # python3 combine_matched_ngrams.py --dataset ${DATASET} --lang ${LANG}

    ### Uncomment this line to run the final combination step after ALL removal jobs have completed
    # python3 combine_jsonl_files.py --dataset ${DATASET} --lang ${LANG} --compress --compression-level 9


    ### Uncomment this line to run the n-gram matching step for each language
    ### NOT RECOMMENDED: unless you already know what you're doing and dataset has few shards per language
    # python3 submitter_ngram_match_array.py \
    #         --shards-jsonl "4_jobs/$DATASET/${DATASET}_${LANG}.jsonl" \
    #         --node-limit 50 \
    #         --partition small \
    #         --time-limit 20:00:00 \
    #         --n-workers 24 \
    #         --mem 448G \
    #         --parse-jsonl

    echo "[${IDX}/${TOTAL}] Done with LANG=${LANG}"
done

echo "All ${TOTAL} language(s) processed for DATASET=${DATASET}."
