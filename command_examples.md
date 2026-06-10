## Relevant paths
### Original training catalogue
/appl/local/openeurollm/training/catalogue

### Baby training catalogue
/scratch/project_465002530/training/collection/baby

### Log paths to matching and removal steps per dataset and language
/scratch/project_462000963/users/tudormateiu/decontamination/{DATASET}/{LANG}/logs/{DECONT_STEP}



## Task preparation examples
### Prepare task splits files - FinePDFs example
python3 ./2_generate_splits.py --datasets 2_datasets/finepdfs-1.0.0.yaml --output_dir 3_splits

### Prepare job files - FinePDFs example
python3 3_create_jobs.py --dataset 2_datasets/finepdfs-1.0.0.yaml



## Task submitter examples
### FinePDFs - matching example
python3 submitter_ngram_match_array.py --shards-jsonl 4_jobs/finepdfs-1.0.0/finepdfs-1.0.0_eng_Latn.jsonl --node-limit 20 --partition small --time-limit 8:00:00 --n-workers 20 --mem 448G --account project_465002530 

### DCLM - matching example
python3 submitter_ngram_match_array.py --shards-jsonl 4_jobs/nemotron-cc-1.0/nemotron-cc-1.0_eng_Latn.jsonl --node-limit 20 --partition small --time-limit 10:00:00 --n-workers 20 --mem 448G --account project_465002530 

### FinePDFs - removal example (deu_Latn, th=20)
python3 submitter_remove_matches_array.py --shards-jsonl 4_jobs/finepdfs-1.0.0/finepdfs-1.0.0_eng_Latn.jsonl --node-limit 20 --partition small --time-limit 8:00:00 --n-workers 20 --mem 448G --account project_465002530 --match-threshold 20 

### DCLM - removal example (eng_Latn, th=10)
python3 submitter_remove_matches_array.py --shards-jsonl 4_jobs/nemotron-cc-1.0/nemotron-cc-1.0_eng_Latn.jsonl --node-limit 20 --partition small --time-limit 8:00:00 --n-workers 20 --mem 448G --account project_465002530 --match-threshold 10 

### Examples on "debug" partition for testing
#### Matching
python3 submitter_ngram_match_array.py --shards-jsonl 4_jobs/finepdfs-1.0.0/finepdfs-1.0.0_eng_Latn.jsonl --node-limit 1 --partition debug --time-limit 00:30:00 --n-workers 10 --mem 224G --account project_465002530 
#### Removal
python3 submitter_remove_matches_array.py --shards-jsonl 4_jobs/finepdfs-1.0.0/finepdfs-1.0.0_eng_Latn.jsonl --node-limit 1 --partition debug --time-limit 00:30:00 --n-workers 10 --mem 224G --account project_465002530 --match-threshold 20 

## Other commands
### Matched pickles concatenation example
python3 combine_matched_ngrams.py --dataset finepdfs-1.0.0 --lang deu_Latn

### Task status update example
python3 utils/status.py --shards-jsonl 4_jobs/finepdfs-1.0.0/finepdfs-1.0.0_eng_Latn.jsonl

### Fast script (or sbatch script) caller
bash fast_script_caller.sh
sbatch fast_sbatch_script_caller.sh

### 
