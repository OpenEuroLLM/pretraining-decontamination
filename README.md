
## 1. Environment 

### 1.1. Variables setup
Prior to running any commands, modify the `env_variables.yaml` file and update the following variables to reflect your own configuration:

```
# Your Lumi username and compute project
USERNAME: "your_username_YYYY"
PROJECT_ID: "project_XXXX"

# The two paths below should exist prior to running the pipeline
# Temporary folder where data will be temporarily decompressed, etc.
TMP_DIR: "/scratch/project_XXXX/users/your_username_YYYY/tmp"
# Work folder for the decontamination pipeline: logs, middle files, final output.
DECONTAMINATION_DIR: "/scratch/project_XXXX/users/your_username_YYYY/decontamination"
```

This step should only be performed once, unless the username or compute project changes at any point during processing.

### 1.2. Singularity image

This pipeline makes use of a custom Singularity image containing a modified version of NemoCurator. For convenience, the Singularity image may be copied directly from the following Lumi path into your root directory of this repository:

`/users/tudormateiu/decontamination/pretraining-decontamination/nemo.sif`

### 1.3. Indexed benchmark generated n-grams

To save compute, the generated benchmark n-grams have already been indexed and prepared in the NemoCurator input format. They can be found and copied from:

```
/users/tudormateiu/decontamination/pretraining-decontamination/1_task_ngrams
```

## 2. Task preparation

For each dataset to be decontaminated, the following steps must be configured in advance. The steps must be executed in the prescribed order, any modification to a prior step requires re-execution of all subsequent steps, with the relevant output folders removed beforehand.

It is recommended that the output folder for each step be located within the working directory of this repository. Examples of the output can be found in the respective folders:

```
2_datasets/
3_splits/
4_jobs/
```

### 2.1. Prepare dataset configuration file
A configuration `.yaml` file must be prepared for each dataset on which decontamination must be run. This file must have the following format:

```
DATASET_NAME:
  dataset_path: 
  metadata_path: 
  chunk_size:
  lang: 
  directories:
    - 
    - 
```

where,

- `dataset_path` is the "root" sub-folder path to the data, following DATASETS_DIR in env_variables.yaml, e.g. FinePDFs is found in `/scratch/project_465002530/training/collection/flag/finepdfs-1.0.0`, given that `DATASETS_DIR = /scratch/project_465002530/training` then the `dataset_path = collection/flag/finepdfs-1.0.0`
- `metadata_path` OPTIONAL, but REQUIRED when it exists. Path to the dataset's 'metadata.yaml' file in the catalogue
- `chunk_size` is the number of shards processed by a single Slurm job. This value should be reduced when individual shards are large.
- `lang` declare this field if the dataset is monolingual (e.g. for English-only datasets)
- `directories` list of sub-paths within `dataset_path` that lead to the data files

Some examples are available in `2_datasets/`. 

#### FinePDFs example

This example uses the `flag` training catalogue folder structure. As the dataset is multilingual, `lang` is declared inline within the `directories` paths rather than as a top-level field.

```
finepdfs-1.0.0:
  dataset_path: collection/flag/finepdfs-1.0.0
  metadata_path: collection/flag/finepdfs-1.0.0/metadata.yaml
  chunk_size: 10
  directories:
    - "source/{lang}"
```

#### Nemotron-CC example

This example uses the `flag` training catalogue folder structure and demonstrates the declaration of a monolingual `lang` value (`eng_Latn`). Multiple `directories` entries are specified to cover the different data sub-paths.

```
nemotron-cc-1.0:
  dataset_path: collection/flag/nemotron-cc-1.0
  metadata_path: collection/flag/nemotron-cc-1.0/metadata.yaml
  lang: eng_Latn
  chunk_size: 5
  directories:
    - "source/high/actual"
    - "source/medium/actual"
    - "source/medium-high/actual"
```

### 2.2. Prepare task splits files

The next step is to generate text files which contain a `chunk_size` amount of shard paths per file for use in subsequent slurm jobs.

The following example shows how to create these text files for FinePDFs:

```
python3 ./2_generate_splits.py --datasets 2_datasets/finepdfs-1.0.0.yaml --output_dir 3_splits
```

where,

- `--datasets` is the file to a dataset configuration file from step `2.1.`
- `--output-dir` is the output directory where all generated split/shard text files will be stored, for use in step `2.3.`

### 2.3. Prepare Slurm job files

The next step is to generate the Slurm job `.jsonl` files, which are produced automatically on a per-language basis for each dataset. These files contain all the information relevant to the jobs for a language:

1. Matching job Slurm ID and STATUS, which start as null.
2. Marking job Slurm ID and STATUS, generated after marking jobs are submitted.
3. Paths to individual shard text files from `2.2.`
4. Other relevant information for jobs and the pipeline

The following example shows how to create the slurm job files for FinePDFs:

```
python3 3_create_jobs.py --dataset 2_datasets/finepdfs-1.0.0.yaml
```

where,

- `--datasets` is the file to a dataset configuration file from step `2.1.`

This command will output the slurm job files to `4_jobs/`, organized under the dataset name and its respective languages.

## 3. Matching process

### 3.1. Running matching jobs

To run the NemoCurator matching process, use the following command, which submits Slurm array jobs to the cluster. The example below targets FinePDFs Catalan:

```
python3 submitter_ngram_match_array.py --shards-jsonl 4_jobs/finepdfs-1.0.0/finepdfs-1.0.0_cat_Latn.jsonl --node-limit 20 --partition small --time-limit 8:00:00 --n-workers 20 --mem 448G --account project_465002530
```

where,

- `--shards-jsonl` is the slurm job file created in step `2.3.` for the specific language you want to run
- `--node-limit` the maximum number of Slurm jobs to submit concurrently; note that Lumi enforces an upper limit on simultaneous job submissions
- `--partition` the Lumi partition to use; defaults to `small`
- `--time-limit` time limit per job in this array submission, in `HH:MM:SS` format. Set higher for larger sharded datasets.
- `--n-workers` number of parallel workers to use within each slurm job
- `--mem` memory allocation per job (e.g. `448G`)
- `--account` the Lumi compute project account to charge for the jobs

This command must be re-run until all matching jobs report a `COMPLETED` status. Job statuses can be refreshed using the status utility described in the Utils section below.

### 3.2. Generate combined matching per language

Once all matching jobs for a given language have completed, the results must be combined into a single pickle file. This can be accomplished with the following command:

```
python3 combine_matched_ngrams.py --dataset finepdfs-1.0.0 --lang deu_Latn
```

where,

- `--dataset` is the name of the dataset
- `--lang` is the language

This command can be invoked more conveniently via the fast script caller described in the Utils section below.

### 3.3. Decontamination threshold exploration

To determine an appropriate threshold for marking contaminated samples, run the following command (example for FinePDFs):

```
python3 threshold_exploration.py finepdfs-edu-1.0.0
```

This script additionally generates logs under `threshold_logs/` to aid in threshold analysis.

## 4. Marking process

### 4.1. Running marking jobs

To run the NemoCurator marking (removal) process, use the following command, which submits Slurm array jobs to the cluster. The example below targets FinePDFs Catalan:

```
python3 submitter_remove_matches_array.py --shards-jsonl 4_jobs/finepdfs-1.0.0/finepdfs-1.0.0_cat_Latn.jsonl --node-limit 20 --partition small --time-limit 8:00:00 --n-workers 20 --mem 448G --account project_465002530 --match-threshold 10
```

Most arguments are identical to those of the matching process; however, the following parameter must be tuned based on the results of step `3.3.`:

- `--match-threshold` an integer threshold below which (inclusive) a sample is considered contaminated. For example, if the threshold is set to 10, any indexed benchmark n-gram matched 10 times or fewer is flagged as contaminated.

### 4.2. Final result preparation

Once all marking jobs for a given language have completed, the contamination annotations must be merged back into their respective original shards. To perform this final concatenation, for example for FinePDFs Catalan:

```
python3 combine_jsonl_files.py --dataset finepdfs-1.0.0 --lang ${LANG} --compress --compression-level 9
```

where,

- `--dataset` is the dataset name
- `--lang` is the language
- `--compress` flag to compress the final concatenated shards into `.jsonl.zst` format. This is the intended final output format; use `--no-compress` if uncompressed output is required for debugging
- `--compression-level` is the compression level


## Utilities

### Status update

To update all Slurm job statuses (for both matching and marking), run the status command. The example below is for FinePDFs Catalan:

```
python3 utils/status.py --shards-jsonl 4_jobs/finepdfs-1.0.0/finepdfs-1.0.0_cat_Latn.jsonl
```

where,

- `--shards-jsonl` is the slurm job file created in step `2.3.` for the specific language you want to run

### Fast script caller

A Bash script for efficiently invoking multiple pipeline commands is available at `fast_script_caller.sh`. All necessary configuration and instructions for running the various commands described above are documented within the script.

