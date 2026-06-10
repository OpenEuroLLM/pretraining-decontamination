
import yaml
from pathlib import Path
import os
import json
import iso639
import argparse
POSSIBLE_LANGS = None
with open('env_variables.yaml', 'r') as file:
    variables = yaml.safe_load(file)
    POSSIBLE_LANGS = variables["LANGS"]


def generate_jobs(datasets_file, input_dir, output_dir):
    with open(datasets_file, 'r') as file:
        splits = yaml.safe_load(file)

    for dataset_name, value in splits.items():
        dataset_path = value["dataset_path"]
        if "lang" in value:
            possible_langs = [value["lang"]]
        else:
            possible_langs = POSSIBLE_LANGS
        for dataset_lang in possible_langs:
            output_path = Path(output_dir, dataset_name)
            output_path.mkdir(exist_ok=True, parents=True)
            output_file = Path(output_path, f"{dataset_name}_{dataset_lang}.jsonl")
            with open(output_file, "w", encoding="utf-8") as output_jobs:
                
                for directory in value["directories"]:
                    full_language_name = iso639.Lang(dataset_lang.split("_")[0]).name.lower()
                    if "greek" in full_language_name:
                        full_language_name = "greek"
                    if "latvian" in full_language_name:
                        full_language_name = "latvian"
                    if "norwegian" in full_language_name:
                        full_language_name = "norwegian"
                    if "albanian" in full_language_name:
                        full_language_name = "albanian"
                    if "estonian" in full_language_name:
                        full_language_name = "estonian"
                        
                    
                    files_dir = Path(input_dir) / dataset_name  / directory.replace("{lang}", dataset_lang)
                    if not files_dir.exists():
                        print(f"{files_dir} does not exist, skipping")
                        continue  
                    files = os.listdir(files_dir)
                    files.sort(key=lambda s: int(s.rsplit("_", 1)[-1].split(".")[0]))
                    for file_path in files:
                        metadata_path = value.get("metadata_path")
                        job = {
                            "dataset_name": f"{dataset_name}/{dataset_lang}",
                            "name": f"{dataset_path}/{file_path}".replace("/", "_")[:-4],
                            "status": "NOT_STARTED",
                            "path": str(Path(files_dir, file_path)),
                            "job_id": None,
                            "task_benchmark_ngrams": f"1_task_ngrams/{full_language_name}_benchmarks.pkl",
                            "metadata_path": f"/data/{metadata_path}" if metadata_path else None,
                        }
                        json.dump(job, output_jobs)
                        output_jobs.write("\n")
        
   
def main():
    parser = argparse.ArgumentParser(description="Split directory file list into chunk files.")
    parser.add_argument("--dataset", required=True, help="Directory containing files.")
    parser.add_argument("--input_dir", default="3_splits", help="Directory containing files.")
    parser.add_argument("--output_dir", default="4_jobs", help="Output directory for chunk files.")
    args = parser.parse_args()
    generate_jobs(args.dataset, args.input_dir, args.output_dir)
    print("Ok")

main()