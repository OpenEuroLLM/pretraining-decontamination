import logging
import pickle
import os
from collections import Counter, defaultdict
import argparse
from pathlib import Path
import glob
import yaml

WORK_DIRECTORY = None
with open('env_variables.yaml', 'r') as file:
    variables = yaml.safe_load(file)
    WORK_DIRECTORY = variables["DECONTAMINATION_DIR"]
    
def merge_level1(total, data):
    for field, value in data.items():
        if isinstance(value, defaultdict):
            if field not in total:
                total[field] = defaultdict(int)
            for k, v in value.items():
                total[field][k] += v
    return total

def combine_related_pickles(dataset, lang, final_pickle_name="all_matched_ngrams"):
    folder = Path(WORK_DIRECTORY, dataset, lang, "matched_ngrams")
    total = {}
    pickle_files = [
    f
    for f in glob.glob(os.path.join(folder, "**/*.pkl"), recursive=True)
    if final_pickle_name not in os.path.basename(f)
]
    if len(pickle_files) == 0:
        raise ValueError(f"No pickle files found in {folder} to combine.")
    number_of_jobs = 0
    with open(f"4_jobs/{dataset}/{dataset}_{lang}.jsonl") as f:
        number_of_jobs = sum(1 for _ in f)
    if len(pickle_files) != number_of_jobs:
        raise Exception("The number of pickles doesn't match the number of jobs!!!!!")
    for pf in pickle_files:
        with open(pf, "rb") as fp:
            data = pickle.load(fp)
            total = merge_level1(total, data)
            total["ngrams-freq"] = data["ngrams-freq"]
            total["max-ngram-size"] = data["max-ngram-size"]
            total["min-ngram-size"] = data["min-ngram-size"]
    output_path = os.path.join(folder, f"{final_pickle_name}.pkl")
    # save unified pickle
    with open(output_path, "wb") as fp:
        pickle.dump(total, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Split directory file list into chunk files.")
    parser.add_argument("--dataset", required=True, help="Directory containing files.")
    parser.add_argument("--lang", required=True, help="Directory containing files.")
    args = parser.parse_args()
    combine_related_pickles(args.dataset, args.lang)
    print("Ok")
main()