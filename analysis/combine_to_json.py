# Iterate over a folder and combine all JSONL files into a single JSONl file
from pathlib import Path

input_folder = Path("/work/alicia/decontamination_2026/nemotron/eng_Latn/final_removed_data").rglob("*.jsonl")
output_file = Path("/work/alicia/decontamination_2026/nemotron/eng_Latn/combined_removed_nemotron-cc.jsonl")
with output_file.open("w", encoding="utf-8") as outfile:
    for jsonl_file in input_folder:
        with jsonl_file.open("r", encoding="utf-8") as infile:
            for line in infile:
                outfile.write(line)