# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import pickle
from pathlib import Path
import random
import nemo_curator
import shutil
import pyarrow.parquet as pq
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import get_client, read_data, write_to_disk
from nemo_curator.utils.file_utils import (
    expand_outdir_and_mkdir,
    get_all_files_paths_under,
    get_batched_files

)
from nemo_curator.utils.script_utils import ArgumentHelper
from utils_nemo import (
    get_all_files_from_text,
    decompress_files,
    delete_decompressed_files
)

def batch_list(lst, batch_size=50):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

logging.basicConfig(level=logging.DEBUG)

def main(args: argparse.Namespace) -> None:
    client = get_client(**ArgumentHelper.parse_client_args(args))  # noqa: F841
    output_rm_doc_dir = None
    if args.output_removed_doc_dir is not None:
        output_rm_doc_dir = expand_outdir_and_mkdir(args.output_removed_doc_dir)

    output_rm_doc_dir = Path(output_rm_doc_dir) / Path(args.input_data_file).stem
    if output_rm_doc_dir.is_dir():
        logging.info("Directory already exists, removing")
        shutil.rmtree(output_rm_doc_dir, ignore_errors=True)
    output_rm_doc_dir.mkdir(exist_ok=True, parents=True)
    
    # Each rank read in the task data
    print(f"Reading in matched n-grams from {args.input_matched_ngrams}")
    print(f"Saving removed in {output_rm_doc_dir}")
    with open(args.input_matched_ngrams, "rb") as fp:
        logging.info(f"Reading from {args.input_matched_ngrams}")
        matched_ngram_data = pickle.load(fp)  # noqa: S301
    
    # Unpack the results from find_matched_ngrams
    matched_ngrams = matched_ngram_data["matched-ngrams"]
    ngrams_freq = matched_ngram_data["ngrams-freq"]
    max_ngram_size = matched_ngram_data["max-ngram-size"]
    decontaminator = nemo_curator.TaskDecontamination(
        [],
        text_field=args.input_text_field,
        max_ngram_size=max_ngram_size,
        max_matches=args.match_threshold,
        max_splits=args.max_document_splits,
        removed_dir=output_rm_doc_dir,
    )
    files = get_all_files_from_text(args.input_data_file)
    
    # Decompress them
    TMP_DIR = f"/tmp_nemo/removing/{args.input_data_file.split('/')[-1][:-4]}_{random.randint(0,500)}"
    OUTPUT_DIR = f"/tmp_nemo/cleaned/{args.input_data_file.split('/')[-1][:-4]}_{random.randint(0,500)}"
    Path(TMP_DIR).mkdir(parents=True, exist_ok=True)
    decompressed_files = decompress_files(files, decompressed_dir=TMP_DIR, parse_jsonl=args.parse_jsonl, metadata_path=args.metadata_path)
    for batch in batch_list(decompressed_files):
        dataset = DocumentDataset(
            read_data(
                batch,
                file_type=args.input_file_type,
                files_per_partition=None,
                blocksize="128MB",
                backend="pandas",
                add_filename=True,
            )
        )
        decontaminated_dataset = decontaminator.remove_matching_ngrams(
            matched_ngrams, ngrams_freq, dataset
        )
        write_to_disk(
            decontaminated_dataset.df,
            OUTPUT_DIR,
            write_to_filename=True,
            output_type=args.output_file_type,
        )
    # Delete decompressed files in decompressed dir
    delete_decompressed_files(decompressed_dir=TMP_DIR)
    shutil.rmtree(OUTPUT_DIR)
    print("Finished decontaminating all files")


def attach_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    arg_helper = ArgumentHelper(parser)

    arg_helper.add_arg_batch_size()
    arg_helper.add_arg_input_file_type()
    arg_helper.add_arg_input_text_field()
    arg_helper.add_arg_output_file_type()
    arg_helper.add_distributed_args()
    parser.add_argument(
            "--input-data-file",
            type=str,
            default=None,
            help=help,
    )

    parser.add_argument(
        "--input-matched-ngrams",
        type=str,
        default=None,
        required=True,
        help="Input dictionary (.pkl file) that contains matched n-gram data from the find_matching_ngrams code.",
    )
    parser.add_argument(
        "--match-threshold",
        type=int,
        default=10,
        help="A threshold that determines if a matched n-gram will be "
        "considered for removal in remove_matching_ngrams. N-grams that "
        "exceed this number of matches in the training dataset will not be "
        "considered during the removal stage.",
    )
    parser.add_argument(
        "--max-document-splits",
        type=int,
        default=10,
        help="A threshold used to determine if a document should be removed "
        "from the corpus if it is split more than "
        "--max-document-splits number of times.",
    )
    parser.add_argument(
        "--output-removed-doc-dir",
        type=str,
        default=None,
        help="Output directory to where removed documents will be written. "
        "Documents will be removed from the corpus if they are split more "
        "than --max-document-splits number of times, or if the user specifies "
        "that they be removed via the flag --remove-split-docs.",
    )

    parser.add_argument("--parse-jsonl",
                        action="store_true",
                        default=False,
                        help="Flag used when jsonl fields differ in metadata type and pandas crashes.")
    parser.add_argument("--metadata-path",
                        type=str,
                        default=None,
                        help="Path to the dataset metadata.yaml inside the container. Used to read id and text field names.")

    # parser.add_argument(
    #     "--output-task-deduped-dir",
    #     type=str,
    #     default=None,
    #     required=True,
    #     help="Output directory to where task-deduplicated (split) documents will be written.",
    # )

    return parser


def console_script() -> None:
    parser = argparse.ArgumentParser(
        """
 Using the matching n-grams find by
 nemo_curator/scripts/find_matching_ngrams.py
 (provided by the argument --input-matched-ngrams),
 passes over all documents and removes matching n-grams from the corpus by
 splitting documents containing the match. If a document is split more than
 --max-splits times, it is removed from the corpus.
""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    main(attach_args(parser).parse_args())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        """
        Using the matching n-grams find by
        nemo_curator/scripts/find_matching_ngrams.py
        (provided by the argument --input-matched-ngrams),
        passes over all documents and removes matching n-grams from the corpus by
        splitting documents containing the match. If a document is split more than
        --max-splits times, it is removed from the corpus.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    main(attach_args(parser).parse_args())
