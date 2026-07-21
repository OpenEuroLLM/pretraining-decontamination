import argparse
import pickle
import random
import nemo_curator
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import get_client, read_data
from nemo_curator.utils.file_utils import get_all_files_paths_under
from nemo_curator.utils.script_utils import ArgumentHelper
import logging
from utils_nemo import get_all_files_from_text, decompress_files, delete_decompressed_files, resolve_text_field
from pathlib import Path
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("PIL").setLevel(logging.WARNING)

def main(args: argparse.Namespace) -> None:

    client = get_client(**ArgumentHelper.parse_client_args(args))  # noqa: F841
    logging.info(client)

    args.input_text_field = resolve_text_field(args.metadata_path, args.parallel_dataset_config)

    with open(args.input_task_ngrams, "rb") as fp:
        task_ngrams = pickle.load(fp)  # noqa: S301

    decontaminator = nemo_curator.TaskDecontamination(
        [], text_field=args.input_text_field, max_ngram_size=args.max_ngram_size
    )

    TMP_DIR = f"/tmp_nemo/finding/{args.input_data_file.split('/')[-1][:-4]}_{random.randint(0,500)}"
    Path(TMP_DIR).mkdir(exist_ok=True, parents=True)
    
    files = get_all_files_from_text(args.input_data_file)
    # Decompress them
    decompressed_files = decompress_files(files, 
                                          decompressed_dir=TMP_DIR,
                                          parse_jsonl=args.parse_jsonl,
                                          metadata_path=args.metadata_path,
                                          parallel_dataset_config=args.parallel_dataset_config)
    logging.info("Reading datasets")
    dataset = DocumentDataset(
        read_data(
            decompressed_files,
            file_type=args.input_file_type,
            backend="pandas",
            blocksize="128MB",
            files_per_partition=None,
        )
    )
    logging.info("Start finding ngrams")
    result = decontaminator.find_matching_ngrams(task_ngrams, dataset).compute()
    print(f"Found a total of {len(result['matched-ngrams'])} matching n-grams")

    output = {
        "matched-ngrams": result["matched-ngrams"],
        "ngrams-freq": result["ngrams-freq"],
        "max-ngram-size": args.max_ngram_size,
        "min-ngram-size": args.min_ngram_size,
    }
    Path(args.output_matched_ngram_dir).mkdir(parents=True, exist_ok=True)
    output_file = args.output_matched_ngram_dir + f"/{Path(args.input_data_file).stem}.pkl"
    with open(output_file, "wb") as fp:
        pickle.dump(output, fp)
    logging.info("Finished finding ngrams")
    
    # Delete decompressed files in decompressed dir
    delete_decompressed_files(decompressed_dir=TMP_DIR)
    logging.info("Finished deleting decompressed files")

def attach_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        """
    Searches for matching task n-grams in the input dataset
    and writes out a list of n-grams that were found.
""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    arg_helper = ArgumentHelper(parser)

    #arg_helper.add_arg_input_data_dir()
    parser.add_argument(
            "--input-data-file",
            type=str,
            default=None,
            help=help,
    )
    arg_helper.add_arg_input_file_type()
    arg_helper.add_arg_input_text_field()
    arg_helper.add_distributed_args()
    parser.add_argument(
        "--input-task-ngrams",
        type=str,
        default=None,
        help="",
    )
    parser.add_argument(
        "--max-ngram-size",
        type=int,
        default=13,
        help="The maximum n-gram size to consider within the dataset.",
    )
    parser.add_argument(
        "--min-ngram-size",
        type=int,
        default=8,
        help="The minimum n-gram size to consider within the datset.",
    )
    parser.add_argument(
        "--output-matched-ngram-dir",
        type=str,
        default=None,
        help="Output dictionary that contains the output matched n-grams "
        "and the frequency of their matches, min-ngram size, max-ngram "
        "size, and the frequencies of n-gram sizes. All of these data will be "
        "used by remove_matching_grams for which this program is a prerequisite.",
    )
    parser.add_argument("--n_workers", type=int, default=None)
    parser.add_argument("--parse-jsonl",
                        action="store_true",
                        default=False,
                        help="Flag used when jsonl fields differ in metadata type and pandas crashes.")
    parser.add_argument("--metadata-path",
                        type=str,
                        default=None,
                        help="Path to the dataset metadata.yaml inside the container. Used to read id and text field names.")
    parser.add_argument("--parallel-dataset-config",
                        type=str,
                        default=None,
                        help="Path to the parallel dataset configuration file (2_datasets/*.yaml). Used only for parallel corpora.")

    # parser.add_argument(
    #     "--memory_limit",
    #     type=str,
    #     default=None
    # )
    return parser


if __name__ == "__main__":
    main(attach_args().parse_args())
