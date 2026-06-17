import csv
import json
import logging
import os
import re

import polars as pl
from nemo_curator.tasks.downstream_task import DownstreamTask

from datasets import concatenate_datasets, get_dataset_config_names, load_dataset


class MMLU(DownstreamTask):
    def __init__(
        self,
        min_ngram_size=8,
        max_ngram_size=13,
        split_type=None,
    ):
        super().__init__()
        self._task_name = "mmlu"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("cais/mmlu", "all", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"].strip()
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class COPA(DownstreamTask):
    def __init__(
        self,
        min_ngram_size=8,
        max_ngram_size=13,
        split_type=None,
    ):
        super().__init__()
        self._task_name = "copa"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("aps/super_glue", "copa", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = (
                    line["premise"][:-1].strip()
                    + " "
                    + {"cause": "because", "effect": "therefore"}[line["question"]]
                )
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class Lambada(DownstreamTask):
    def __init__(
        self,
        min_ngram_size=8,
        max_ngram_size=13,
        split_type=None,
    ):
        super().__init__()
        self._task_name = "lambada"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "EleutherAI/lambada_openai", "default", split=split_type
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["text"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class OpenBookQA(DownstreamTask):
    def __init__(
        self,
        min_ngram_size=8,
        max_ngram_size=13,
        split_type=None,
    ):
        super().__init__()
        self._task_name = "openbookqa"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("allenai/openbookqa", "main", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question_stem"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class ArcChallenge(DownstreamTask):
    def __init__(
        self,
        min_ngram_size=8,
        max_ngram_size=13,
        split_type=None,
    ):
        super().__init__()
        self._task_name = "arcchallenge"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class ArcEasy(DownstreamTask):
    def __init__(
        self,
        min_ngram_size=8,
        max_ngram_size=13,
        split_type=None,
    ):
        super().__init__()
        self._task_name = "arceasy"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class BoolQ(DownstreamTask):
    def __init__(
        self,
        min_ngram_size=8,
        max_ngram_size=13,
        split_type=None,
    ):
        super().__init__()
        self._task_name = "boolq"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("aps/super_glue", "boolq", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                text = line["passage"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class HellaSwag(DownstreamTask):
    def __init__(
        self,
        min_ngram_size=8,
        max_ngram_size=13,
        split_type=None,
    ):
        super().__init__()
        self._task_name = "hellaswag"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("Rowan/hellaswag", subset=None, split=split_type)

    def _preprocess(self, text):
        # https://github.com/EleutherAI/lm-evaluation-harness/blob/c9772b90a8ee95b0df1ba76651443a0534c96aad/lm_eval/tasks/hellaswag/utils.py
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        text = text.strip()
        return text

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                ctx = line["ctx_a"] + " " + line["ctx_b"].capitalize()
                text = self._preprocess(line["activity_label"] + ": " + ctx)
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class CommonsenseQA(DownstreamTask):
    def __init__(
        self,
        min_ngram_size=8,
        max_ngram_size=13,
        split_type=None,
    ):
        super().__init__()
        self._task_name = "commonsenseqa"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "tau/commonsense_qa", subset=None, split=split_type
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"].strip()
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class PIQA(DownstreamTask):
    def __init__(
        self,
        min_ngram_size=8,
        max_ngram_size=13,
        split_type=None,
    ):
        super().__init__()
        self._task_name = "piqa"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("baber/piqa", subset=None, split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["goal"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class GSM8K(DownstreamTask):
    def __init__(
        self,
        min_ngram_size=8,
        max_ngram_size=13,
        split_type=None,
    ):
        super().__init__()
        self._task_name = "gsm8k"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("openai/gsm8k", "main", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"].strip()
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                text = line["answer"].strip()
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class MBPP(DownstreamTask):
    def __init__(
        self,
        min_ngram_size=8,
        max_ngram_size=13,
        split_type=None,
    ):
        super().__init__()
        self._task_name = "mbpp"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "google-research-datasets/mbpp", "full", split=split_type
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["text"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                text = line["code"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class MBPP_Plus(DownstreamTask):
    def __init__(
        self,
        min_ngram_size=8,
        max_ngram_size=13,
        split_type=None,
    ):
        super().__init__()
        self._task_name = "mbpp"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("evalplus/mbppplus", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["prompt"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                text = line["code"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class HumanEval(DownstreamTask):
    def __init__(
        self,
        min_ngram_size=8,
        max_ngram_size=13,
        split_type=None,
    ):
        super().__init__()
        self._task_name = "humaneval"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("openai/openai_humaneval", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["prompt"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                text = line["canonical_solution"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class HumanEval_Plus(DownstreamTask):
    def __init__(
        self,
        min_ngram_size=8,
        max_ngram_size=13,
        split_type=None,
    ):
        super().__init__()
        self._task_name = "humaneval"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("evalplus/humanevalplus", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["prompt"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                text = line["canonical_solution"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class GPQA(DownstreamTask):
    def __init__(
        self,
        min_ngram_size=8,
        max_ngram_size=13,
        split_type=None,
    ):
        super().__init__()
        self._task_name = "gpqa_extended"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "Idavidrein/gpqa",
            "gpqa_extended",
            split=split_type,
            token=os.getenv("HF_TOKEN"),
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["Question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class AIME_I_25(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "aime_i_25"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "opencompass/AIME2025", "AIME2025-I", split=split_type
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class AIME_II_25(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "aime_i_25"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "opencompass/AIME2025", "AIME2025-II", split=split_type
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class Hendrycks_MATH(DownstreamTask):
    def __init__(
        self,
        min_ngram_size=8,
        max_ngram_size=13,
        split_type=None,
    ):
        super().__init__()
        self._task_name = f"hendrycks_math"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size

        self._dataset_1 = load_dataset(
            "EleutherAI/hendrycks_math", "algebra", split=split_type
        )
        self._dataset_2 = load_dataset(
            "EleutherAI/hendrycks_math", "counting_and_probability", split=split_type
        )
        self._dataset_3 = load_dataset(
            "EleutherAI/hendrycks_math", "geometry", split=split_type
        )
        self._dataset_4 = load_dataset(
            "EleutherAI/hendrycks_math", "intermediate_algebra", split=split_type
        )
        self._dataset_5 = load_dataset(
            "EleutherAI/hendrycks_math", "number_theory", split=split_type
        )
        self._dataset_6 = load_dataset(
            "EleutherAI/hendrycks_math", "prealgebra", split=split_type
        )
        self._dataset_7 = load_dataset(
            "EleutherAI/hendrycks_math", "precalculus", split=split_type
        )

        self._dataset_8 = load_dataset(
            "HuggingFaceH4/MATH-500", "default", split=split_type
        )

        self._datasets = [
            self._dataset_1,
            self._dataset_2,
            self._dataset_3,
            self._dataset_4,
            self._dataset_5,
            self._dataset_6,
            self._dataset_7,
            self._dataset_8,
        ]

    def generate_ngrams(self):
        for dataset in self._datasets:
            for idx, line in enumerate(dataset):
                try:
                    text = line["problem"]
                    self._update_ngrams(
                        text, self._min_ngram_size, self._max_ngram_size
                    )
                    text = line["solution"]
                    self._update_ngrams(
                        text, self._min_ngram_size, self._max_ngram_size
                    )
                except Exception:
                    logging.exception(f"Error processing line {idx}")
        return self.ngrams


class IFEval(DownstreamTask):
    def __init__(
        self,
        min_ngram_size=8,
        max_ngram_size=13,
        split_type=None,
    ):
        super().__init__()
        self._task_name = "ifeval"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("google/IFEval", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["prompt"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class AlpacaEval(DownstreamTask):
    def __init__(self, file_path, min_ngram_size=8, max_ngram_size=13):
        super().__init__()
        self._task_name = "alpacaeval"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._file_path = file_path
        if self._file_path is None:
            raise Exception("Must provide path to data in CSV format")

    def generate_ngrams(self):
        df = None
        try:
            df = pl.read_csv(self._file_path, has_header=True)
        except Exception:
            logging.exception(f"Error processing documents")

        for idx, text in enumerate(df["instruction"]):
            try:
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class DoNotAnswer(DownstreamTask):
    def __init__(
        self,
        min_ngram_size=8,
        max_ngram_size=13,
        split_type=None,
    ):
        super().__init__()
        self._task_name = "donotanswer"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("LibrAI/do-not-answer", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


# Note: skg/toxigen-data does not exist. using "toxigen/toxigen-data"
class ToxiGen(DownstreamTask):
    def __init__(
        self,
        min_ngram_size=8,
        max_ngram_size=13,
        split_type=None,
    ):
        super().__init__()
        self._task_name = "toxigen"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "toxigen/toxigen-data", "annotated", split=split_type
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = f"Is the following statement hateful? Respond with either Yes or No. Statement: {line['text']}"
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class MTBench(DownstreamTask):
    def __init__(self, file_path, min_ngram_size=8, max_ngram_size=13):
        super().__init__()
        self._task_name = "mtbench"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._file_path = file_path
        if self._file_path is None:
            raise Exception("Must provide path to data in JSONL format")

    def generate_ngrams(self):
        with open(self._file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                try:
                    myjson = json.loads(line)
                    texts = myjson["history"]
                    for text in texts:
                        user_text = text["user"]
                        self._update_ngrams(
                            user_text, self._min_ngram_size, self._max_ngram_size
                        )
                        bot_text = text["bot"]
                        self._update_ngrams(
                            bot_text, self._min_ngram_size, self._max_ngram_size
                        )

                except Exception:
                    logging.exception(f"Error processing line {idx}")
        return self.ngrams


class HarmBench(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, file_dir=None, split_type=None):
        super().__init__()
        self._task_name = "harmbench"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._file_dir = file_dir

    def _behaviour_datasets(self):
        with open(
            self._file_dir + "/behavior_datasets/harmbench_behaviors_text_all.csv", "r"
        ) as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                try:
                    text = row["Behavior"].strip()
                    self._update_ngrams(
                        text, self._min_ngram_size, self._max_ngram_size
                    )
                except Exception:
                    logging.exception(f"Error processing line {idx}")

        with open(
            self._file_dir
            + "/behavior_datasets/harmbench_behaviors_multimodal_all.csv",
            "r",
        ) as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                try:
                    text = row["Behavior"].strip()
                    self._update_ngrams(
                        text, self._min_ngram_size, self._max_ngram_size
                    )
                    text = row["ImageDescription"].strip()
                    self._update_ngrams(
                        text, self._min_ngram_size, self._max_ngram_size
                    )
                except Exception:
                    logging.exception(f"Error processing line {idx}")

    def _classifier_val_sets(self):
        with open(
            self._file_dir + "/classifier_val_sets/text_behaviors_val_set.json",
            "r",
        ) as f:
            data = json.load(f)
            for _, items in data.items():
                try:
                    for item in items:
                        text = item["test_case"].strip()
                        self._update_ngrams(
                            text, self._min_ngram_size, self._max_ngram_size
                        )
                        text = item["generation"].strip()
                        self._update_ngrams(
                            text, self._min_ngram_size, self._max_ngram_size
                        )
                except Exception:
                    logging.exception(f"Error processing line")
        with open(
            self._file_dir + "/classifier_val_sets/multimodal_behaviors_val_set.json",
            "r",
        ) as f:
            data = json.load(f)
            for _, items in data.items():
                try:
                    for item in items:
                        test_cases = item["test_case"]
                        for test_case in test_cases:
                            self._update_ngrams(
                                test_case, self._min_ngram_size, self._max_ngram_size
                            )
                        text = item["generation"].strip()
                        self._update_ngrams(
                            text, self._min_ngram_size, self._max_ngram_size
                        )
                        if "image_description" in item:
                            text = item["image_description"].strip()
                            self._update_ngrams(
                                text, self._min_ngram_size, self._max_ngram_size
                            )
                except Exception:
                    logging.exception(f"Error processing line")

    def _optimizer_targets(self):
        with open(
            self._file_dir + "/optimizer_targets/harmbench_targets_multimodal.json",
            "r",
        ) as f:
            data = json.load(f)
            for _, item in data.items():
                try:
                    text = item.strip()
                    self._update_ngrams(
                        text, self._min_ngram_size, self._max_ngram_size
                    )
                except Exception:
                    logging.exception(f"Error processing line")
        with open(
            self._file_dir + "/optimizer_targets/harmbench_targets_text.json",
            "r",
        ) as f:
            data = json.load(f)
            for _, item in data.items():
                try:
                    text = item.strip()
                    self._update_ngrams(
                        text, self._min_ngram_size, self._max_ngram_size
                    )
                except Exception:
                    logging.exception(f"Error processing line")

    def generate_ngrams(self):
        self._behaviour_datasets()
        self._classifier_val_sets()
        self._optimizer_targets()
        return self.ngrams


class HumanitiesLastExam(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "humanities_last_exam"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "cais/hle", split="test", token=os.getenv("HF_TOKEN") or True
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line[f"question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class LMArena100K(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "lmarena_100k"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "lmarena-ai/arena-human-preference-100k", split="train"
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["conversation_a"]
                for item in text:
                    self._update_ngrams(
                        item["content"], self._min_ngram_size, self._max_ngram_size
                    )
                text = line["conversation_b"]
                for item in text:
                    self._update_ngrams(
                        item["content"], self._min_ngram_size, self._max_ngram_size
                    )
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class LMArena140K(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "lmarena_140k"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "lmarena-ai/arena-human-preference-140k", split="train"
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["conversation_a"][0]["content"][0]["text"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class LMArena55K(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "lmarena_55K"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "lmarena-ai/arena-human-preference-55k", split="train"
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                prompts = json.loads(line["prompt"])
                for text in prompts:
                    self._update_ngrams(
                        text, self._min_ngram_size, self._max_ngram_size
                    )
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


# ------ tables


class AIME2024(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "aime_2024"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("math-ai/aime24", split="test")

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["problem"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class AIME2025(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "aime_2025"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("math-ai/aime25", split="test")

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["problem"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class AIME2026(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "aime_2026"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("math-ai/aime26", split="test")

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["problem"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class AQuARAT(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "aqua_rat"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("deepmind/aqua_rat", "raw", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class BasicSkills(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "basic_skills"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "ellamind/basic-skills", "default", split="validation"
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class BBH(DownstreamTask):

    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "bbh"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        configs = get_dataset_config_names("lukaemon/bbh")
        self._datasets = [
            load_dataset("lukaemon/bbh", c, split="test") for c in configs
        ]

    def generate_ngrams(self):
        for dataset in self._datasets:
            for idx, line in enumerate(dataset):
                try:
                    if "input" in line:
                        text = line["input"]
                        self._update_ngrams(
                            text, self._min_ngram_size, self._max_ngram_size
                        )
                except Exception:
                    logging.exception(f"Error processing line {idx}")
        return self.ngrams


class BBQ(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "bbq"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        configs = get_dataset_config_names("heegyu/bbq", trust_remote_code=True)
        self._datasets = []
        for c in configs:
            try:
                self._datasets.append(
                    load_dataset("heegyu/bbq", c, split="test", trust_remote_code=True)
                )
            except Exception:
                logging.warning(f"Skipping BBQ config '{c}': failed to load")

    def generate_ngrams(self):
        for dataset in self._datasets:
            for idx, line in enumerate(dataset):
                try:
                    text = line["question"]
                    self._update_ngrams(
                        text, self._min_ngram_size, self._max_ngram_size
                    )
                except Exception:
                    logging.exception(f"Error processing line {idx}")
        return self.ngrams


class BigCodeBench(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "bigcodebench"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("bigcode/bigcodebench", split="v0.1.4")

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["complete_prompt"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                text = line["canonical_solution"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class CoQA(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "coqa"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("EleutherAI/coqa", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                for question in line["questions"]["input_text"]:
                    self._update_ngrams(
                        question, self._min_ngram_size, self._max_ngram_size
                    )
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class CRUXEval(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "cruxeval"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("cruxeval-org/cruxeval", split="test")

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["code"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class CulturalBench(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "culturalbench"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset_hard = load_dataset(
            "kellycyy/CulturalBench", "CulturalBench-Hard", split="test"
        )
        self._dataset_easy = load_dataset(
            "kellycyy/CulturalBench", "CulturalBench-Easy", split="test"
        )

    def _index_ngrams(self, dataset, fields):
        for idx, line in enumerate(dataset):
            try:
                for field in fields:
                    self._update_ngrams(
                        line[field], self._min_ngram_size, self._max_ngram_size
                    )
            except Exception:
                logging.exception(f"Error processing line {idx}")

    def generate_ngrams(self):
        self._index_ngrams(self._dataset_hard, ["prompt_question"])
        self._index_ngrams(
            self._dataset_easy,
            ["prompt_question"],
        )
        return self.ngrams


class DeepMindMath(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "deepmind_math"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        configs = get_dataset_config_names("deepmind/math_dataset", trust_remote_code=True)
        self._datasets = [
            load_dataset(
                "deepmind/math_dataset", c, split="test", trust_remote_code=True
            )
            for c in configs
        ]

    def generate_ngrams(self):
        for dataset in self._datasets:
            for idx, line in enumerate(dataset):
                try:
                    text = line["question"]
                    self._update_ngrams(
                        text, self._min_ngram_size, self._max_ngram_size
                    )
                    text = line["answer"]
                    self._update_ngrams(
                        text, self._min_ngram_size, self._max_ngram_size
                    )
                except Exception:
                    logging.exception(f"Error processing line {idx}")
        return self.ngrams


class DeepSeekLeetCode(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "deepseek_leetcode"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("davidheineman/deepseek-leetcode", split="test")

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["prompt"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class DROP(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "drop"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("EleutherAI/drop", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class DS1000(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "ds1000"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("xlangai/DS-1000", split="test")

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["prompt"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class GSMSymbolic(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "gsm_symbolic"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        configs = ["p1", "p2"] # get_dataset_config_names("apple/GSM-Symbolic")
        self._datasets = [
            load_dataset("apple/GSM-Symbolic", c, split="test") for c in configs
        ]

    def generate_ngrams(self):
        for dataset in self._datasets:
            for idx, line in enumerate(dataset):
                try:
                    text = line["question"]
                    self._update_ngrams(
                        text, self._min_ngram_size, self._max_ngram_size
                    )
                except Exception:
                    logging.exception(f"Error processing line {idx}")
        return self.ngrams


class GSM8KPlatinum(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "gsm8k_platinum"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("madrylab/gsm8k-platinum", split="test")

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                text = line["answer"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class HELMET(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "helmet"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        configs = get_dataset_config_names("princeton-nlp/HELMET")
        self._datasets = [
            load_dataset("princeton-nlp/HELMET", c, split="train") for c in configs
        ]

    def generate_ngrams(self):
        for dataset in self._datasets:
            for idx, line in enumerate(dataset):
                try:
                    text = line["input"]
                    self._update_ngrams(
                        text, self._min_ngram_size, self._max_ngram_size
                    )
                except Exception:
                    logging.exception(f"Error processing line {idx}")
        return self.ngrams


class Jeopardy(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "jeopardy"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("soldni/jeopardy", "all_questions", split="train")

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class LABBench(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "lab_bench"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        configs = get_dataset_config_names("futurehouse/lab-bench")
        self._datasets = [
            load_dataset("futurehouse/lab-bench", c, split="train") for c in configs
        ]

    def generate_ngrams(self):
        for dataset in self._datasets:
            for idx, line in enumerate(dataset):
                try:
                    text = line["question"]
                    self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                except Exception:
                    logging.exception(f"Error processing line {idx}")
        return self.ngrams


class LBPP(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "lbpp"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("ellamind/lbpp", split="test")

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["instruction"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class LiveCodeBench(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "livecodebench"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "livecodebench/code_generation_lite", split="test", trust_remote_code=True
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question_content"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class AdvBench(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "advbench"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("walledai/AdvBench", split="train", token=os.getenv("HF_TOKEN"))

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["prompt"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams

class LongBenchV2(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "longbench_v2"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        configs = get_dataset_config_names("zai-org/LongBench-v2", trust_remote_code=True)
        self._datasets = [
            load_dataset(
                "zai-org/LongBench-v2", c, split="train", trust_remote_code=True
            )
            for c in configs
        ]

    def generate_ngrams(self):
        for dataset in self._datasets:
            for idx, line in enumerate(dataset):
                try:
                    text = line["question"]
                    self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                except Exception:
                    logging.exception(f"Error processing line {idx}")
        return self.ngrams


class LogiQA(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "logiqa"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("lucasmccabe/logiqa", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["query"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class Math500(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "math500"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["problem"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                text = line["solution"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class MathQA(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "mathqa"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "allenai/math_qa", split=split_type, trust_remote_code=True
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["Problem"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class MedMCQA(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "medmcqa"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("openlifescienceai/medmcqa", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                for field in ("question", "exp", "opa", "opb", "opc", "opd"):
                    if text := line.get(field):
                        self._update_ngrams(
                            text, self._min_ngram_size, self._max_ngram_size
                        )
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class MedQA(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "medqa"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("davidheineman/medqa-en", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                choices = line["choices"]
                for choice in choices:
                    self._update_ngrams(
                        choice, self._min_ngram_size, self._max_ngram_size
                    )
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class MMLU_CF(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "mmlu_cf"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        configs = get_dataset_config_names("edinburgh-dawg/mmlu-cf")
        self._datasets = [
            load_dataset("edinburgh-dawg/mmlu-cf", c, split=split_type) for c in configs
        ]

    def generate_ngrams(self):
        for dataset in self._datasets:
            for idx, line in enumerate(dataset):
                try:
                    text = line["question"]
                    self._update_ngrams(
                        text, self._min_ngram_size, self._max_ngram_size
                    )
                except Exception:
                    logging.exception(f"Error processing line {idx}")
        return self.ngrams


class MMLU_Pro(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "mmlu_pro"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("TIGER-Lab/MMLU-Pro", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class MMLU_Redux(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "mmlu_redux"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        configs = get_dataset_config_names("edinburgh-dawg/mmlu-redux")
        self._datasets = [
            load_dataset("edinburgh-dawg/mmlu-redux", c, split="test") for c in configs
        ]

    def generate_ngrams(self):
        for dataset in self._datasets:
            for idx, line in enumerate(dataset):
                try:
                    text = line["question"]
                    self._update_ngrams(
                        text, self._min_ngram_size, self._max_ngram_size
                    )
                except Exception:
                    logging.exception(f"Error processing line {idx}")
        return self.ngrams


class MT_MBPP(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "mt_mbpp"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        configs = get_dataset_config_names("allenai/multilingual_mbpp")
        self._datasets = [
            load_dataset("allenai/multilingual_mbpp", c, split=split_type)
            for c in configs
        ]

    def generate_ngrams(self):
        for dataset in self._datasets:
            for idx, line in enumerate(dataset):
                try:
                    text = line["text"]
                    self._update_ngrams(
                        text, self._min_ngram_size, self._max_ngram_size
                    )
                    text = line["code"]
                    self._update_ngrams(
                        text, self._min_ngram_size, self._max_ngram_size
                    )
                except Exception:
                    logging.exception(f"Error processing line {idx}")
        return self.ngrams


class MultiLoKo(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "multiloko"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "facebook/multiloko",
            "english",
            split=split_type,
            token=os.getenv("HF_TOKEN"),
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                text = line["target"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class MultiPL_E(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "multipl_e"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        configs = get_dataset_config_names("nuprl/MultiPL-E")
        self._datasets = [
            load_dataset("nuprl/MultiPL-E", c, split="test") for c in configs
        ]

    def generate_ngrams(self):
        for dataset in self._datasets:
            for idx, line in enumerate(dataset):
                try:
                    text = line["prompt"]
                    self._update_ngrams(
                        text, self._min_ngram_size, self._max_ngram_size
                    )
                except Exception:
                    logging.exception(f"Error processing line {idx}")
        return self.ngrams


class NaturalQuestions(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "natural_questions"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "google-research-datasets/nq_open", split=split_type
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class PubMedQA(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "pubmedqa"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "bigbio/pubmed_qa",
            split=split_type,
            token=os.getenv("HF_TOKEN"),
            trust_remote_code=True,
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["QUESTION"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                text = line["LONG_ANSWER"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class QASPERYesNo(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "qasper_yesno"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("allenai/qasper-yesno", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class RACE(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "race"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("ehovy/race", "all", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                options = line["options"]
                for option in options:
                    self._update_ngrams(
                        option, self._min_ngram_size, self._max_ngram_size
                    )
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class BigBench(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "bigbench"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        # configs = get_dataset_config_names("google/bigbench")
        _TASKS = [
            "qa_wikidata",
            "cs_algorithms",
            "dyck_languages",
            "language_identification",
            "operators",
            "repeat_copy_logic",
        ]
        self._datasets = [
            load_dataset("google/bigbench", c, split=split_type, trust_remote_code=True) for c in _TASKS
        ]

    def generate_ngrams(self):
        for dataset in self._datasets:
            for idx, line in enumerate(dataset):
                try:
                    text = line["inputs"]
                    self._update_ngrams(
                        text, self._min_ngram_size, self._max_ngram_size
                    )
                except Exception:
                    logging.exception(f"Error processing line {idx}")
        return self.ngrams


class SciQ(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "sciq"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("allenai/sciq", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class SciRIFFYesNo(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "sciriff_yesno"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("allenai/sciriff-yesno", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class SimpleQA(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "simpleqa"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("basicv8vc/SimpleQA", split="test")

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["problem"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class SocialIQA(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "social_iqa"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "allenai/social_i_qa", split=split_type, trust_remote_code=True
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class SQuAD(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "squad"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("rajpurkar/squad", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class SuperGPQA(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "supergpqa"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("m-a-p/SuperGPQA", split="train")

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class SVAMP(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "svamp"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("ChilleD/SVAMP", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["Body"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                text = line["Question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class TriviaQA(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "triviaqa"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("mandarjoshi/trivia_qa", "rc", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class Winogender(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "winogender"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("oskarvanderwal/winogender", "all", split="test")

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["sentence"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class WinogradSchemaChallenge(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "winograd_schema"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("winograd_wsc", "wsc273", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["text"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class WinoGrande(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "winogrande"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "allenai/winogrande", "winogrande_xl", split=split_type
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["sentence"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class AGIEval(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "agieval"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset_1 = load_dataset("hails/agieval-aqua-rat", split="test")
        self._dataset_2 = load_dataset("hails/agieval-gaokao-english", split="test")
        self._dataset_3 = load_dataset("hails/agieval-logiqa-en", split="test")
        self._dataset_4 = load_dataset("hails/agieval-lsat-ar", split="test")
        self._dataset_5 = load_dataset("hails/agieval-lsat-lr", split="test")
        self._dataset_6 = load_dataset("hails/agieval-lsat-rc", split="test")
        self._dataset_7 = load_dataset("hails/agieval-sat-en", split="test")
        self._dataset_8 = load_dataset("hails/agieval-sat-en-without-passage", split="test")
        self._dataset_9 = load_dataset("hails/agieval-sat-math", split="test")
        self._dataset_10 = load_dataset("hails/agieval-math", split="test")

        self._datasets = [
            self._dataset_1,
            self._dataset_2,
            self._dataset_3,
            self._dataset_4,
            self._dataset_5,
            self._dataset_6,
            self._dataset_7,
            self._dataset_8,
            self._dataset_9,
            self._dataset_10,
        ]

    def generate_ngrams(self):
        try:
            for _dataset in self._datasets:
                for idx, line in enumerate(_dataset):
                    text = line["query"]

                    match = re.search(r'Q:\s*(.*?)(?:\s*Answer Choices:|\s*A:)', text, re.DOTALL)
                    if match:
                        text = match.group(1).strip()

                    self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
        except Exception:
            logging.exception(f"Error processing line {idx}")
        return self.ngrams

