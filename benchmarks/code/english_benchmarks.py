import json
import logging
import os
import re

import polars as pl
from nemo_curator.tasks.downstream_task import DownstreamTask

from datasets import load_dataset


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
        self._dataset = load_dataset("super_glue", "copa", split=split_type)

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
        self._dataset = load_dataset("openbookqa", "main", split=split_type)

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
        self._dataset = load_dataset("ai2_arc", "ARC-Challenge", split=split_type)

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
        self._dataset = load_dataset("ai2_arc", "ARC-Easy", split=split_type)

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
        self._dataset = load_dataset("super_glue", "boolq", split=split_type)

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
        self._dataset = load_dataset(
            "evalplus/mbppplus", split=split_type
        )

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
