import csv
import json
import logging
import os
import re

import polars as pl
from nemo_curator.tasks.downstream_task import DownstreamTask

from datasets import concatenate_datasets, get_dataset_config_names, load_dataset


class NorEcSentence(DownstreamTask):
    def __init__(
        self,
        min_ngram_size=8,
        max_ngram_size=13,
        split_type=None,
    ):
        super().__init__()
        self._task_name = "norecsentence"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        configs = get_dataset_config_names("ltg/norec_sentence")
        self._dataset = concatenate_datasets(
            [load_dataset("ltg/norec_sentence", c, split=split_type) for c in configs]
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["review"].strip()
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class NorEcDocument(DownstreamTask):
    def __init__(
        self,
        min_ngram_size=8,
        max_ngram_size=13,
        split_type=None,
    ):
        super().__init__()
        self._task_name = "norecdocument"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        configs = get_dataset_config_names("ltg/norec_document")
        self._dataset = concatenate_datasets(
            [load_dataset("ltg/norec_document", c, split=split_type) for c in configs]
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["review"].strip()
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class NCB(DownstreamTask):
    def __init__(
        self,
        min_ngram_size=8,
        max_ngram_size=13,
    ):
        super().__init__()
        self._task_name = "NCB"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("hcfa/ncb", split="train")

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["correct"].strip()
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                text = line["wrong"].strip()
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class NorIdiom(DownstreamTask):
    def __init__(
        self,
        min_ngram_size=8,
        max_ngram_size=13,
    ):
        super().__init__()
        self._task_name = "NorIdiom"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._datasets = [
            load_dataset("Sprakbanken/Norwegian_idioms", "default", split="train"),
            load_dataset(
                "Sprakbanken/Norwegian_idioms",
                "include_translated_idioms",
                split="train",
            ),
        ]

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["idiom_start"].strip()
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                for completion in line["accepted_completions"]:
                    self._update_ngrams(
                        completion, self._min_ngram_size, self._max_ngram_size
                    )
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class NorBelebele(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "nor_belebele"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("facebook/belebele", "nob_Latn", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                self._update_ngrams(
                    line["question"], self._min_ngram_size, self._max_ngram_size
                )
                self._update_ngrams(
                    line["flores_passage"], self._min_ngram_size, self._max_ngram_size
                )
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class NRKQuizQA(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13):
        super().__init__()
        self._task_name = "nrk_quiz_qa"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = concatenate_datasets(
            [
                load_dataset("ltg/nrk_quiz_qa", "nb", split="test"),
                load_dataset("ltg/nrk_quiz_qa", "nn", split="test"),
            ]
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                self._update_ngrams(
                    line["question"], self._min_ngram_size, self._max_ngram_size
                )
                for choice in line["choices"]["text"]:
                    self._update_ngrams(
                        choice, self._min_ngram_size, self._max_ngram_size
                    )
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class NorOpenBookQA(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "noropenbookqa"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = concatenate_datasets(
            [
                load_dataset("ltg/noropenbookqa", c, split=split_type)
                for c in ["nb", "nn"]
            ]
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                self._update_ngrams(
                    line["question_stem"], self._min_ngram_size, self._max_ngram_size
                )
                self._update_ngrams(
                    line["fact"], self._min_ngram_size, self._max_ngram_size
                )
                for choice in line["choices"]["text"]:
                    self._update_ngrams(
                        choice, self._min_ngram_size, self._max_ngram_size
                    )
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class NorCommonsenseQA(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type="train"):
        super().__init__()
        self._task_name = "norcommonsenseqa"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = concatenate_datasets(
            [
                load_dataset("ltg/norcommonsenseqa", "nb", split=split_type),
                load_dataset("ltg/norcommonsenseqa", "nn", split=split_type),
            ]
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                self._update_ngrams(
                    line["question"], self._min_ngram_size, self._max_ngram_size
                )
                for choice in line["choices"]["text"]:
                    self._update_ngrams(
                        choice, self._min_ngram_size, self._max_ngram_size
                    )
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class NorTruthfulQAMC(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type="validation"):
        super().__init__()
        self._task_name = "nortruthfulqa_mc"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = concatenate_datasets(
            [
                load_dataset("ltg/nortruthfulqa_mc", "nb", split=split_type),
                load_dataset("ltg/nortruthfulqa_mc", "nn", split=split_type),
            ]
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                self._update_ngrams(
                    line["question"], self._min_ngram_size, self._max_ngram_size
                )
                for choice in line["mc1_targets"]["choices"]:
                    self._update_ngrams(
                        choice, self._min_ngram_size, self._max_ngram_size
                    )
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class NorQuAD(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "norquad"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("ltg/norquad", "default", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                self._update_ngrams(
                    line["context"], self._min_ngram_size, self._max_ngram_size
                )
                self._update_ngrams(
                    line["question"], self._min_ngram_size, self._max_ngram_size
                )
                answers = line["answers"]["text"]
                for answer in answers:
                    self._update_ngrams(
                        answer, self._min_ngram_size, self._max_ngram_size
                    )
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class NorTruthfulQAGen(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type="validation"):
        super().__init__()
        self._task_name = "nortruthfulqa_gen"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = concatenate_datasets(
            [
                load_dataset("ltg/nortruthfulqa_gen", "nb", split=split_type),
                load_dataset("ltg/nortruthfulqa_gen", "nn", split=split_type),
            ]
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                self._update_ngrams(
                    line["question"], self._min_ngram_size, self._max_ngram_size
                )
                self._update_ngrams(
                    line["best_answer"], self._min_ngram_size, self._max_ngram_size
                )
                for answer in line["correct_answers"]:
                    self._update_ngrams(
                        answer, self._min_ngram_size, self._max_ngram_size
                    )
                for answer in line["incorrect_answers"]:
                    self._update_ngrams(
                        answer, self._min_ngram_size, self._max_ngram_size
                    )
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class AskGEC(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "ask_gec"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("ltg/ask-gec", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                self._update_ngrams(
                    line["source"], self._min_ngram_size, self._max_ngram_size
                )
                self._update_ngrams(
                    line["correction"], self._min_ngram_size, self._max_ngram_size
                )
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class NorSumm(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "norsumm"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = concatenate_datasets(
            [load_dataset("SamiaT/NorSumm", c, split=split_type) for c in ["nb", "nn"]]
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                self._update_ngrams(
                    line["article"], self._min_ngram_size, self._max_ngram_size
                )
                for summary in line["summaries"]:
                    self._update_ngrams(
                        summary, self._min_ngram_size, self._max_ngram_size
                    )
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class NorRewriteInstruct(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type="test"):
        super().__init__()
        self._task_name = "norrewrite_instruct"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "ltg/norrewrite-instruct", "default", split=split_type
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                self._update_ngrams(
                    line["prompt"], self._min_ngram_size, self._max_ngram_size
                )
                self._update_ngrams(
                    line["context"], self._min_ngram_size, self._max_ngram_size
                )
                self._update_ngrams(
                    line["response"], self._min_ngram_size, self._max_ngram_size
                )
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class NorSummarizeInstruct(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type="test"):
        super().__init__()
        self._task_name = "norsummarize_instruct"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "ltg/norsummarize-instruct", "default", split=split_type
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                self._update_ngrams(
                    line["prompt"], self._min_ngram_size, self._max_ngram_size
                )
                self._update_ngrams(
                    line["context"], self._min_ngram_size, self._max_ngram_size
                )
                self._update_ngrams(
                    line["response"], self._min_ngram_size, self._max_ngram_size
                )
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class TatoebaEngNor(DownstreamTask):
    # English → Bokmål/Nynorsk: extract the Norwegian targetString
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "tatoeba_eng_nor"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = concatenate_datasets(
            [
                load_dataset(
                    "Helsinki-NLP/tatoeba_mt",
                    pair,
                    split=split_type,
                    token=os.getenv("HF_TOKEN"),
                    trust_remote_code=True,
                )
                for pair in ["eng-nob", "eng-nno", "eng-nor"]
            ]
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                self._update_ngrams(
                    line["targetString"], self._min_ngram_size, self._max_ngram_size
                )
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class TatoebaNorEng(DownstreamTask):
    # Bokmål/Nynorsk as source language: extract the Norwegian sourceString
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "tatoeba_nor_eng"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        nor_pairs = [
            "nob-nno",
            "nob-rus",
            "nob-spa",
            "nob-swe",
            "nor-pol",
            "nor-por",
            "nor-rus",
            "nor-spa",
            "nor-swe",
            "nor-ukr",
            "nor-zho",
            "nno-nob",
        ]
        self._dataset = concatenate_datasets(
            [
                load_dataset(
                    "Helsinki-NLP/tatoeba_mt",
                    pair,
                    split=split_type,
                    token=os.getenv("HF_TOKEN"),
                    trust_remote_code=True,
                )
                for pair in nor_pairs
            ]
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                self._update_ngrams(
                    line["sourceString"], self._min_ngram_size, self._max_ngram_size
                )
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams
