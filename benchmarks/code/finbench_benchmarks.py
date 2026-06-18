import re
from datasets import load_dataset
from nemo_curator.tasks.downstream_task import DownstreamTask
import logging
import os
from benchmarks.code.utils import general_detokenize


class FinnishBench_ARC_Challenge(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "arc_challenge_fi"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "TurkuNLP/finbenchv2-arc-c-fi-ht", split=split_type
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class FinnishBench_Belebele(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "belebele_fi"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "TurkuNLP/finbenchv2-belebele-fi-og", split=split_type
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                text = line["flores_passage"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class FinnishBench_GoldenSwag(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "goldenswag_fi"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "TurkuNLP/finbenchv2-goldenswag-fi-ht", split=split_type
        )
    
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


class FinnishBench_OpenGPT_X_TruthfulQAX(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "opengpt_x_truthfulqax_fi"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset_1 = load_dataset(
            "TurkuNLP/finbenchv2-opengpt-x_truthfulqax-fi-mt", "gen_FI", split=split_type
        )
        self._dataset_2 = load_dataset(
            "TurkuNLP/finbenchv2-opengpt-x_truthfulqax-fi-mt", "mc_FI", split=split_type
        )

        self._datasets = [
            self._dataset_1,
            self._dataset_2
        ]

    def generate_ngrams(self):
        for _dataset in self._datasets:
            for idx, line in enumerate(_dataset):
                try:
                    text = line["question"]
                    self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                except Exception:
                    logging.exception(f"Error processing line {idx}")
        return self.ngrams


class FinnishBench_Scandisent(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "scandisent_fi"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "TurkuNLP/finbenchv2-scandisent-fi-mini", split=split_type
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["text"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


## POSSIBLE PROBLEM WITH LICENSE
class FinnishBench_Squad_Strip(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "squad_strip_fi"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "", split=split_type
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line[""]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class FinnishBench_SIB_200(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "sib_200_fi"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "TurkuNLP/finbenchv2-sib-200-fi-og", split=split_type
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["text"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


## PROBLEM, IDK WTF TO DO WITH THIS ONE, I THINK I WONT INCLUDE IT
## ONLY WORDS IN THE DATASET
class FinnishBench_FBV1_Stripped(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "fbv1_stripped_fi"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "TurkuNLP/finbenchv2-fbv1-stripped-fi-ht", split=split_type
        )

        configs = get_dataset_config_names("TurkuNLP/finbenchv2-fbv1-stripped-fi-ht")
        self._dataset = concatenate_datasets(
            [
                load_dataset("TurkuNLP/finbenchv2-fbv1-stripped-fi-ht", c, split=split_type)
                for c in configs
            ]
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line[""]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams

