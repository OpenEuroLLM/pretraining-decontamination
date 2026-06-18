import logging
import re

from datasets import load_dataset
from nemo_curator.tasks.downstream_task import DownstreamTask


class SpanishBench_Cocoteros(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "cocoteros_es"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("gplsi/cocoteros", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                keywords = line["keywords"]
                context = line["context"]
                text = f"Genera una frase corta con estas palabras: {keywords}. El contexto es: {context}"
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class SpanishBench_COPA(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "copa_es"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("BSC-LT/COPA-es", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = (
                    line["premise"][:-1].strip()
                    + " "
                    + {"cause": "porque", "effect": "y por lo tanto"}[line["question"]]
                )
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class SpanishBench_ESCOLA(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "escola_es"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("nbel/EsCoLA", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                sentence = line["Sentence"]
                text = f"{sentence}\nPregunta: ¿Tiene sentido esta frase?\nRespuesta:"
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class SpanishBench_OpenBookQA(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "openbookqa_es"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("BSC-LT/openbookqa-es", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question_stem"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class SpanishBench_WNLI(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "wnli_es"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("PlanTL-GOB-ES/wnli-es", split=split_type, trust_remote_code=True)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["sentence1"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                text = line["sentence2"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class SpanishBench_XLSUM(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "xlsum_es"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("csebuetnlp/xlsum", "spanish", split=split_type, trust_remote_code=True)

    def generate_ngrams(self):
        # https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/spanish_bench/utils.py
        for idx, line in enumerate(self._dataset):
            try:
                text = re.sub(r" +", " ", line["title"])
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                text = re.sub(r" +", " ", line["text"])
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                text = re.sub(r" +", " ", line["summary"])
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class SpanishBench_Phrases(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "phrases_es"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("gplsi/ES-VA_translation_test", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                premise = line["es"]
                self._update_ngrams(premise, self._min_ngram_size, self._max_ngram_size)
                premise = line["va"]
                self._update_ngrams(premise, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class SpanishBench_XQUAD(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "xquad_es"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("google/xquad", f"xquad.es", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                context = line["context"]
                self._update_ngrams(context, self._min_ngram_size, self._max_ngram_size)
                question = line["question"]
                self._update_ngrams(
                    question, self._min_ngram_size, self._max_ngram_size
                )
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams
