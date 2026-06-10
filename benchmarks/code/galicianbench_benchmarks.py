import logging

from datasets import load_dataset
from nemo_curator.tasks.downstream_task import DownstreamTask


class GalicianBench_Belebele(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "belebele_gl"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("proxectonos/belebele_gl", split=split_type)

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


class GalicianBench_Galcola(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "galcola_gl"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("proxectonos/galcola", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["sentence"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class GalicianBench_MGSM(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "mgsm_direct_gl"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("proxectonos/mgsm_gl", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class GalicianBench_OpenBookQA(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "openbookqa_gl"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("proxectonos/openbookqa_gl", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question_stem"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class GalicianBench_Parafrases(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "parafrases_gl"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("proxectonos/parafrases_gl", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                frase = line["Frase"]
                self._update_ngrams(frase, self._min_ngram_size, self._max_ngram_size)
                parafrase = line["Paráfrase"]
                self._update_ngrams(
                    parafrase, self._min_ngram_size, self._max_ngram_size
                )
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class GalicianBench_PAWS(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "paws_gl"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("proxectonos/PAWS-gl", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                sentence1 = line["sentence1"]
                self._update_ngrams(
                    sentence1, self._min_ngram_size, self._max_ngram_size
                )
                sentence2 = line["sentence2"]
                self._update_ngrams(
                    sentence2, self._min_ngram_size, self._max_ngram_size
                )
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class GalicianBench_Summarization(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "summarization_gl"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("proxectonos/summarization_gl", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["text"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                text = line["summary"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class GalicianBench_Truthfulqa(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "truthfulqa_gl"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "proxectonos/truthfulqa_gl", "generation", split=split_type
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class GalicianBench_XNLI(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "xnli_gl"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("proxectonos/xnli_gl", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                sentence1 = line["sentence1"]
                self._update_ngrams(
                    sentence1, self._min_ngram_size, self._max_ngram_size
                )
                sentence2 = line["sentence2"]
                self._update_ngrams(
                    sentence2, self._min_ngram_size, self._max_ngram_size
                )
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class GalicianBench_XStoryCloze(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "xstorycloze_gl"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("proxectonos/xstorycloze_gl", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                texts = [
                    line["InputSentence1"],
                    line["InputSentence2"],
                    line["InputSentence3"],
                    line["InputSentence4"],
                ]
                text = " ".join(texts)
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams
