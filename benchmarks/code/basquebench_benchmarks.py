import logging
from datasets import get_dataset_config_names, load_dataset
from nemo_curator.tasks.downstream_task import DownstreamTask


class BasqueBench_XNLI(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "xnli_eu"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("HiTZ/xnli-eu", "eu", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                premise = line["premise"]
                self._update_ngrams(premise, self._min_ngram_size, self._max_ngram_size)
                hypothesis = line["hypothesis"]
                self._update_ngrams(
                    hypothesis, self._min_ngram_size, self._max_ngram_size
                )
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class BasqueBench_ArcEasy(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "arceasy_eu"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("HiTZ/ARC-eu", "ARC-Easy", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class BasqueBench_ArcChallenge(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "arcchallenge_eu"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("HiTZ/ARC-eu", "ARC-Challenge", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class BasqueBench_EusExams(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "eusexams_eu"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        eu_subsets = get_dataset_config_names("HiTZ/EusExams")
        self._eu_subsets = list(filter(lambda x: "eu_" in x, eu_subsets))
        self._split_type = split_type
        logging.info(f"Subsets for EusExams: {self._eu_subsets}")

    def generate_ngrams(self):
        for subset in self._eu_subsets:
            dataset = load_dataset("HiTZ/EusExams", subset, split=self._split_type)
            for idx, line in enumerate(dataset):
                try:
                    text = line["question"]
                    self._update_ngrams(
                        text, self._min_ngram_size, self._max_ngram_size
                    )
                except Exception:
                    logging.exception(f"Error processing line {idx}")
        return self.ngrams


class BasqueBench_EusProficiency(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "eusproficiency_eu"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("HiTZ/EusProficiency", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                # Fill in the blanks
                if "....." in line["question"]:
                    solution = line["candidates"][line["answer"] - 1]
                    if " · " in solution:
                        parts = solution.split(" · ")
                        for part in parts:
                            text = line["question"].replace("....", part, 1)
                    else:
                        text = line["question"].replace(".....", solution)
                    self._update_ngrams(
                        text, self._min_ngram_size, self._max_ngram_size
                    )
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class BasqueBench_EusReading(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "eusreading_eu"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("HiTZ/EusReading", "default", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                text = line["context"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class BasqueBench_EusTrivia(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "eustrivia_eu"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("HiTZ/EusTrivia", "default", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class BasqueBench_MGSM(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "mgsm_eu"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("HiTZ/MGSM-eu", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class BasqueBench_PAWS(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "paws_eu"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("HiTZ/PAWS-eu", split=split_type)

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


class BasqueBench_PIQA(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "piqa_eu"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("HiTZ/PIQA-eu", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["goal"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class BasqueBench_WNLI(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "wnli_eu"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("HiTZ/wnli-eu", split=split_type)

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


class BasqueBench_XCOPA(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "xcopa_eu"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("HiTZ/XCOPA-eu", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = (
                    line["premise"][:-1].strip()
                    + " "
                    + {"cause": " Izan ere,", "effect": " Beraz,"}[line["question"]]
                )
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class BasqueBench_EusTrivia(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "eus_trivia"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("HiTZ/EusTrivia", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class BasqueBench_QNLI(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "qnli_eu"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("orai-nlp/basqueGLUE", "qnli", split=split_type, trust_remote_code=True)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"] + " " + line["sentence"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams
