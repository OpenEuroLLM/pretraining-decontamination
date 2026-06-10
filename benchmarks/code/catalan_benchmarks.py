import re
from datasets import load_dataset
from nemo_curator.tasks.downstream_task import DownstreamTask
import logging
import os
from benchmarks.code.utils import general_detokenize


class CatalanBench_ArcEasy(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "arcchallenge_ca"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "projecte-aina/arc_ca", "ARC-Easy", split=split_type
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class CatalanBench_ArcChallenge(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "arcchallenge_ca"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "projecte-aina/arc_ca", "ARC-Challenge", split=split_type
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class CatalanBench_Cabreu(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "cabreu_ca"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("projecte-aina/caBreu", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = re.sub(r" +", " ", line["title"])
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                text = re.sub(r" +", " ", line["subtitle"])
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                text = re.sub(r" +", " ", line["content"])
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                for _, summaries in line["summaries"].items():
                    for _, summary in summaries.items():
                        text = re.sub(r" +", " ", summary)
                        self._update_ngrams(
                            text, self._min_ngram_size, self._max_ngram_size
                        )
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class CatalanBench_CatalanQA(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "catalanqa_ca"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("projecte-aina/catalanqa", split=split_type)

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


class CatalanBench_Catcola(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "catalanqa_ca"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("nbel/CatCoLA", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["Sentence"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class CatalanBench_Cocoteros(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "cocoteros_ca"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "gplsi/cocoteros_va", split=split_type, token=os.getenv("HF_TOKEN")
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                keywords = line["keywords"]
                context = line["context"]
                text = f"Genera una frase curta amb estes paraules: {keywords}. El context és: {context}"
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class CatalanBench_COPA(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "copa_ca"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("projecte-aina/COPA-ca", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = (
                    line["premise"][:-1].strip()
                    + " "
                    + {"cause": "perquè", "effect": "i per tant"}[line["question"]]
                )
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class CatalanBench_CoqCat(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "coqca"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("projecte-aina/CoQCat", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["story"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                questions = " ".join(line["questions"])
                self._update_ngrams(
                    questions, self._min_ngram_size, self._max_ngram_size
                )
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class CatalanBench_MGSM(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "mgsm_ca"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("projecte-aina/mgsm_ca", split=split_type)

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


class CatalanBench_OpenBookQA(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "openbookqa_ca"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("projecte-aina/openbookqa_ca", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question_stem"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class CatalanBench_Parafraseja(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "parafraseja_ca"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("projecte-aina/Parafraseja", split=split_type)

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


class CatalanBench_PAWS(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "paws_ca"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("projecte-aina/PAWS-ca", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                sentence1 = general_detokenize(line["sentence1"])
                self._update_ngrams(
                    sentence1, self._min_ngram_size, self._max_ngram_size
                )
                sentence2 = general_detokenize(line["sentence2"])
                self._update_ngrams(
                    sentence2, self._min_ngram_size, self._max_ngram_size
                )
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class CatalanBench_PIQA(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "piqa_ca"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("projecte-aina/piqa_ca", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["goal"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class CatalanBench_SIQA(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "siqa_ca"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("projecte-aina/siqa_ca", split=split_type)

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


class CatalanBench_TECA(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "teca_ca"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("projecte-aina/teca", split=split_type)

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


class CatalanBench_WNLI(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "wnli_ca"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("projecte-aina/wnli-ca", split=split_type)

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


class CatalanBench_XNLI(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "xnli_ca"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("projecte-aina/xnli-ca", split=split_type)

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


class CatalanBench_XNLIVA(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "xnli_va"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("gplsi/xnli_va", split=split_type)

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


class CatalanBench_XQUAD(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "xquad_ca"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("projecte-aina/xquad-ca", split=split_type)

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


class CatalanBench_XStoryCloze(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "xstory_cloze_ca"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "projecte-aina/xstorycloze_ca", "ca", split=split_type
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                texts = [
                    line["input_sentence_1"],
                    line["input_sentence_2"],
                    line["input_sentence_3"],
                    line["input_sentence_4"],
                ]
                text = " ".join(texts)
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class CatalanBench_PhrasesCAVA(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        self._task_name = "phrases_cava"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("gplsi/CA-VA_alignment_test", split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                catalan = line["ca"]
                self._update_ngrams(catalan, self._min_ngram_size, self._max_ngram_size)
                valencian = line["va"]
                self._update_ngrams(
                    valencian, self._min_ngram_size, self._max_ngram_size
                )
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams
