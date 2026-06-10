import json
from datasets import load_dataset
from nemo_curator.tasks.downstream_task import DownstreamTask
from pathlib import Path
import logging
import iso639
import yaml
from benchmarks.code.utils import general_detokenize

with open("benchmarks/code/langmaps.yaml", "r") as f:
    data = yaml.safe_load(f)

langs_script = data["langs_script"]
langs_mmmlu = data["langs_mmmlu"]
langs_global_mmlu = data["langs_global_mmlu"]
langs_include = data["langs_include"]
langs_xcopa = data["langs_xcopa"]
langs_polymath = data["langs_polymath"]
langs_xwinograd = data["langs_xwinograd"]
langs_mgsm = data["langs_mgsm"]
langs_belebele = data["langs_belebele"]
langs_multiblimp = data["langs_multiblimp"]
langs_marenahard = data["langs_marenahard"]
langs_mtbenchx = data["langs_mtbenchx"]
langs_multi_ifeval = data["langs_multiifeval"]
langs_xstorycloze = data["langs_xstorycloze"]
langs_xnli = data["langs_xnli"]
langs_pawsx = data["langs_pawsx"]


class MultiBlimp(DownstreamTask):
    def __init__(self, lang, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        if lang not in langs_multiblimp:
            raise Exception(f"Language not available")
        self._lang = iso639.Lang(lang).pt3
        self._task_name = f"multiblimp_{self._lang}"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("jumelet/multiblimp", self._lang, split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["sen"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                text = line["wrong_sen"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class Belebele(DownstreamTask):
    def __init__(self, lang, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()

        if lang not in langs_belebele:
            raise Exception(f"Language not available")
        lang_pt3 = iso639.Lang(lang).pt3
        self._lang = lang_pt3
        self._script = langs_script[self._lang]
        self._task_name = f"belebele_{self._lang}_{self._script}"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "facebook/belebele", f"{self._lang}_{self._script}", split=split_type
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


class MMMLU(DownstreamTask):
    def __init__(self, lang, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        if lang not in langs_mmmlu:
            raise Exception("Language not available")
        self._lang = lang
        self._task_name = f"mmmlu_{self._lang}"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "openai/MMMLU", langs_mmmlu[self._lang], split=split_type
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["Question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class Include(DownstreamTask):
    def __init__(self, lang, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        if lang not in langs_include:
            raise Exception("Language not available")
        if lang == "el":
            self._lang = "Greek"
        else:
            self._lang = iso639.Lang(lang).name
        self._task_name = f"include_{self._lang}"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "CohereLabs/include-base-44", self._lang, split=split_type
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class XStoryCloze(DownstreamTask):
    def __init__(self, lang, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        if lang not in langs_xstorycloze:
            raise Exception("Language not available")
        self._lang = lang
        self._task_name = f"xstory_cloze_{self._lang}"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "juletxara/xstory_cloze", self._lang, split=split_type
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


class XCopa(DownstreamTask):
    def __init__(self, lang, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        if lang not in langs_xcopa:
            raise Exception("Language not available")
        self._lang = lang
        self._task_name = f"xcopa_{self._lang}"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("cambridgeltl/xcopa", self._lang, split=split_type)
        self.xcopa_connectors = {
            "it": {"cause": "perché", "effect": "quindi"},
            "et": {"cause": "sest", "effect": "seetõttu"},
        }

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = (
                    line["premise"][:-1].strip()
                    + " "
                    + self.xcopa_connectors[self._lang][line["question"]]
                )
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class PolyMath(DownstreamTask):
    def __init__(self, lang, min_ngram_size=8, max_ngram_size=13):
        super().__init__()
        if lang not in langs_polymath:
            raise Exception("Language not available")
        self._lang = lang
        self._task_name = f"polymath_{self._lang}_all-splits"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size

        self._dataset_low = load_dataset("Qwen/PolyMath", self._lang, split="low")
        self._dataset_medium = load_dataset("Qwen/PolyMath", self._lang, split="medium")
        self._dataset_high = load_dataset("Qwen/PolyMath", self._lang, split="high")
        self._dataset_top = load_dataset("Qwen/PolyMath", self._lang, split="top")
        self._datasets = [
            self._dataset_low,
            self._dataset_medium,
            self._dataset_high,
            self._dataset_top,
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


class MGSM(DownstreamTask):
    def __init__(
        self,
        min_ngram_size=8,
        max_ngram_size=13,
        split_type=None,
        lang=None,
    ):
        super().__init__()
        if lang not in langs_mgsm:
            raise Exception("Language not available")
        self._task_name = f"mgsm_direct_{lang}"
        self._lang = lang
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("juletxara/mgsm", self._lang, split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class XWinograd(DownstreamTask):
    def __init__(self, lang, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        if lang not in langs_xwinograd:
            raise Exception("Language not available")
        self._lang = lang
        self._task_name = f"xwinograd_{self._lang}"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "Muennighoff/xwinograd", self._lang, split=split_type
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["sentence"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class XWinogradNoBlanks(DownstreamTask):
    def __init__(self, lang, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        if lang not in langs_xwinograd:
            raise Exception("Language not available")
        self._lang = lang
        self._task_name = f"xwinograd_{self._lang}"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "Muennighoff/xwinograd", self._lang, split=split_type
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                choices = [line["option1"], line["option2"]]
                answer_index = line["answer"]
                text = line["sentence"].replace("_", choices[int(answer_index) - 1])
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class m_ArenaHard(DownstreamTask):
    def __init__(
        self,
        min_ngram_size=8,
        max_ngram_size=13,
        split_type=None,
        lang: str = None,
    ):
        super().__init__()
        if lang not in langs_marenahard:
            raise Exception(f"Language not available")
        self._task_name = "m-arenahard"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("CohereLabs/m-ArenaHard", lang, split=split_type)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["prompt"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception as e:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class MTBench_X(DownstreamTask):
    def __init__(self, lang, file_dir, min_ngram_size=8, max_ngram_size=13):
        super().__init__()

        if lang not in langs_mtbenchx:
            raise Exception("Language not available")
        self._task_name = f"mtbench-x_{lang}"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._file_dir = file_dir

        self._file_path = Path(file_dir, f"mt_bench_{lang.upper()}", "question.jsonl")
        if self._file_path is None:
            raise Exception("Must provide path to data in JSONL format")

    def generate_ngrams(self):
        with open(self._file_path, "r") as f:
            for idx, line in enumerate(f):
                try:
                    texts = json.loads(line)["turns"]
                    for text in texts:
                        self._update_ngrams(
                            text, self._min_ngram_size, self._max_ngram_size
                        )
                except Exception as e:
                    logging.exception(f"Error processing line {idx}")
            return self.ngrams


class Multi_IFEval(DownstreamTask):
    def __init__(self, lang, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        if lang not in langs_multi_ifeval:
            raise Exception("Language not available")
        self._task_name = "multi_ifeval"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size

        self._lang = lang
        self._dataset = load_dataset("facebook/Multi-IF", split=split_type)

    def generate_ngrams(self):
        full_lang_name = iso639.Lang(self._lang).name
        for idx, line in enumerate(self._dataset):
            try:
                if line["language"] == full_lang_name:
                    texts = [
                        line["turn_1_prompt"],
                        line["turn_2_prompt"],
                        line["turn_3_prompt"],
                    ]
                    for text in texts:
                        if text is not None:
                            text = json.loads(text)["content"]
                            self._update_ngrams(
                                text, self._min_ngram_size, self._max_ngram_size
                            )
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class GlobalMMLU_Full(DownstreamTask):
    def __init__(self, lang, min_ngram_size=8, max_ngram_size=13, split_type=None):
        super().__init__()
        if lang not in langs_global_mmlu:
            raise Exception("Language not available")
        self._task_name = f"global_mmlu_{lang}"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._lang = lang
        self._dataset = load_dataset(
            "CohereLabs/Global-MMLU", self._lang, split=split_type
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"].strip()
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception as e:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class Flores200(DownstreamTask):
    def __init__(
        self,
        min_ngram_size=8,
        max_ngram_size=13,
        split_type=None,
        lang=None,
    ):
        super().__init__()
        self._task_name = f"flores200_{lang}"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        if lang.lower() == "lv":
            self._lang = "lvs"
        else:
            self._lang = iso639.Lang(lang).pt3
        self._script = langs_script[self._lang]
        self._dataset = load_dataset(
            "Muennighoff/flores200",
            f"{self._lang}_{self._script}",
            split=split_type,
            trust_remote_code=True,
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line[f"sentence"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class PAWS_x(DownstreamTask):
    def __init__(
        self,
        min_ngram_size=8,
        max_ngram_size=13,
        split_type=None,
        lang=None,
    ):
        super().__init__()
        if lang not in langs_pawsx:
            raise Exception(f"Language not available")
        self._task_name = f"paws_{lang}"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._lang = lang
        self._dataset = load_dataset(
            "paws-x",
            self._lang,
            split=split_type,
            trust_remote_code=True,
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = general_detokenize(line[f"sentence1"])
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                text = general_detokenize(line[f"sentence2"])
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class XNLI(DownstreamTask):
    def __init__(
        self,
        min_ngram_size=8,
        max_ngram_size=13,
        split_type=None,
        lang=None,
    ):
        super().__init__()
        if lang not in langs_xnli:
            raise Exception(f"Language not available")
        self._task_name = f"xnli_{lang}"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._lang = lang
        self._dataset = load_dataset(
            "xnli",
            self._lang,
            split=split_type,
            trust_remote_code=True,
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = general_detokenize(line[f"premise"])
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                text = general_detokenize(line[f"hypothesis"])
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams
