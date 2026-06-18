import csv
import json
import logging
import os
import re
from pathlib import Path
from huggingface_hub import hf_hub_download
from datasets import Dataset
import iso639
import yaml
from benchmarks.code.utils import general_detokenize
from nemo_curator.tasks.downstream_task import DownstreamTask
import os
os.environ["HF_TOKEN"] = "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"

from datasets import concatenate_datasets, get_dataset_config_names, load_dataset

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
langs_oneruler = data["langs_oneruler"]
langs_doclevel_mt = data["langs_doclevel_mt"]
langs_arc_challenge_mt = data["langs_arc_challenge_mt"]
langs_hellaswag_mt = data["langs_hellaswag_mt"]
langs_xcsqa = data["langs_xcsqa"]
langs_global_piqa = data["langs_global_piqa"]
langs_global_mgsm = data["langs_global_mgsm"]
langs_sib200 = data["langs_sib200"]
langs_mmlu_pro_x = data["langs_mmlu_pro_x"]
langs_wikiann = data["langs_wikiann"]
langs_m_dolly_mt = data["langs_m_dolly_mt"]
langs_aya_redteaming = data["langs_aya_redteaming"]
langs_marenahard_v2 = data["langs_marenahard_v2"]
langs_marenahard_v21 = data["langs_marenahard_v21"]
langs_multijail = data["langs_multijail"]
langs_blend = data["langs_blend"]
langs_tatoebamt = data["langs_tatoebamt"]
langs_aya_evaluation_suite = data["langs_aya_evaluation_suite"]
langs_maime2025 = data["langs_maime2025"]
langs_maime2026 = data["langs_maime2026"]


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
            "maximedb/paws-x-all",
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
            "facebook/xnli",
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


class mAIME2026(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, lang=None):
        super().__init__()
        self._task_name = "maime2026"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        if lang not in langs_maime2026:
            raise Exception("Language not available")
        self._lang = lang
        self._dataset = load_dataset(
            "LumiOpen/mAIME2026", self._lang + "_combined", split="test"
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class mAIME2025(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, lang=None):
        super().__init__()
        self._task_name = "maime2025"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        if lang not in langs_maime2025:
            raise Exception("Language not available")
        self._lang = lang
        self._dataset = load_dataset(
            "LumiOpen/mAIME2025", self._lang + "_combined", split="test"
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams

class IFEval(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, file_dir=None):
        super().__init__()
        self._task_name = "ifeval"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._file_dir = file_dir
        self._file_path = Path(file_dir, "input_data.jsonl")

    def generate_ngrams(self):
        with open(self._file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for idx, item in enumerate(data):
                try:
                    text = item["prompt"].strip()
                    self._update_ngrams(
                        text, self._min_ngram_size, self._max_ngram_size
                    )
                except Exception:
                    logging.exception(f"Error processing line {idx}")


class ComparIA_FR(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13):
        super().__init__()
        self._task_name = "comparia_fr"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "ministere-culture/comparia-conversations", split="train"
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["opening_msg"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                text = line["system_prompt_a"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                text = line["system_prompt_b"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                text = line["conversation_a"]
                for item in text.items():
                    self._update_ngrams(
                        item["content"], self._min_ngram_size, self._max_ngram_size
                    )
                text = line["conversation_b"]
                for item in text.items():
                    self._update_ngrams(
                        item["content"], self._min_ngram_size, self._max_ngram_size
                    )
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class OneRuler(DownstreamTask):
    CONTEXT_SIZES = ["4096", "8192", "16384"]

    def __init__(
        self, min_ngram_size=8, max_ngram_size=13, lang=None, context_size=None
    ):
        super().__init__()
        if lang not in langs_oneruler:
            raise Exception("Language not available")
        self._task_name = f"one_ruler_{lang}"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        sizes = [context_size] if context_size else self.CONTEXT_SIZES
        self._dataset = load_dataset("starbix/oneruler", f"{lang}_16384", split="test")
  

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                # Word doc says that it is not necessary to include context
                # text = line["context"]
                # self._update_ngrams(
                #     text, self._min_ngram_size, self._max_ngram_size
                # )
                text = line["question"]
                self._update_ngrams(
                    text, self._min_ngram_size, self._max_ngram_size
                )
                text = line["answer_prefix"]
                self._update_ngrams(
                    text, self._min_ngram_size, self._max_ngram_size
                )
                answers = line["answer"]
                for text in answers:
                    self._update_ngrams(
                        text, self._min_ngram_size, self._max_ngram_size
                    ) 
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class ArcChallengeMt(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, lang=None, split_type=None):
        super().__init__()
        if lang not in langs_arc_challenge_mt:
            raise Exception("Language not available")
        self._task_name = "arc_challenge_mt"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "LumiOpen/arc_challenge_mt", lang, split=split_type
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class HellaSwagMt(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, lang=None, split_type=None):
        super().__init__()
        if lang not in langs_hellaswag_mt:
            raise Exception("Language not available")
        self._task_name = "hellaswag_mt"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("jon-tow/okapi_hellaswag", lang, split=split_type)

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


class Xcsqa(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, lang=None, split_type=None):
        super().__init__()
        if lang not in langs_xcsqa:
            raise Exception("Language not available")
        self._task_name = "xcsqa"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "INK-USC/xcsr", f"X-CSQA-{lang}", split=split_type, trust_remote_code=True
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]["stem"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                texts = line["question"]["choices"]["text"]
                for text in texts:
                    self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class GlobalPiqa(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, lang=None, split_type=None):
        super().__init__()

        std_pt3 = iso639.Lang(lang).pt3
        self._script = langs_script[std_pt3].lower()  # globalpiqa uses lowercase script names

        lang_overrides = {
            "est": "ekk",  # Standard Estonian
            "lav": "lvs",  # Standard Latvian
            "sqi": "als",  # Tosk Albanian
        }

        self._lang = lang_overrides.get(std_pt3, std_pt3)
        self._full_lang = f"{self._lang}_{self._script}"

        if "fra" in self._full_lang:
            self._full_lang += "_fran"
        elif "por" in self._full_lang:
            self._full_lang += "_port"
        elif "spa" in self._full_lang:
            self._full_lang += "_spai"

        self._task_name = "global_piqa_" + self._full_lang

        if self._full_lang not in langs_global_piqa:
            raise Exception("Language not available")

        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "mrlbenchmarks/global-piqa-nonparallel",
            self._full_lang,
            split="test",
            trust_remote_code=True,
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["prompt"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                text = line["solution0"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                text = line["solution1"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class OpenSubtitles(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, lang=None):
        super().__init__()
        self._task_name = "open_subtitles"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        arrow_path = hf_hub_download(
            repo_id="Helsinki-NLP/OpenSubtitles2024-40-langs-15-movies",
            filename="devtest/data-00000-of-00001.arrow",
            repo_type="dataset",
            token=os.getenv("HF_TOKEN")
        )
        self._lang = lang
        self._dataset = Dataset.from_file(arrow_path)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                langs = line["translation"].keys()
                found_langs = [lang for lang in langs if lang.startswith(self._lang)]
                if not found_langs:
                    raise Exception(f"Language {self._lang} not found in line {idx}")
                for lang in found_langs:
                    text = line["translation"][lang]
                    self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class GlobalMGSM(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, lang=None):
        super().__init__()
        if lang not in langs_global_mgsm:
            raise Exception("Language not available")
        self._task_name = "global_mgsm"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "CohereLabs/Global-mgsm", lang, split="test", trust_remote_code=True
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                text = line["instruction"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception: 
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


TATOEBA_LANGMAP = {
    # Germanic
    'afr': 'afr-eng',
    'dan': 'dan-eng',
    'deu': 'deu-eng',
    'eng': 'eng-fra',
    'frr': 'deu-frr',
    'fry': 'eng-fry',
    'gos': 'eng-gos',
    'hrx': 'eng-hrx',
    'isl': 'eng-isl',
    'ltz': 'eng-ltz',
    'nds': 'eng-nds',
    'nld': 'eng-nld',
    'nob': 'eng-nob',
    'nno': 'eng-nno',
    'nor': 'eng-nor',
    'swe': 'eng-swe',
    'swg': 'deu-swg',
    'yid': 'eng-yid',
    # Romance
    'cat': 'cat-eng',
    'egl': 'egl-ita',
    'fra': 'eng-fra',
    'gcf': 'fra-gcf',
    'glg': 'eng-glg',
    'ita': 'eng-ita',
    'lad': 'eng-lad',
    'lad_Latn': 'eng-lad_Latn',
    'lat': 'eng-lat',
    'lat_Latn': 'lat_Latn-por',
    'oci': 'eng-oci',
    'pcd': 'fra-pcd',
    'pms': 'eng-pms',
    'por': 'eng-por',
    'ron': 'eng-ron',
    'spa': 'eng-spa',
    # Slavic
    'bel': 'bel-eng',
    'bos_Latn': 'eng-bos_Latn',
    'bul': 'bul-eng',
    'ces': 'ces-eng',
    'dsb': 'dsb-hsb',
    'hbs': 'eng-hbs',
    'hrv': 'eng-hrv',
    'hsb': 'deu-hsb',
    'orv': 'eng-orv',
    'pol': 'eng-pol',
    'rus': 'eng-rus',
    'slv': 'eng-slv',
    'srp_Cyrl': 'eng-srp_Cyrl',
    'srp_Latn': 'eng-srp_Latn',
    'ukr': 'eng-ukr',
    # Celtic
    'bre': 'bre-eng',
    'cor': 'cor-eng',
    'cym': 'cym-eng',
    'gla': 'eng-gla',
    'gle': 'eng-gle',
    # Baltic
    'lav': 'eng-lav',
    'lit': 'eng-lit',
    'prg': 'eng-prg',
    # Hellenic
    'ell': 'ell-eng',
    'grc': 'eng-grc',
    # Uralic
    'est': 'eng-est',
    'fin': 'eng-fin',
    'fkv': 'fin-fkv',
    'hun': 'eng-hun',
    # Other European
    'eus': 'eng-eus',
    'fao': 'eng-fao',
    'got': 'eng-got',
    'gsw': 'eng-gsw',
    'hye': 'eng-hye',
    'mkd': 'eng-mkd',
    'mlt': 'eng-mlt',
    'sqi': 'eng-sqi',
    'tur': 'eng-tur',
    'ota': 'eng-ota',
    'ota_Arab': 'eng-ota_Arab',
    'ota_Latn': 'eng-ota_Latn',
    'zza': 'eng-zza',
    # Constructed languages
    'epo': 'eng-epo',
    'ido': 'eng-ido',
    'ido_Latn': 'eng-ido_Latn',
    'ile': 'eng-ile',
    'ile_Latn': 'epo-ile_Latn',
    'ina': 'eng-ina',
    'ina_Latn': 'ina_Latn-lfn_Latn',
    'jbo': 'eng-jbo',
    'jbo_Latn': 'eng-jbo_Latn',
    'lfn': 'eng-lfn',
    'lfn_Cyrl': 'lfn_Cyrl-por',
    'lfn_Latn': 'eng-lfn_Latn',
    'nov': 'eng-nov',
    'tlh': 'eng-tlh',
    'tlh_Latn': 'fra-tlh_Latn',
    'toki': 'eng-toki',
    'toki_Latn': 'fra-toki_Latn',
    'tzl': 'eng-tzl',
    'tzl_Latn': 'eng-tzl_Latn',
    'vol': 'eng-vol',
}


class TatoebaChallenge(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, lang=None, split_type=None):
        super().__init__()
        self._task_name = "tatoeba_challenge"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size

        if lang == "srp_Latn" or lang == "srp_Cyrl":
            self._lang = lang
        else:
            self._lang = iso639.Lang(lang).pt3

        if self._lang not in langs_tatoebamt:
            raise Exception("Language not available")
        lang_pairs = TATOEBA_LANGMAP.get(self._lang, f"{self._lang}-eng")
        self._dataset = load_dataset(
            "Helsinki-NLP/tatoeba_mt",
            lang_pairs,
            split=split_type,
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN"),
        )


    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["sourceString"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class DocLevelMT(DownstreamTask):
    def __init__(
        self, lang, file_dir, min_ngram_size=8, max_ngram_size=13, split_type=None
    ):
        super().__init__()
        if lang not in langs_doclevel_mt:
            raise Exception("Language not available")
        self._task_name = f"doclevelmt_{lang}"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._file_dir = file_dir

        self._file_paths = [
            Path(file_dir, "doclevel-MT-benchmark", split_type, f"wmt.tok.{lang}"),
            Path(file_dir, "doclevel-MT-benchmark", split_type, f"ost.tok.{lang}"),
        ]
        if self._file_paths is None:
            raise Exception("Must provide path to data in JSONL format")

    def generate_ngrams(self):
        for file_path in self._file_paths:
            with open(file_path, "r") as f:
                for idx, line in enumerate(f):
                    try:
                        text = general_detokenize(line)
                        self._update_ngrams(
                            text, self._min_ngram_size, self._max_ngram_size
                        )
                    except Exception as e:
                        logging.exception(f"Error processing line {idx}")
            return self.ngrams


class Sib200(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, lang=None, split_type=None):
        super().__init__()

        if lang == "sq":
            lang_pt3 = "als"
        elif lang == "lv":
            lang_pt3 = "lvs"
        else:
            lang_pt3 = iso639.Lang(lang).pt3
        self._lang = lang_pt3
        self._script = langs_script[self._lang]
        self._full_lang = f"{self._lang}_{self._script}"
        self._task_name = "sib200_" + self._full_lang
        if self._full_lang not in langs_sib200:
            raise Exception("Language not available")
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "Davlan/sib200", self._full_lang, split=split_type, trust_remote_code=True
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["text"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class MMLUProX(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, lang=None, split_type=None):
        super().__init__()
        if lang not in langs_mmlu_pro_x:
            raise Exception("Language not available")
        self._task_name = "mmlu_pro_x"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "li-lab/MMLU-ProX", lang, split=split_type, trust_remote_code=True
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                options = range(0,10)
                for option in options:
                    if f"option_{option}" in line:
                        text = line[f"option_{option}"]
                        if text != None:
                            self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class BLEnD(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, lang=None):
        super().__init__()
        self._task_name = "blend"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        LANG_TO_BLEND = {
            'el': 'GR',
            'id': 'ID', 
            'es': 'ES',  # !! or MX
            'en': 'US',  # !! or GB/NG
            'fa': 'IR',
            'zh': 'CN',
            'ko': 'KR',  # !! or KP
            'az': 'AZ',
            'am': 'ET',
            'ar': 'DZ',
        }
        self._lang = LANG_TO_BLEND.get(lang, lang.upper())
        if self._lang not in langs_blend:
            raise Exception("Language not available")
        self._dataset = load_dataset("nayeon212/BLEND", "short-answer-questions", split=self._lang, trust_remote_code=True)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["Question"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class MultiJail(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, lang=None):
        super().__init__()
        self._task_name = "multijail"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        if lang not in langs_multijail:
            raise Exception("Language not available")
        self._dataset = load_dataset(
            "DAMO-NLP-SG/MultiJail", split="train", trust_remote_code=True
        )
        self._lang = lang

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line[self._lang]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class Menlo(DownstreamTask):
    LANG_TO_MENLO = {
        'bg': 'bg_BG',
        'cs': 'cs_CZ',
        'da': 'da_DK',
        'de': 'de_DE',
        'el': 'el_GR',
        'en': 'en_GB',
        'es': 'es_ES',
        'fr': 'fr_FR',
        'hr': 'hr_HR',
        'hu': 'hu_HU',
        'it': 'it_IT',
        'nl': 'nl_NL',
        'pl': 'pl_PL',
        'pt': 'pt_PT',
        'ro': 'ro_RO',
        'sk': 'sk_SK',
        'sv': 'sv_SE',
        'tr': 'tr_TR',
        'uk': 'uk_UA',
    }

    def __init__(self, min_ngram_size=8, max_ngram_size=13, split_type=None, lang=None):
        super().__init__()
        self._task_name = "menlo"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "facebook/menlo", split=split_type, trust_remote_code=True
        )
        self._lang = self.LANG_TO_MENLO.get(lang, lang)

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                if line["lang_code"] == self._lang:
                    text = line["raw_prompt"]
                    self._update_ngrams(
                        text, self._min_ngram_size, self._max_ngram_size
                    )
                    text = line["response_a"]
                    self._update_ngrams(
                        text, self._min_ngram_size, self._max_ngram_size
                    )
                    text = line["response_b"]
                    self._update_ngrams(
                        text, self._min_ngram_size, self._max_ngram_size
                    )
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class WikiAnn(DownstreamTask):
    # TODO: Recheck this
    def __init__(self, min_ngram_size=8, max_ngram_size=13, lang=None, split_type=None):
        super().__init__()
        if lang != "no":
            if lang not in langs_wikiann:
                raise Exception("Language not available")
        
        self._task_name = "wikiann"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "unimelb-nlp/wikiann", lang, split=split_type, trust_remote_code=True
        )
        self._lang = lang

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                tokens = line["tokens"]
                text = " ".join(tokens)
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class mArenaHard_v2(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, lang=None):
        super().__init__()
        self._task_name = "m-arenahard_v2"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        if lang not in langs_marenahard_v2:
            raise Exception("Language not available")
        self._dataset = load_dataset(
            "CohereLabs/m-ArenaHard-v2.0",
            lang,
            split="test",
            token=os.getenv("HF_TOKEN"),
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["prompt"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception as e:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class mArenaHard_v21(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, lang=None):
        super().__init__()
        self._task_name = "m-arenahard_v21"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        if lang not in langs_marenahard_v21:
            raise Exception("Language not available")
        self._dataset = load_dataset(
            "CohereLabs/m-ArenaHard-v2.1",
            lang,
            split="test",
            token=os.getenv("HF_TOKEN"),
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["prompt"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception as e:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class mDollyMT(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, lang=None):
        super().__init__()
        if lang not in langs_m_dolly_mt:
            raise Exception("Language not available")
        self._task_name = "m-dolly-mt"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            "CohereLabs/dolly-machine-translated-v2", lang, split="test"
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["prompt"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception as e:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class AyaEvaluationSuite(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, lang=None):
        super().__init__()
        self._task_name = "aya_evaluation_suite"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        configs = get_dataset_config_names("CohereLabs/aya_evaluation_suite")
        self._dataset = concatenate_datasets(
            [
                load_dataset("CohereLabs/aya_evaluation_suite", c, split="test")
                for c in configs
            ]
        )

        if lang == "lv":
            self._lang = "lvs"
        else:
            self._lang = iso639.Lang(lang).pt3
        
        if self._lang not in langs_aya_evaluation_suite:
            raise Exception("Language not available")

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                if line["language"] == self._lang:
                    text = line["inputs"]
                    self._update_ngrams(
                        text, self._min_ngram_size, self._max_ngram_size
                    )
                    text = line["targets"]
                    self._update_ngrams(
                        text, self._min_ngram_size, self._max_ngram_size
                    )
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams


class AyaRedTeaming(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13, lang=None):
        super().__init__()
        self._task_name = "aya_red_teaming"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._lang = iso639.Lang(lang).name.lower()
        self._dataset = load_dataset(
            "CohereLabs/aya_redteaming",
            split=self._lang,
            token=os.getenv("HF_TOKEN"),
        )

    def generate_ngrams(self):
        for idx, line in enumerate(self._dataset):
            try:
                text = line["prompt"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
            except Exception:
                logging.exception(f"Error processing line {idx}")
        return self.ngrams
