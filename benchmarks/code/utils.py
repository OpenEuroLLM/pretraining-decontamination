import re

def general_detokenize(string) -> str:
    # Some benchmarks are tokenized. Extracted from: 
    # https://github.com/EleutherAI/lm-evaluation-harness/blob/c9772b90a8ee95b0df1ba76651443a0534c96aad/lm_eval/utils.py#L235
    string = string.replace(" n't", "n't")
    string = string.replace(" )", ")")
    string = string.replace("( ", "(")
    string = string.replace('" ', '"')
    string = string.replace(' "', '"')
    string = re.sub(r" (['.,])", r"\1", string)
    return string