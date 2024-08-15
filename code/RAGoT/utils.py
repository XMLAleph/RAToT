import os
from typing import List, Dict

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import json
from pathlib import Path

import _jsonnet
from rapidfuzz import fuzz

stemmer = PorterStemmer()

stop_words_set = set(stopwords.words("english"))

QUESTION_MARKER = " Q: "
COMPQ_MARKER = " QC: "
SIMPQ_MARKER = " QS: "
INTERQ_MARKER = " QI: "
ANSWER_MARKER = " A: "
EOQ_MARKER = "[EOQ]"
LIST_JOINER = " + "
BLANK = "__"
WH_WORDS = set(["who", "what", "where", "how", "why", "when", "which"])


def get_sequence_representation(
    origq: str,
    question_seq: List[str],
    answer_seq: List[str],
    compq_marker: str = COMPQ_MARKER,
    interq_marker: str = INTERQ_MARKER,
    answer_marker: str = ANSWER_MARKER,
    simpq_marker: str = SIMPQ_MARKER,
):
    ret_seq = compq_marker + origq
    if len(question_seq) != len(answer_seq):
        raise ValueError(
            "Number of generated questions and answers should match before"
            "question generation. Qs: {} As: {}".format(question_seq, answer_seq)
        )

    for aidx in range(len(answer_seq)):
        ret_seq += interq_marker
        ret_seq += question_seq[aidx]
        ret_seq += answer_marker + answer_seq[aidx]
    ret_seq += simpq_marker
    return ret_seq


def tokenize_str(input_str):
    return word_tokenize(input_str)


def stem_tokens(token_arr):
    return [stemmer.stem(token) for token in token_arr]


def filter_stop_tokens(token_arr):
    return [token for token in token_arr if token not in stop_words_set]


def stem_filter_tokenization(input_str):
    return stem_tokens(filter_stop_tokens(tokenize_str(input_str.lower())))


# functions borrowed from AllenNLP to parse JSONNET with env vars
def get_environment_variables() -> Dict[str, str]:
    """
    Wraps `os.environ` to filter out non-encodable values.
    """
    return {key: value for key, value in os.environ.items() if _is_encodable(value)}


def _is_encodable(value: str) -> bool:
    """
    We need to filter out environment variables that can't
    be unicode-encoded to avoid a "surrogates not allowed"
    error in jsonnet.
    """
    # Idiomatically you'd like to not check the != b""
    # but mypy doesn't like that.
    return (value == "") or (value.encode("utf-8", "ignore") != b"")

def infer_dataset_from_file_path(file_path: str) -> str:
    matching_datasets = []
    file_path = str(file_path)
    for dataset in ["hotpotqa", "2wikimultihopqa", "musique", "iirc"]:
        if dataset.lower() in file_path.lower():
            matching_datasets.append(dataset)
    if not matching_datasets:
        raise Exception(f"Dataset couldn't be inferred from {file_path}. No matches found.")
    if len(matching_datasets) > 1:
        raise Exception(f"Dataset couldn't be inferred from {file_path}. Multiple matches found.")
    return matching_datasets[0]


def infer_source_target_prefix(config_filepath: str, evaluation_path: str) -> str:
    source_dataset = infer_dataset_from_file_path(config_filepath)
    target_dataset = infer_dataset_from_file_path(evaluation_path)
    source_target_prefix = "_to_".join([source_dataset, target_dataset]) + "__"
    return source_target_prefix

def get_config_file_path(experiment_name_or_path: str) -> str:
    if not experiment_name_or_path.endswith(".jsonnet"):
        # It's a name
        assert (
            len(experiment_name_or_path.split(os.path.sep)) == 1
        ), "Experiment name shouldn't contain any path separators."
        matching_result = list(Path(".").rglob("**/*" + experiment_name_or_path + ".jsonnet"))
        matching_result = [
            _result
            for _result in matching_result
            if os.path.splitext(os.path.basename(_result))[0] == experiment_name_or_path
        ]
        if len(matching_result) != 1:
            exit(f"Couldn't find one matching path with the given name ({experiment_name_or_path}).")
        config_filepath = matching_result[0]
    else:
        # It's a path
        config_filepath = experiment_name_or_path
    return config_filepath


def read_json(file_path: str) -> Dict:
    with open(file_path, "r") as file:
        instance = json.load(file)
    return instance


def read_jsonl(file_path: str) -> List[Dict]:
    with open(file_path, "r") as file:
        instances = [json.loads(line.strip()) for line in file.readlines() if line.strip()]
    return instances


def write_json(instance: Dict, file_path: str):
    with open(file_path, "w") as file:
        json.dump(instance, file)


def write_jsonl(instances: List[Dict], file_path: str):
    with open(file_path, "w") as file:
        for instance in instances:
            file.write(json.dumps(instance) + "\n")

'''
def find_matching_paragraph_text(corpus_name: str, original_paragraph_text: str) -> str:
    # 这里改一下
    retriever_address_config = get_retriever_address()
    retriever_host = str(retriever_address_config["host"])
    retriever_port = str(retriever_address_config["port"])

    params = {
        "query_text": original_paragraph_text,
        "retrieval_method": "retrieve_from_elasticsearch",
        "max_hits_count": 1,
        "corpus_name": corpus_name,
    }

    url = retriever_host.rstrip("/") + ":" + str(retriever_port) + "/retrieve"
    result = requests.post(url, json=params)

    if not result.ok:
        print("WARNING: Something went wrong in the retrieval. Skiping this mapping.")
        return None

    result = result.json()
    retrieval = result["retrieval"]

    for item in retrieval:
        assert item["corpus_name"] == corpus_name

    retrieved_title = retrieval[0]["title"]
    retrieved_paragraph_text = retrieval[0]["paragraph_text"]

    match_ratio = fuzz.partial_ratio(original_paragraph_text, retrieved_paragraph_text)
    if match_ratio > 95:
        return {"title": retrieved_title, "paragraph_text": retrieved_paragraph_text}
    else:
        print(f"WARNING: Couldn't map the original paragraph text to retrieved one ({match_ratio}).")
        return None
'''