import argparse
import json
import logging
import os
import re
from typing import Dict, Any

from tqdm import tqdm
import _jsonnet

from modules.data_instances import StructuredDataInstance

from modules.Dataset_Readers import MultiParaRCReader

from utils import (
    get_environment_variables, 
    infer_source_target_prefix,
)

def parse_arguments():
    # python main.py --input  --output  --config  --debug
    arg_parser = argparse.ArgumentParser(description="Convert HotPotQA dataset into SQUAD format")
    
    arg_parser.add_argument("--input", type=str, required=False, help="input dataset file")
    # arg_parser.add_argument("--output", type=str, required=False, help="output result file") 
    arg_parser.add_argument("--config", type=str, required=True, help="config file")
    # arg_parser.add_argument("--output", type=str, required=True, help="output file")

    return arg_parser.parse_args()

def load_config(args):
    config_path = args.config
    # 不清楚这里variable replacement具体是什么功能，后面出问题再调整
    ext_vars = get_environment_variables() # 获取环境变量
    config = json.loads(_jsonnet.evaluate_file(config_path, ext_vars=ext_vars))
    return config

def load_ground_truths(
    experiment_config: Dict,
    ground_truth_file_path: str,
    question_type_key: str = None,
    question_type_value: str = None,
) -> Dict:
    # 测试集的路径实际上就是ground_truth的路径，数据集文件里已经记录了答案和相关文档，只是测试的时候不载入
    # Reader载入的数据的metadata中就包含了完整的gold文档，可以直接用来和框架检索的情况对比（gold文档有多少被选中了）
    # Load the config
    reader_config = experiment_config["reader"]
    reader_name = reader_config.pop("name")
    reader = MultiParaRCReader(**reader_config)

    # Prep prediction_type and reader
    prediction_type = experiment_config["prediction_type"]
    if prediction_type in ("titles", "pids", "real_pids") and reader_name != "multi_para_rc":
        exit("The titles and pids prediction evaluation is only supported for multi_para_rc reader.")

    if prediction_type in ("titles", "pids", "real_pids", "paras"):
        
        reader.add_paras = False
        reader.add_gold_paras = True
        reader.add_pinned_paras = False
        reader.remove_pinned_para_titles = True
        reader.add_paras_from_files = None
    
    # prep ground_truths
    id_to_ground_truths = {}
    id_to_gold_docs = {}
    for example in reader.read_examples(ground_truth_file_path):

        if question_type_key is not None or question_type_value is not None:
            if question_type_key is None or question_type_value is None:
                raise Exception("Both question type key and value must be passed if any one of them is passed.")
            if question_type_key not in example["metadata"]:
                raise Exception(f"Key {question_type_key} not present in the example instance.")

            if example["metadata"][question_type_key] != question_type_value:
                continue

        id_ = example["qid"]
        if prediction_type in ("answer", "paras"):
            id_to_ground_truths[id_] = example["answer"]
        elif prediction_type == "titles":
            id_to_ground_truths[id_] = example["titles"]
        elif prediction_type == "pids":
            id_to_ground_truths[id_] = example["pids"]
        elif prediction_type == "real_pids":
            id_to_ground_truths[id_] = example["real_pids"]
        else:
            raise Exception("Unknown prediction_type.")
        id_to_gold_docs[id_] = example["metadata"]["gold_titles"] #用这个信息应该就能对比了
        # 这个评价后面再具体实现，优先考虑整体的效果，做这个对比需要实验的设定是保存之前的相关文档且只只保留被选出的thought对应的文档（假设就是相关文档）
    return id_to_ground_truths, id_to_gold_docs

# 从.txt文件中读取每条数据推理过程的检索文档，返回为id_to_list
# 这里是专门用来算ragot的
def load_ragot_retrieved_docs(prediction_file_path):
    temp_item = []
    id_to_retrieved = {}
    with open(prediction_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if line == "\n":
                if len(temp_item)>0:
                    id_ = temp_item[1]
                    retrieved = eval(temp_item[-9][3:])
                    id_to_retrieved[id_] = retrieved
                temp_item = [] # 清空
            temp_item.append(line.strip('\n'))
    return id_to_retrieved

def load_ircot_retrieved_docs(prediction_file_path):
    # 最后一项应该是没输入
    temp_item = []
    id_to_retrieved = {}
    with open(prediction_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if line == "\n":
                if len(temp_item)>0:
                    id_ = temp_item[1]
                    retrieved = eval(temp_item[-8][3:])
                    id_to_retrieved[id_] = retrieved
                temp_item = [] # 清空
            temp_item.append(line.strip('\n'))
    if len(temp_item)>0:
        id_ = temp_item[1]
        retrieved = eval(temp_item[-6][3:])
        id_to_retrieved[id_] = retrieved
    return id_to_retrieved

def perform_retrieve_evaluate(args):
    # 计算检索结果的recall/precision值，
    # 命令行输入还是和之前一样
    # 这部分具体运行的时候看下文件名的情况
    experiment_name = os.path.splitext(os.path.basename(args.config))[0] #就是config的文件名
    prediction_directory = os.path.join("predictions", experiment_name) #和output的文件名第一部分相同
    prediction_file_name = os.path.splitext(os.path.basename(args.input))[0]
    #prediction_file_name = infer_source_target_prefix(config_filepath, args.input) + prediction_file_name
    datasets = ["hotpotqa","2wikimultihopqa","musique"]
    prediction_file_path_prefix = ""
    if "ircot" in args.config:
        for dataset in datasets:
            if dataset in args.input:
                prediction_file_path_prefix = f"{dataset}_to_{dataset}__"
                break
    prediction_file_path = os.path.join(prediction_directory, "prediction__" + prediction_file_path_prefix + prediction_file_name + "_chains.txt")
    experiment_config = load_config(args)
    print(prediction_file_path)
    
    _, id_to_gold_docs = load_ground_truths(experiment_config, args.input)
    if "ragot" in args.config:
        id_to_retrieved_docs = load_ragot_retrieved_docs(prediction_file_path)
    else:
        id_to_retrieved_docs = load_ircot_retrieved_docs(prediction_file_path)
    
    if set(id_to_gold_docs.keys()) != set(id_to_retrieved_docs.keys()):
        exit("Ids in input examples and predictions don't match.")
    
    correct_count = 0
    sum_precision = []
    sum_recall = []
    for id_ in list(id_to_gold_docs.keys()):
        count = 0
        gold_docs = id_to_gold_docs[id_]
        retrieved_docs = id_to_retrieved_docs[id_]
        for gold_doc in gold_docs:
            if gold_doc in retrieved_docs:
                count += 1
        precision = count / len(retrieved_docs)
        recall = count / len(gold_docs)
        sum_precision.append(precision)
        sum_recall.append(recall)
        correct_count += 1
    
    precision_ = sum(sum_precision) / len(sum_precision)
    recall_ = sum(sum_recall) / len(sum_recall)
    print(f"precision:{precision_}, recall:{recall_}")
    f1_score = (2*precision_*recall_) / (precision_+recall_)
    print(f"f1-score:{f1_score}")
    
if __name__ == "__main__":
    parsed_args = parse_arguments()
    perform_retrieve_evaluate(parsed_args)