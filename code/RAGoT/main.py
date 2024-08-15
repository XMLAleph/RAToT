"""
# RAGoT的最外层主代码，通过命令行读取模型配置等，然后载入框架，读取数据集，通过ragot框架处理其中的每条数据，最后将答案等信息输出为文件

### 特别注意colab与本地的版本同步问题！！！
# 运行调试统一在colab上做，特别注意和本地的版本同步
# RAGoT_0320.zip：本地代码3月20日“第一次”的压缩包，3月20日修改后的版本是RAGoT_0321.zip
# experiments_0320.ipynb：colab上代码3月20日“最后一次”修改后的版本
# 特别注意每天备份colab上的.ipynb文件！！！

### 具体安排
# 已完成：
    数据集处理
    框架代码实现    
    框架代码正确性调整
    具体模型和检索器代码实现（确认下有没有需要改的地方）
    测试gpt或flan-t5接口调用能不能一次性返回多个不同的输出来实现重复采样，如果可以的话，voter进一步实现多次投票（而不是通过for循环重复采样）
    测试gpt返回logprobs的情况（gpt相关的测试另外准备一个代码）
    调整代码，使voter进行多次投票（n和temperature等设置在config里确定）
    准备两种generator的prompt，一种是ircot的原始设置（带distractor），一种是不带distractor，只包含原始文本
    (prompt需要运行生成相关的代码，顺便看下有没有问题)
    准备voter的两种prompt（带[INST]等符号和不带这种符号）
    (带sign这种符号的感觉目前不太确定怎么写，先不考虑相关的实验)
    构造config文件
    先手动构建一个完整的config，先确保能正常运行，等前提实验做完再考虑具体的实验设置
    生成prompt（需要看下包含和不包含distractor的prompt的效果差别
    实际运行测试，确保能正常运行
    写评价体系，注意需要设计gold文档确实被检索到的比例（不过这个可能还需要修改推理完成后保存的结果）
    完成和ircot类似的qa环节，做前提实验，看相似文档的实际影响，顺便测试评价体系的正确性
        注意修改reader的设置，返回全部para和goldpara，分别测试生成的情况
        如果只用gold文档进行检索增强，效果是不是会比用gold和非gold文档更好；或者用gold和无关文档。按之前the power of noise的报告，应该有gold+无关>gold>gold+相似。但是最好进行实验验证
        # 这一点是整个方法有意义的基础
        # 因为前提实验只需要llm_qa一个模块，answer_extractor无法按get_last_reason_answer的方式提取答案，所以把get_last_reason_answer换回get_last_answer（不过按现在的做法应该没打乱将每个模块的生成加入到cot中的顺序，应该没有影响）
        # 从2wikimqa上的结果看，相似文本确实会明显影响RAG的效果（实验设置是数据集原始的10条上下文文本，包括2-4个相关文本，其他的都是相似文本，在ircot的实际执行过程中，相似文本应该会比这种情况更多，所以想办法筛选相关文本（一定程度上也相当于筛选掉相似文本）的做法应该是有意义的，
            但是不知道相似文本数量变化对RAG效果影响的具体趋势（筛选掉一部分相似文档对效果的影响有多大）
    再完善一下整体的结构，保证确实能正常运行
        修改state.data的结构，把各个备选的thought、投票结果等voter的细节信息都像add_answer一样作为cot的一部分加入进去，并打印出来
        有办法输出检索到的所有文档（题目）
        加上ircot最后一步的qa模块
            这里想一下怎么加，最好是两个阶段的答案都能保留，这样一次实验就能获取两种做法的效果
            答案在代码执行的过程中都会输出，但两个都能用来做测试感觉比较难
            answer_extractor应该是最后一个模块，final_answer的participant应该都是answer_extractor
            主要是还是不太清楚answer_extractor会被调用几次
            如果确实只调用了生成最终答案这两次的话，应该可以提取所有answer_extractor的结果
            初步测试要不就不用这些做法了，就大致看下两个阶段的答案的区别
        特别注意输出或保存各种重要信息（不过应该可以通过qid找到问题，从而复现ragot处理问题的过程）
            ircot的代码就会输出qid，可以直接从数据集文件中找到相应的代码
            reasoning chain本身也会保存相应的文件
        再完善一下整体的结构，保证确实能正常运行
            写一下相应的config文件（主要是加了最后一步的qa模块） 
            调整超参数设置。修改tempreature（原始设置都是0，这里希望能提高投票的“多样性”，改成0.7
            有个问题，怎么根据文档题目找到文档内容？
                检索器可能有办法实现这个操作（现在通过retrieve_t2p.py可以实现）
            注意文件保存和运行过程等具体情况
            
# 目前进行:
    在subsubsample_dev上试一下完整的框架（包括超参数）
        现在应该能从输出的可视化思维链看到ragot过程的答案和最后qa环节得到的答案（最后用来评测的是qa环节的答案），看下ragot的答案和ragot-qa的答案有没有变化
        先看下目前的情况。现在主要在调整的是vote的保底机制和各种prompt
        想办法看一下每个生成的置信分数，看正确答案的分数有没有可能更高
            这个后面再看，先跑结果
        这种逐步生成思维链的做法实际上也有缺点，有可能给定的上下文文档已经足够回答问题了，但还要继续检索文档，进行生成
            不过去掉之前已经出现过的文档的做法能一定程度上缓解这种问题
    在整个2wikimqa数据集上先完整运行一次ragot，根据实际输出情况考虑后面具体的改进方向
    整理目前的想法和问题，准备后面讨论（周日基本上做好PPT）
        问题：需要验证的设置比较多（例如之前检索到的上下文文档是否保留），但都做实验的话时间和成本太大，这一点讨论的时候提一下
        问题：感觉prompt是比较需要修改的点，打算就通过目前2wikimqa数据集的两条数据观察修改的效果，不知道这种做法是不是合理
        问题：PPT里放现在具体的prompt
        
# 后面需要：
    调整vote机制
    看下评价体系的具体实现过程，实现gold文档筛选的相关评价指标计算（后面还需要完善相关的实验设置）
    把prompt生成、模型调用等环节单独写一个代码，后面方便调试不同的prompt
    看下flan-t5具体怎么并行输入（各个输入间互不干扰），一次性生成多个备选thought
        这样flan-t5模型的效率应该就和ircot差不多了，后面用pro+的话应该也可以用flan-t5-xl
    实现config文件的动态定义（填写基本的配置，直接生成config文件，或者根据文件名直接生成（因为文件名就是实验名）
    
### 其他问题
# 开放域多步推理问题：
    1.需要检索器能做到全面地检索到各个相关文档
        检索相关文档：根据语义相关性检索文档。
        回答问题的过程有隐含的中间步骤。
        这种有向无环图的推理过程算是一种理想化的问题建模，但实际中llm不一定能按这种理想化的离散推理步骤进行生成
    2.需要llm能做到以合理的顺序和正确的逻辑，在这些相关文档的帮助下完成推理
但在实际中：
    1.回答这类问题往往需要模型进行局部的推理
        例如，要生成“天阴时应该收衣服”，很难直接一次性生成“天阴时应该{}”->“天阴时应该{收衣服}”。比如根据问题“天阴时应该{}”进行检索，
          可能会检索到“天阴会下雨”。但还需要进一步检索到“下雨应该收衣服”，但“下雨应该收衣服”与问题“天阴时应该/会{}”的语义相关度
            并不高（，但是有潜在的相关性，与“天阴时会{下雨}”的相关性就比较高）。
        IRCoT的思路就是利用问题与相关文档的潜在相关性，通过不断生成显式的推理过程（来续写问题/扩大问题内容的范围），将这种潜在的相关性
          逐渐转变成显式的语义相关性（因为很难通过一般的检索技术（基于语义相关性）直接检索到这些潜在的相关文档），从而逐渐检索到这些相关
          文档。
        检索是基于语义相关度寻找相关的文档。
        另外从这个例子来看推理实际上是比较广泛的（根据已有信息生成训练中没出现过的新信息），常识性推理应该可以认为是一种比较容易衡量的任务
        也就是例如回答A->C，隐含的推理过程是A->B->C，A—>B和A->C都有相关文档，但A->C没有直接的相关文档。中间步骤（latent step，隐步骤）B->C的相关文档与问题A->{}的语义相关性可能并不高，但与A->B->{}的语义相关性就比较高，更容易被检索到）  
"""

import argparse
import json
import logging
import time
import os
import re
from typing import Dict, Any

from tqdm import tqdm
import _jsonnet

from module_instances import MODULE_NAME_CLASS, READER_NAME_CLASS
from data_instances import StructuredDataInstance

from Dataset_readers import DatasetReader
from Framework import RAGoT, Modules_controller

from Generators.llmgenerator import get_model_and_tokenizer

from metrics.drop_answer_em_f1 import DropAnswerEmAndF1
from metrics.support_em_f1 import SupportEmF1Metric
from metrics.answer_support_recall import AnswerSupportRecallMetric

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

    arg_parser.add_argument("--debug", action="store_true", default=False, help="Debug output")
    # 调试期间的模型输出注意全部用这里的debug控制
    return arg_parser.parse_args()

def answer_extractor(potentially_cot: str) -> str:
    # In a few experiments I forgot the configuring the answer extractor part
    # and so the final answer is a cot chain instead. Instead of having to do
    # all those exps again, I'm just doing answer_extraction here. This needs
    # to be fixed later though.

    if potentially_cot.startswith('"') and potentially_cot.endswith('"'):
        potentially_cot = potentially_cot[1:-1]

    cot_regex = re.compile(".* answer is:? (.*)\\.?")
    match = cot_regex.match(potentially_cot)
    if match:
        output = match.group(1)
        if output.endswith("."):
            output = output[:-1]
    else:
        output = potentially_cot

    return output

def load_config(args):
    config_path = args.config
    # 不清楚这里variable replacement具体是什么功能，后面出问题再调整
    ext_vars = get_environment_variables() # 获取环境变量
    config = json.loads(_jsonnet.evaluate_file(config_path, ext_vars=ext_vars))
    '''
    if args.debug:
        print(20*"#" +" Config " + 20*"#")
        print(config.keys())
        for key in config.keys():
            print(config[key])
    '''
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
    reader = READER_NAME_CLASS[reader_name](**reader_config)

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

def load_predictions(prediction_file_path: str) -> Dict:
    with open(prediction_file_path, "r") as file:
        id_to_predictions = json.load(file)
    return id_to_predictions

def build_dataset_reader(args, config):
    if "reader" in config:
        reader_config = config["reader"]
        reader_name = reader_config.pop("name")
        reader: DatasetReader = READER_NAME_CLASS[reader_name](**reader_config)
    else:
        # 应该没有这种情况
        reader: DatasetReader = READER_NAME_CLASS[args.example_reader]()
    return reader

def build_framework(config):
    # framework是由多个module组成的
    modules = {}
    for key, value in config["modules"].items():
        class_name = value.pop("name")
        if class_name not in MODULE_NAME_CLASS:
            raise ValueError(
                "No class mapped to model name: {} in MODEL_NAME_CLASS:{}".format(class_name, MODULE_NAME_CLASS)
            )
        module = MODULE_NAME_CLASS[class_name](**value)
        if key in config:
            raise ValueError(
                "Overriding key: {} with value: {} using instantiated model of type:"
                " {}".format(key, config[key], class_name)
            )
        config[key] = module.run # 各个模块的统一的运行接口(通过这个接口直接运行模块的功能)，这样通过传入config就能传入所有这些接口，调用比较方便
        # 不过这个好像没必要保持一致，看下其他部分是怎么用这一句的
        modules[key] = module

    '''
    # 这里的效果不清楚
    # Special case for ExecutionParticipant
    for module in modules.values():
        if isinstance(module, ExecutionParticipant):
            module.set_model_lib(module)
    '''
    
    modules_controller = Modules_controller(config, data_class=StructuredDataInstance)
    framework = RAGoT(modules_controller)
    # 这里framework就相当于整个框架实际运行的控制器，如果后面需要试其他框架，直接在这里改即可(不同的框架就完全分开定义)
    return framework

def evaluate_by_dicts(
    prediction_type: str,
    id_to_ground_truths: Dict[str, Any],
    id_to_predictions: Dict[str, Any],
) -> Dict:
    if prediction_type == "answer":
        metrics = [DropAnswerEmAndF1(), SupportEmF1Metric(do_normalize_answer=True)]
    elif prediction_type in ("titles", "pids", "real_pids"):
        metrics = [SupportEmF1Metric()]
    elif prediction_type in ("paras"):
        metrics = [AnswerSupportRecallMetric()]

    for id_ in set(id_to_ground_truths.keys()):
        ground_truth = id_to_ground_truths[id_]
        prediction = id_to_predictions[id_]

        assert isinstance(prediction, (str, list))
        if prediction_type == "answer" and isinstance(prediction, str):
            if prediction.strip().startswith("[") or prediction.strip().endswith("]"):
                prediction = [e for e in prediction.replace('"', "").replace("[", "").replace("]", "").split(",")]
            else:
                prediction = [prediction]

        assert isinstance(prediction, (list, tuple))
        prediction = [str(e) for e in prediction]

        if prediction_type == "answer":
            prediction = [answer_extractor(_prediction) for _prediction in prediction]  # Temporary.
            metrics[0](prediction, [ground_truth])
            metrics[1](prediction, ground_truth)
        elif prediction_type in ("titles", "pids", "real_pids"):
            metrics[0](prediction, ground_truth)
        elif prediction_type in ("paras"):
            predicted_paras = [
                " ".join([eval(prediction_)["title"], eval(prediction_)["paragraph_text"]])
                for prediction_ in prediction
            ]
            metrics[0](predicted_paras, ground_truth)

    evaluation_results = metrics[0].get_metric()
    if prediction_type == "answer":
        evaluation_results_ = metrics[1].get_metric()
        evaluation_results["sp_em"] = evaluation_results_["title_em"]
        evaluation_results["sp_f1"] = evaluation_results_["title_f1"]
        evaluation_results["sp_precision"] = evaluation_results_["title_precision"]
        evaluation_results["sp_recall"] = evaluation_results_["title_recall"]

    return evaluation_results

def perform_reasoning(args, reader, framework):
    qa_chains = [] #results
    
    start_time = time.time()
    print("loading data...")
    iterator = reader.read_examples(args.input)
    print("load data successfully")
    for example in tqdm(iterator):
        qa_chains.append(
            framework.reasoning(
                example, 
                debug=args.debug
            )
        )
    end_time = time.time()
    seconds_taken = round(end_time - start_time)
    print("运行完成，time used: {}".format(seconds_taken))
    
    predictions = {x[0]: x[1] for x in qa_chains}
    
    experiment_name = os.path.splitext(os.path.basename(args.config))[0] #就是config的文件名
    prediction_directory = os.path.join("predictions", experiment_name) #和output的文件名第一部分相同
    prediction_file_name = os.path.splitext(os.path.basename(args.input))[0]
    #prediction_file_name = infer_source_target_prefix(args.config, args.input) + prediction_file_name
    prediction_file_path = os.path.join(prediction_directory, "prediction__" + prediction_file_name + ".json")

    (output_file, output_name) = os.path.split(prediction_file_path)
    
    if not os.path.exists(output_file):
        os.mkdir(output_file)
    print(f"Writing predictions in {prediction_file_path}")
    with open(prediction_file_path, "w") as output_fp:
        json.dump(predictions, output_fp, indent=4)

    ext_index = prediction_file_path.rfind(".")
    
    time_taken_file_path = prediction_file_path[:ext_index] + "_time_taken.txt"
    with open(time_taken_file_path, "w") as file:
        file.write(str(seconds_taken))

    chains = [x[2] for x in qa_chains]
    chain_tsv = prediction_file_path[:ext_index] + "_chains.txt"
    with open(chain_tsv, "w") as output_fp:
        for chain in chains:
            output_fp.write(chain + "\n")

    # Also save original full evaluation path.
    full_eval_path = prediction_file_path[:ext_index] + "_full_eval_path.txt"
    with open(full_eval_path, "w") as output_fp:
        output_fp.write(args.input)
        
    return

def user_input_process(query):
    question = query
    return {
        "qid": "#",
        "query": query,
        "answer": "",
        "question": question,
        "titles":[],
        "paras":[],
        "pids":[],
        "real_pids":[],
        "backup_paras":[],
        "backup_titles":[],
        "valid_titles":[],
        "metadata":{},
    }

def perform_qa(query):
    # web问答系统的接口，对于输入的问题，返回一个答案，不经过调用数据集的阶段，但需要考虑怎么把query组织成和数据集一样的输入（看下reader）
    config = load_config("configs/ircot_qa_codex_2wikimultihopqa____prompt_set_1___bm25_retrieval_count__6___distractor_count__2.jsonnet")
    framework = build_framework(config) #实例化RAGoT框架

    get_model_and_tokenizer()
    
    example = user_input_process(query)
    start_time = time.time()
    result = framework.reasoning(
                example,
            )
    end_time = time.time()
    seconds_taken = round(end_time - start_time)
    print("运行完成，time used: {}".format(seconds_taken))
    answer = result[1]
    print(answer)
    chain = result[3]
    print(chain)
    docs = result[4] # 这些好像不方便排序？因为是分多次检索到的多个，各步之间没有先后关系 # 这里特别注意，千万不能打乱原有的顺序（否则保存结果会出问题）
    # 就先直接这样返回，也不方便计算和问题的相似度
    # 文档和标题可以直接从final_state中获取，不用按规则从chain中提取，但是需要注意返回的格式（不能打乱原有的顺序）
    # docs是[(title, para)]的列表
    return answer, chain, docs
    
def perform_evaluate(args):
    # 主要是需要改config文件的名称
    # 或者根据这里从config文件名称读取prediction的做法，看下output路径是怎么命名的，就不用在命令行输入了
    experiment_name = os.path.splitext(os.path.basename(args.config))[0] #就是config的文件名
    prediction_directory = os.path.join("predictions", experiment_name) #和output的文件名第一部分相同
    prediction_file_name = os.path.splitext(os.path.basename(args.input))[0]
    #prediction_file_name = infer_source_target_prefix(config_filepath, args.input) + prediction_file_name
    prediction_file_path = os.path.join(prediction_directory, "prediction__" + prediction_file_name + ".json")

    save_metrics_path = os.path.join(
        prediction_directory, "evaluation_metrics__" + prediction_file_name + ".json"
    )

    # get prediction_type
    experiment_config = load_config(args)
    prediction_type = experiment_config["prediction_type"]

    # prep ground_truths
    id_to_ground_truths, id_to_gold_docs = load_ground_truths(
        experiment_config,
        args.input,
    )
    
    # prep predictions
    id_to_predictions = load_predictions(prediction_file_path)

    # verify equality
    if set(id_to_ground_truths.keys()) != set(id_to_predictions.keys()):
        exit("Ids in input examples and predictions don't match.")

    # evaluate
    evaluation_results = evaluate_by_dicts(
        prediction_type=prediction_type,
        id_to_predictions=id_to_predictions,
        id_to_ground_truths=id_to_ground_truths,
    )
    print(json.dumps(evaluation_results, indent=4))

    # Save the evaluation metrics
    print(f"Saving metrics in {save_metrics_path}")
    with open(save_metrics_path, "w") as file:
        json.dump(evaluation_results, file, indent=4)
    print(evaluation_results)

    # Save the ground_truth used in the same json/dict format (just for convenience)
    ground_truth_in_dict_file_path = os.path.join(
        prediction_directory, "ground_truth__" + prediction_file_name + ".json"
    )
    with open(ground_truth_in_dict_file_path, "w") as file:
        json.dump(id_to_ground_truths, file, indent=4)
    return
    
if __name__ == "__main__":
    parsed_args = parse_arguments()
    '''
    if parsed_args.debug:
        logging.basicConfig(level=logging.DEBUG) # logging模块不太清楚怎么用，先确保整个流程写完，有需要再看logging怎么写
    else:
        logging.basicConfig(level=logging.ERROR)
    '''
    logging.basicConfig(level=logging.ERROR)
    config = load_config(parsed_args)
    
    # 从运行整个流程的层面上，包含的模块是1.example_reader，读取数据集；2.reason_performer，对数据集中的每个问题，进行多步推理并回答；3.评价器，
    example_reader = build_dataset_reader(args=parsed_args, config=config)
    ragot_reason_performer = build_framework(config) #实例化RAGoT框架

    get_model_and_tokenizer()

    perform_reasoning(
        args=parsed_args,
        reader=example_reader,
        framework=ragot_reason_performer,
        # override_answer_by=override_answer_by,
    )
    # override_answer_by不知道是什么作用
    
    perform_evaluate(args=parsed_args)