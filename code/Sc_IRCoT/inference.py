# 1.实现投票法的sc-ircot(这一步还需要注意模型设置(gpt可以直接调temperature，其他llm需要继续看下参数设置)，保证“能”产生多样化的输出)
# 2.用sc-ircot跑通整个流程，周五开始做完整的实验
# 3.用其他llm实现基于score的sc-ircot，后面也做实验
# 4.这些完成之后先沟通一次，后面再看情况实现进一步的sc-ircot

## 后面注意也用gpt跑一次完整的实验(用验证集，四个数据集分别跑一次)(其他llm用完整的测试集)
## 超参数和ircot用验证集得到的结果保持一致
## 另外最后注意下评价指标
## ircot和sc-ircot都重新跑结果

## ircot, sc-ircot(vote), sc-ircot(score)
## 1.首先正式确定实验配置(ircot和sc-ircot的配置保持一致，sc-ircot的配置尽量与sc-cot的论文保持一致)
## (先确定基本的实验设置，sc-ircot还需要调整模型参数和sc数量，尽量提高多样性)
## 2.然后下载数据集，开始做ircot的实验，特别注意保存实验结果等文件，做好记录，评价指标也同时跑完(但原始的结果文件也要保存)
## 3.ircot运行期间做好sc-ircot的具体实现(先实现vote投票的，score的再确认下score的做法，最晚等运行sc-ircot(vote)的期间修改代码)
## (另外ircot代码中的score并没有用上，所有score的设置都是0)
## 4.开始做sc-ircot的实验
## 5.整理所有的实验结果，沟通

## 因为模型和数据集比较大，需要尽量提高运行效率，减少下载次数
## 实验需要多个模型和数据集(一共16*2(3)=32(48)个结果)，做法是对每种框架的每个模型，一次性做完所有数据集上的结果，分别做好记录，然后换下一个模型
## 论文中的配置没直接给出，按照实际情况，retrieval_count=6，distractor=2应该比较合适(不过测试集上也有distractor吗???)
## 就采用固定的实验配置
## 到底怎么比较合适???
## distractor在测试集上应该也是有效的，测试集运行也需要设置prompt
## 因为token数上限是一个确定的值，不同实验配置对应的样例个数也会不一样，一定程度上会影响效果
## 要不还是做完整的实验，按现在的设置，修改下配置文件即可，但需要的时间会成倍增加
## 不过应该是要按ircot_qa的改，因为还涉及到evaluation的类型
## 先用最小的模型和最小的数据集进行测试，结果正常再考虑其他模型和数据集
## 就按统一的设置(retrieval=4, distractor=2)来
## 修改retriever部分，不然iirc数据集可能无法运行

## sc_count的设置应该至少在10以上
## flan-t5-xl用colab的gpu显存不够，周二沟通看能不能用gpu
## 先在colab上把flan-t5-large的最后一个实验做完，同时准备好flan-t5-base的实验配置，后面跑flan-t5-base的实验调整好sc-ircot的参数设置，保证“能”输出不同的内容（有多样性）
## 后面跑sc-ircot的实验期间沟通gpu
## flan-t5-xl看下有没有可能通过调整参数(token=6000->token=3000)来减少需要的内存
# 另外后面重新做nlp的awesome-list，每周更新
# 后面也看下评价指标具体是怎么算的

# 开始跑sc-ircot(flan-t5-xxx)，先用简单的例子看下原始的模型输入到底是什么，再看情况改成do_sample的模式
import argparse
import json
import logging
import time
import os

from tqdm import tqdm
import _jsonnet

from modules.Dataset_Readers import DatasetReader
from Constant import MODEL_NAME_CLASS, READER_NAME_CLASS
from modules.data_instances import StructuredDataInstance
# from modules.participant_execution import ExecutionParticipant
from modules.Framework import ModelController, Decomposer
from modules.gen_models.llmgenerator import get_model_and_tokenizer

from utils import get_environment_variables

def parse_arguments():
    arg_parser = argparse.ArgumentParser(description="Convert HotPotQA dataset into SQUAD format")
    arg_parser.add_argument("--sc", action="store_true", default=False, help="sc-ircot or ircot mode")
    arg_parser.add_argument("--threads", default=1, type=int, help="Number of threads (use MP if set to >1)")
    arg_parser.add_argument("--input", type=str, required=False, help="input dataset file")
    arg_parser.add_argument("--output", type=str, required=False, help="output result file") 
    arg_parser.add_argument("--config", type=str, required=True, help="config file")

    arg_parser.add_argument("--debug", action="store_true", default=False, help="Debug output")
    arg_parser.add_argument("--silent", action="store_true", default=False, help="Silent / no prints")
    #arg_parser.add_argument("--threads", default=1, type=int, help="Number of threads (use MP if set to >1)")
    return arg_parser.parse_args()

def load_config(config_file):    
    ext_vars = get_environment_variables()
    configs = json.loads(_jsonnet.evaluate_file(config_file, ext_vars=ext_vars))
    return configs

def load_reader(configs):
    reader_config = configs["reader"]
    reader_name = reader_config.pop("name")
    reader: DatasetReader = READER_NAME_CLASS[reader_name](**reader_config)
    return reader

def load_framework_and_models(configs):
    # 这部分1.把模型接口处理好，包括retriever和inference_llm，2.把模型框架处理好(controller和decomposer的部分)
    # print("loading participant models (might take a while)...")
    models = {} #models[<model_name>] = model_class
    for key, value in configs["models"].items():
        model_name = value.pop("name")
        if model_name not in MODEL_NAME_CLASS:
            raise ValueError(
                "No class mapped to model name: {} in MODEL_NAME_CLASS:{}".format(model_name, MODEL_NAME_CLASS)
            )
        model = MODEL_NAME_CLASS[model_name](**value) #MODEL_NAME_CLASS是从model name到model
        if key in configs:
            raise ValueError(
                "Overriding key: {} with value: {} using instantiated model of type:"
                " {}".format(key, configs[key], model_name)
            )
        configs[key] = model.query #这里后面可以改一下，用models记录model.query，定义更清楚
        models[key] = model #这里的models实际上相当于取了MODEL_NAME_CLASS的一部分
    
    '''
    # Special case for ExecutionParticipant
    for model in models.values():
        if isinstance(model, ExecutionParticipant):
            model.set_model_lib(model_map)
    '''
    
    ## instantiating
    # 这里应该就需要开始修改
    controller = ModelController(configs, data_class=StructuredDataInstance)
    decomposer = Decomposer(controller)
    return decomposer, models
    
def conduct_inference(args, dataset_loader, decomposer, model_map):
    qa_chains = [] #results
    
    start_time = time.time()
    print("loading data...")
    iterator = dataset_loader.read_examples(args.input)
    print("load data successfully")
    
    if args.threads > 1:
        import multiprocessing as mp

        mp.set_start_method("spawn")
        with mp.Pool(args.threads) as p:
            if args.sc:
                qa_chains = p.map(decomposer.return_qid_prediction_sc, iterator)
            else:
                qa_chains = p.map(decomposer.return_qid_prediction, iterator)
    else:
        if args.sc:
            for example in tqdm(iterator):
                qa_chains.append(
                    decomposer.return_qid_prediction_sc(
                        example, debug=args.debug, silent=args.silent
                    )
                )
        else:
            for example in tqdm(iterator):
                qa_chains.append(
                    decomposer.return_qid_prediction(
                        example, debug=args.debug, silent=args.silent
                    )
                )
    
    end_time = time.time()
    seconds_taken = round(end_time - start_time)
    print("运行完成，time used: {}".format(seconds_taken))
    predictions = {x[0]: x[1] for x in qa_chains}
    
    (output_file, output_name) = os.path.split(args.output)
    if not os.path.exists(output_file):
        os.mkdir(output_file)
    print(f"Writing predictions in {args.output}")
    with open(args.output, "w") as output_fp:
        json.dump(predictions, output_fp, indent=4)

    ext_index = args.output.rfind(".")
    
    time_taken_file_path = args.output[:ext_index] + "_time_taken.txt"
    with open(time_taken_file_path, "w") as file:
        file.write(str(seconds_taken))

    chains = [x[2] for x in qa_chains]
    chain_tsv = args.output[:ext_index] + "_chains.txt"
    with open(chain_tsv, "w") as output_fp:
        for chain in chains:
            output_fp.write(chain + "\n")

    # Also save original full evaluation path.
    full_eval_path = args.output[:ext_index] + "_full_eval_path.txt"
    with open(full_eval_path, "w") as output_fp:
        output_fp.write(args.input)
    print("保存结果完成")
    
def user_input_process(query):
    question = query
    return {
        "qid": "#",
        "query": query,
        "answer": "",
        "question": question,
        "titles":[],
        "paras":[],
        "urls":[],
        "pids":[],
        "real_pids":[],
        "backup_paras":[],
        "backup_titles":[],
        "valid_titles":[],
        "metadata":{},
    }

def perform_qa(query, qa_sys_configs):
    # web问答系统的接口，对于输入的问题，返回一个答案，不经过调用数据集的阶段，但需要考虑怎么把query组织成和数据集一样的输入（看下reader）
    searchsource = qa_sys_configs["searchsource"].split('_') # ['local','hotpotqa'] web_bing
    llmgenerator = qa_sys_configs["llmgenerator"]
    mode = qa_sys_configs["mode"] # ircot scircot
    if mode == "ragot":
        configs = {}
        if searchsource[0] != 'local':
            configs = load_config("configs/ragot_qa_gpt_3_2wikimultihopqa____prompt_set_1___bm25_retrieval_count__6___distractor_count__2.jsonnet")
            configs["models"]["step_by_step_bm25_retriever"]["retriever_type"] = searchsource[0]
            print("retriever_type is {}".format(searchsource[0]))
            configs["models"]["step_by_step_bm25_retriever"]["retrieval_method"] = f"retrieve_from_{searchsource[1]}"
        else:
            configs = load_config(f"configs/ragot_qa_gpt_3_{searchsource[1]}____prompt_set_1___bm25_retrieval_count__6___distractor_count__2.jsonnet")
        #configs["models"][""]["engine"] = llmgenerator
    else:
        configs = {}
        if searchsource[0] != 'local':
            configs = load_config("configs/ircot_qa_codex_2wikimultihopqa____prompt_set_1___bm25_retrieval_count__6___distractor_count__2.jsonnet")
            configs["models"]["step_by_step_bm25_retriever"]["retriever_type"] = searchsource[0]
            configs["models"]["step_by_step_bm25_retriever"]["retrieval_method"] = f"retrieve_from_{searchsource[1]}"
        else:
            configs = load_config(f"configs/ircot_qa_codex_{searchsource[1]}____prompt_set_1___bm25_retrieval_count__6___distractor_count__2.jsonnet")
        #configs["models"]["step_by_step_cot_reasoning_gen"]["engine"] = llmgenerator
        # ircot的原代码如果gpt-model不是codex就会直接报错，改一下相应的部分
        
    decomposer, _ = load_framework_and_models(configs)
    get_model_and_tokenizer()
    
    example = user_input_process(query)
    start_time = time.time()
    if mode == "ircot":
        result = decomposer.return_qid_prediction(
                example,
            )
    elif mode == "scircot":
        result = decomposer.return_qid_prediction_sc(
                example,
            )
    else:
        result = decomposer.return_qid_prediction(
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
    # docs是(title, para, url)的列表
    return answer, chain, docs

if __name__ == "__main__":
    # python inference.py --config --input --output --debug (--silent)
    # python inference.py --config configs/ircot_flan_t5_large_musique____prompt_set_1___bm25_retrieval_count__2___distractor_count__1.jsonnet --input processed_data/musique/dev_subsubsampled.jsonl --output predictions/ircot_codex_musique____prompt_set_1___bm25_retrieval_count__2___distractor_count__1/prediction__musique_to_musique__dev_subsampled.json --debug --silent
    # 可以先用之前下载的config文件测试流程
    # 先保证ircot的整个流程能按新代码走通，再改成SC-ircot的流程
    # 后面调试的时候做一个更小的数据集
    # 另外建立索引的部分也要参考ircot的做法
    parsed_args = parse_arguments()
    debug = parsed_args.debug
    configs = load_config(parsed_args.config)
    if debug:
        print("config文件载入完成")
    dataset_reader = load_reader(configs) #dataset_reader的具体细节先不改，优先保证能跑通
    if debug:
        print("dataset loader载入完成")
    decomposer, model_map = load_framework_and_models(configs) #这里的逻辑可以改一下
    if debug:
        print("模型框架载入完成")    
    print("\nLoading model and tokenizer.")
    get_model_and_tokenizer()
    print("Loaded model and tokenizer.\n")
    print("Start Running...")
    conduct_inference(
        args=parsed_args,
        dataset_loader=dataset_reader,
        decomposer=decomposer,
        model_map=model_map
    )
    print("Finish running...")