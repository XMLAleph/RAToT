import copy
import heapq
import json
import logging
from collections import defaultdict
import _jsonnet
import pickle

from llm_qa_systems.module_instances import MODEL_NAME_CLASS
from llm_qa_systems.data_instances import BasicDataInstance, StructuredDataInstance 
from llm_qa_systems.utils import get_environment_variables, get_qid_for_query
# 这些import目录注意根据实际情况修改（包括代码里的文件目录）


class ModelController(object):
    """This class is a `ModelController` that takes multiple (arbitrary)
    models and a control specification of how to interface the different
    models (which can be thought of as a kind of state graph). For example

    """

    def __init__(self, model_list, data_class=BasicDataInstance):
        """Create an instance of a ComplexModel

        :param model_list: a list of models with identifiers and
          control flow.
        :type model_list: dict
        """
        if "start_state" not in model_list:
            raise ValueError("Must specify start state")
        if "end_state" not in model_list:
            raise ValueError("Must specify end state")
        self.model_list = model_list
        self.data_class = data_class

    def execute(self, state, debug=False):
        """Executes a command and query

        :param state: a given state in search
        :type state: SearchState (defined here)
        :returns: a list of output
        :rtype: list
        """
        if state.next not in self.model_list:
            self.logger.error("Can not handle next state: " + state.next)
            return []
        try:
            model_func = self.model_list[state.next]

            model_output = model_func(state, debug=debug)

            if not isinstance(model_output, list):
                return [model_output]
            return model_output
        except Exception as e:
            self.logger.error(e, exc_info=True)
            raise ValueError("Error caught during model execution:  %s" % e)

    def init_data(self, data_instance):
        """Create an initialized version of the data object
        that will get through around.

        :param data_instance: any arbitrary piece of data.
        :rtype: self.data_class
        """
        return self.data_class(data_instance)

    @property
    def start_state(self):
        return self.model_list["start_state"]

    @property
    def end_state(self):
        return self.model_list["end_state"]

    @property
    def logger(self):
        """Returns a logger instance"""
        level = ".".join([__name__, type(self).__name__])
        return logging.getLogger(level)
    
class SearchState(object):
    """Tracks and records the state of a given search."""

    def __init__(self, json_data, command, score=0.0):
        """Keep track of different stages in the state

        :param json_data: some basic, json represntation of data
        """
        self._data = json_data
        self._score = score
        self._next = command

    def copy(self):
        """Does a deep copy of the state

        :returns: new search state
        """
        new_data = copy.deepcopy(self._data)
        new_score = copy.deepcopy(self._score)
        new_next = copy.deepcopy(self._next)

        return SearchState(new_data, new_next, new_score)

    def __lt__(self, other):
        if self.score < other.score:
            return True
        return False

    def __eq__(self, other):
        if self.score == other.score:
            return True
        return False

    @property
    def data(self):
        return self._data

    @property
    def score(self):
        return self._score

    @property
    def next(self):
        return self._next

    @next.setter
    def next(self, value):
        self._next = value

    @data.setter
    def data(self, value):
        self._data = value
 
def load_config(config_file):    
    ext_vars = get_environment_variables()
    configs = json.loads(_jsonnet.evaluate_file(config_file, ext_vars=ext_vars))
    return configs

def load_framework_and_models(qa_sys_configs):
    searchsource = qa_sys_configs["search_source"].split('_') # ['local','hotpotqa'] web_bing
    #llmgenerator = qa_sys_configs["llmgenerator"]
    mode = qa_sys_configs["search_mode"] # ircot scircot ragot
    configs = {}
    if mode == "ragot":
        if searchsource[0] != 'local':
            configs = load_config("llm_qa_systems/configs/ragot_qa_gpt_3_2wikimultihopqa____prompt_set_1___bm25_retrieval_count__6___distractor_count__2.jsonnet")
            configs["models"]["step_by_step_bm25_retriever"]["retriever_type"] = searchsource[0]
            configs["models"]["step_by_step_bm25_retriever"]["retrieval_method"] = f"retrieve_from_{searchsource[1]}"
        else:
            configs = load_config(f"llm_qa_systems/configs/ragot_qa_gpt_3_{searchsource[1]}____prompt_set_1___bm25_retrieval_count__6___distractor_count__2.jsonnet")
    
    elif mode in ["ircot", "scircot"]:
        if searchsource[0] != 'local':
            configs = load_config("llm_qa_systems/configs/ircot_qa_codex_2wikimultihopqa____prompt_set_1___bm25_retrieval_count__6___distractor_count__2.jsonnet")
            configs["models"]["step_by_step_bm25_retriever"]["retriever_type"] = searchsource[0]
            configs["models"]["step_by_step_bm25_retriever"]["retrieval_method"] = f"retrieve_from_{searchsource[1]}"
        else:
            configs = load_config(f"llm_qa_systems/configs/ircot_qa_codex_{searchsource[1]}____prompt_set_1___bm25_retrieval_count__6___distractor_count__2.jsonnet")
    
    else:
        raise ValueError("Not implemented mode")
        
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
        configs[key] = model.run
        models[key] = model
    
    controller = ModelController(configs, data_class=StructuredDataInstance)
    return controller

def save_state_heap(qid, heap):
    # 这里完成对heap的保存。目前测试data和next分开保存，load时再组合成state的做法
    # 应该能直接保存heap，先用保存heap的方式，如果运行有问题再改成保存state的方式
    '''
    if len(heap) > 1:
        raise ValueError("can't handle heap with more than 1 element")
    '''
    save_path = str(qid) + ".pkl"
    #state = heapq.heappop(heap) # 现在这种做法必须确保heap里只有一个元素
    #state_data = state.data
    #state_infs = {"next":state.next, "score":state.score}
    with open(save_path,"wb") as f:
        #pickle.dump(state, f)
        pickle.dump(heap, f)
    # pickle.dump(state_data, f)
    #dumps/loads
    return 

def load_state_heap(qid):
    save_path = str(qid) + ".pkl"
    #heap = []
    with open(save_path, "rb") as f:
        #state = pickle.load(f)
        heap = pickle.load(f)
    '''
    state = SearchState(
        state_data,  ## initial input
        state_command,  ## starting point
        state_score,  ## starting score
    )
    '''
    #heapq.heappush(heap, state)
    return heap

def self_consistency_vote(sample_results):
    if len(sample_results) == 1:
        return sample_results[0]
    answer2index = {}
    answer2time = defaultdict(int)
    for i in range(len(sample_results)):
        candidate_answer = sample_results[i]["answer"]
        answer2index[candidate_answer] = i # 这里不选state，直接选对应的下标，返回sample_results[i]
        answer2time[candidate_answer] += 1
    answer_time = sorted(answer2time.items(), key = lambda e:e[1],reverse=True)
    final_answer = answer_time[0][0]
    final_result_index = answer2index[final_answer]
    return sample_results[final_result_index]
    
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

def init_state_heap(qa_sys_configs, debug=True):
    # 这一段也希望能持久保存，但controller不好保存（结构过于复杂，pickle不一定能保存
    # 这部分目前暂定每次调用时实时载入，不通过pickle保存（主要是考虑到qa_sys_config本来就需要保存，pickle也需要进行一次文件操作
    # 这种做法速度比较慢，但操作更简单
    # 先改一下后面的堆操作
    # controller也可能能通过pickle保存，但现在不优先考虑
    controller = load_framework_and_models(qa_sys_configs)
    #get_model_and_tokenizer()
    
    query = qa_sys_configs["search_query"]
    qid = qa_sys_configs["search_qid"]
    json_input = user_input_process(query)
    
    start_command = controller.start_state
    start_data = controller.init_data(json_input)

    ## min-heap
    heap = []
    init_input = json_input["question"] if json_input["question"] else "UNKNOWN"
    if debug:
        print("[START QUERY] : %s" % init_input)

    init_state = SearchState(
        start_data,  ## initial input
        start_command,  ## starting point
        score=0.0,  ## starting score
    )
    ## push it to heap
    heapq.heappush(heap, init_state)
    
    # 开始heap操作
    ## pop from heap
    current_state = heapq.heappop(heap)
    
    if current_state.next == controller.end_state:
        if debug:
            print("[FAILED]: %s" % init_input)
            
    if debug:
        print("[MIN_STATE] command=%s" % (current_state.next))
        
    for new_state in controller.execute(current_state, debug=debug): # 这里因为是第一轮运行，不属于if达到终止的情况
        heapq.heappush(heap, new_state)
    
    save_state_heap(qid, heap)

    return
    # 这里没有返回的内容（问答系统还没有生成内容）

def run_state_step(qa_sys_configs, debug=True):
    
    controller = load_framework_and_models(qa_sys_configs)
    
    qid = qa_sys_configs["search_qid"]
    
    heap = load_state_heap(qid)
    
    if len(heap) == 0:
        if debug:
            print("[FAILED]")
        return None, 1

    ## pop from heap
    current_state = heapq.heappop(heap)

    done = 0
    
    if debug:
        print("[MIN_STATE] command=%s" % (current_state.next))
      
    if current_state.next == controller.end_state:
        # 暂定不再考虑tasks的情况
        # 如果tasks没清空导致结果异常，再考虑直接在modules里取消tasks
        # 目前的代码里各个模块（包括participant_qa）里“只有”question_copyer会在指定进行多次qa时会通过add_task生成下一次qa的task
        # 这个机制本来也算是另一种sc-ircot，但sc-ircot就按和论文方法一样的做法实现
        if debug:
            print("[TERMINATED]")
        done = 1
    else:
        for new_state in controller.execute(current_state, debug=debug):
            heapq.heappush(heap, new_state)
    '''
    if not (current_state.next == controller.end_state):
        for new_state in controller.execute(current_state, debug=debug):
            heapq.heappush(heap, new_state)
    else:
        if current_state.data.has_tasks():
            new_task = current_state.data.pop_task()
            # print("popped task!")
            # print(new_task)
            new_state = current_state.copy()
            if new_task.task_question:
                new_state.data.add_qgen(new_task.task_question)
            new_state.next = new_task.task_participant
            heapq.heappush(heap, new_state)
        else:
            if debug:
                print("[TERMINATED]") # 这里想办法改一下输出，改成通过解析输出能判断是否停止的做法（改current_state）
            done = 1
    '''
    save_state_heap(qid, heap)
    
    return current_state.data, done

if __name__ == "__main__":
    qa_sys_configs = {}
    qa_sys_configs["search_query"] = "When was Ludwig Gruno Of Hesse-Homburg's father born?"
    qa_sys_configs["search_qid"] = get_qid_for_query()
    qa_sys_configs["search_source"] = "local_2wikimultihopqa"
    qa_sys_configs["search_generator"] = "gpt-3.5-turbo-instruct"
    qa_sys_configs["search_mode"] = "ircot"
    init_state_heap(qa_sys_configs)
    run_time = 0
    while True:
        print(f"第{run_time+1}轮")
        run_time += 1
        if run_time > 20: #发生异常，无法正常退出
            break
        state, done = run_state_step(qa_sys_configs)
        if done:
            print("运行完成")
            data = state.data
            chain = "\n" + qa_sys_configs["search_query"]
            chain += "\n" + data.get_printable_reasoning_chain()
            print(chain)