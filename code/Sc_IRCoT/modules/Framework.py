import copy
import heapq
import json
import logging
from collections import defaultdict
from modules.data_instances import BasicDataInstance


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


## utility class for controlling and recording search state


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

    ## important to implement to work
    ## with the heap datastructures
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


class QuestionSearchBase(object):
    def __init__(self, model_controller):
        """Create a `QuestionDecomposer instance`

        :param model_ensemble: a collection of models with control instructions
        """
        self.controller = model_controller

    def find_answer_decomp(self, json_input, debug=False):
        """Main question decomposition function

        :param json_input: the input to all of the models.
        """
        raise NotImplementedError

    def return_qid_prediction(
        self,
        example,
        override_answer_by=None,
        debug=False,
        silent=False,
    ):
        final_state, other_states = self.find_answer_decomp(example, debug=debug)
        if final_state is None:
            if not silent:
                print(example["question"] + " FAILED!")
            chain = "\n" + example["qid"] + "\n" + example["question"]
            if not silent:
                print("\n")
            return (example["qid"], "", chain, "", [])
        else:
            data = final_state._data
            chain = "\n" + example["qid"] + "\n" + example["question"]
            chain += "\n" + data.get_printable_reasoning_chain()
            chain += "\nS: " + str(final_state._score)
            if not silent:
                print(chain)
            if override_answer_by is not None:
                if override_answer_by not in data:
                    print(f"WARNING: The key {override_answer_by} is not present in the data dict.")
                final_answer = data.get(override_answer_by, "")
                if not isinstance(final_answer, str):
                    final_answer = json.dumps(final_answer)
            else:
                final_answer = data.get_last_answer()
            try:
                json_answer = json.loads(final_answer)
                # use this only if list (ignore numbers, etc)
                if isinstance(json_answer, list) or isinstance(json_answer, str):
                    final_answer = json_answer
            except ValueError:
                # Not a valid json ignore
                pass
            if not silent:
                print("\n")
                
            printable_reason_chain = data.get_answers()[1] #reader生成的推理链应该就在这个位置
            # 注意，之前索引到的不是reason_chain，因为get_answers是根据get_last_answers改的，answers的列表是倒过来的，所以索引到的一直是检索文档
            
            titles = data["titles"]
            paras = data["paras"]
            urls = data["urls"]
            
            assert len(titles) == len(paras) and len(titles) == len(urls), "the num of titles and paras are not equal"
            docs = [(titles[i], paras[i], urls[i]) for i in range(len(titles))]
            
            return (example["qid"], final_answer, chain, printable_reason_chain, docs)
        

    def return_qid_prediction_sc(
        self,
        example,
        override_answer_by=None,
        debug=False,
        silent=False,
    ):
        sc_count = 8
        answer2state = {}
        answer2time = defaultdict(int)
        print("i.i.d. sample {} times...".format(sc_count))
        for i in range(sc_count):
            candidate_final_state, other_states = self.find_answer_decomp(example, debug=debug)
            
            if candidate_final_state is None:
                if not silent:
                    print(example["question"] + " FAILED!")
                chain = "\n" + example["qid"] + "\n" + example["question"]
                if not silent:
                    print("\n")
                return (example["qid"], "", chain, "", [])
            
            
            candidate_answer = candidate_final_state._data.get_last_answer()
            answer2state[candidate_answer] = candidate_final_state
            answer2time[candidate_answer] += 1
        
        answer_time = sorted(answer2time.items(), key = lambda e:e[1],reverse=True)
        final_answer = answer_time[0][0]
        final_state = answer2state[final_answer]
        
        data = final_state._data
        chain = "\n" + example["qid"] + "\n" + example["question"]
        chain += "\n" + data.get_printable_reasoning_chain()
        chain += "\nS: " + str(final_state._score)
        if not silent:
            print(chain)
        
        if override_answer_by is not None:
            if override_answer_by not in data:
                print(f"WARNING: The key {override_answer_by} is not present in the data dict.")
            final_answer = data.get(override_answer_by, "")
            if not isinstance(final_answer, str):
                final_answer = json.dumps(final_answer)
        else:
            final_answer = data.get_last_answer()
        try:
            json_answer = json.loads(final_answer)
            # use this only if list (ignore numbers, etc)
            if isinstance(json_answer, list) or isinstance(json_answer, str):
                final_answer = json_answer
        except ValueError:
            # Not a valid json ignore
            pass
        if not silent:
            print("\n")
            
        printable_reason_chain = data.get_answers()[1] #reader生成的推理链应该就在这个位置
        
        titles = data["titles"]
        paras = data["paras"]
        urls = data["urls"]
        
        assert len(titles) == len(paras) and len(titles) == len(urls), "the num of titles and paras are not equal"
        docs = [(titles[i], paras[i], urls[i]) for i in range(len(titles))]
        
        return (example["qid"], final_answer, chain, printable_reason_chain, docs)

class Decomposer(QuestionSearchBase):
    def find_answer_decomp(self, json_input, debug=False):
        """Run the question decomposer. The main function here is to use
        the controller to pass around inputs to the different models, then
        keep a track of the search state and terminate when the shortest path
        has been found.

        :param json_input: some input to the model
        """
        # 主要问题是为什么ircot要用堆的形式???(应该说如果是链式结构，没必要用堆，感觉SC-ircot才需要堆结构)
        # 而且堆是按照state的score维护的，这个score具体是怎么规定的?
        ## start state of controller : e.g., generate
        start_command = self.controller.start_state
        start_data = self.controller.init_data(json_input)

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
        # heap本身仍然是个列表，这些堆操作可能可以看作按score对列表中的state进行排序
        # 因为整个过程用到的其实就是heappush和heappop，并没有涉及其他堆操作
        ## push it to heap
        heapq.heappush(heap, init_state)

        ## start the main search
        while True:
            if len(heap) == 0:
                if debug:
                    print("[FAILED]: %s" % init_input)
                return None, []

            ## pop from heap
            current_state = heapq.heappop(heap)
            # 如果是小顶堆，因为init state的score就是0，后面的score会变成负值???

            if debug:
                print("[MIN_STATE] command=%s" % (current_state.next))
            # if current_state.next is None:
            # print(current_state.data.get_printable_reasoning_chain())
            #     current_state.next = current_state.data.get_last_generator()
            ## end state
            if current_state.next == self.controller.end_state:
                if current_state.data.has_tasks():
                    new_task = current_state.data.pop_task()
                    # print("popped task!")
                    # print(new_task)
                    new_state = current_state.copy()
                    if new_task.task_question:
                        new_state.data.add_qgen(new_task.task_question)
                    new_state.next = new_task.task_participant
                    heapq.heappush(heap, new_state)
                    continue
                else:
                    if debug:
                        print("[TERMINATED]")
                    return current_state, heap

            ## generate output and new stated
            # 感觉这里想办法把第一轮cot的返回结果从单个改成多个就可以了(因为无论返回结果是几个，return的都是列表的形式)
            for new_state in self.controller.execute(current_state, debug=debug):

                ## push onto heap
                heapq.heappush(heap, new_state)
