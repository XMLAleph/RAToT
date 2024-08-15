# 定义ragot的总框架，控制整个流程的运行

import heapq
import json
import logging

from data_instances import BasicDataInstance, ReasonState
    

class Modules_controller(object):
    def __init__(self, modules, data_class=BasicDataInstance):
        if "start_state" not in modules:
            raise ValueError("Must specify start state")
        if "end_state" not in modules:
            raise ValueError("Must specify end state")
        
        self.module_list = modules
        self.data_class = data_class
    
    def init_data(self, data_instance):
        # state和data_instance是什么关系？
        # state应该是框架“中”运行的过程，data_instance是框架全局的数据类型（维护整个过程产生的cot等）
        # 那state里的data是什么？command是怎么变化的？
        """Create an initialized version of the data object
        that will get through around.

        :param data_instance: any arbitrary piece of data.
        :rtype: self.data_class
        """
        return self.data_class(data_instance)
    
    def execute_module(self, state, debug=False):
        """Executes a command and query

        :param state: a given state in search
        :type state: SearchState (defined here)
        :returns: a list of output
        :rtype: list
        """
        # state是“当前”的状态，执行这里的execute函数，从而进入到下一个状态
        if state.next not in self.module_list:
            self.logger.error("Can not handle next state: " + state.next)
            return []
        try:
            module_func = self.module_list[state.next]

            module_output = module_func(state, debug=debug)

            if not isinstance(module_output, list):
                return [module_output]
            return module_output
        except Exception as e:
            self.logger.error(e, exc_info=True)
            raise ValueError("Error caught during model execution:  %s" % e)

    @property
    def start_state(self):
        return self.module_list["start_state"]

    @property
    def end_state(self):
        return self.module_list["end_state"]

    @property
    def logger(self):
        """Returns a logger instance"""
        level = ".".join([__name__, type(self).__name__])
        return logging.getLogger(level)
    
    
class RAGoT(object):
    def __init__(self, modules_controller):
        """Create a `QuestionDecomposer instance`

        :param model_ensemble: a collection of models with control instructions
        """
        # 这里init接受的参数modules实际上是config(不过config的主要内容也确实就是模块的定义)。
        # 在main函数中载入modules时，就把modules的run函数（主运行函数）加入到config中了，所以这里通过modules就能直接调用
        self.controller = modules_controller
        
    def reasoning(
        self,
        example,
        override_answer_by=None,
        debug=False,
    ):
        # 这个函数就是框架运行的总控制函数，输入一个问题(example)，返回问题id，问题的答案和推理链
        final_state, other_states = self.perform_reasoning(example, debug=debug)
        # 返回值是final_state和"other_state"的原因：当开始执行final_state所在的这一层时，堆中（队列中）这一层以前的所有state（如果有多个）肯定都已经执行完了。当这一层的state达到最终状态（next_state=<EOF>）时，就返回结果，此时堆中其他的元素就是和达到最终状态的state同一层的"other state"
        # 后面是对输出的处理，核心的推理过程在ragot_reasoning
        if final_state is None:
            if debug:
                print(example["question"] + " FAILED!")
            chain = "\n" + example["qid"] + "\n" + example["question"]
            if debug:
                print("\n")
            return (example["qid"], "", chain, "", [])
        else:
            # 这里的具体操作与state的内部设计有关，写模块的时候具体看下
            # chain是一整个字符串
            data = final_state._data
            chain = "\n" + example["qid"] + "\n" + example["question"] # 根据qid就能定位到数据集中的特定问题
            chain += "\n" + data.get_printable_reasoning_chain()
            chain += "\nS: " + str(final_state._score) # 这里其实不太需要
            chain += "\nGroundTruth: " + str(example["answer"]) # 这样方便后面对比结果
            if debug:
                print(chain)
            if override_answer_by is not None:
                if override_answer_by not in data:
                    print(f"WARNING: The key {override_answer_by} is not present in the data dict.")
                final_answer = data.get(override_answer_by, "")
                if not isinstance(final_answer, str):
                    final_answer = json.dumps(final_answer)
            else:
                final_answer = data.get_last_answer() # 这里主要是希望能获取两次answer
            try:
                json_answer = json.loads(final_answer)
                # use this only if list (ignore numbers, etc)
                if isinstance(json_answer, list) or isinstance(json_answer, str):
                    final_answer = json_answer
            except ValueError:
                # Not a valid json ignore
                pass
            if debug:
                print("\n")
                
            printable_reason_chain = data.get_answers()[-1] #reader生成的推理链应该就在这个位置
            
            titles = data["titles"]
            paras = data["paras"]
            urls = data["urls"]
            assert len(titles) == len(paras), "the num of titles and paras are not equal"
            docs = [(titles[i], paras[i], urls[i]) for i in range(len(titles))]
            
            return (example["qid"], final_answer, chain, printable_reason_chain, docs)
        
    def perform_reasoning(self, json_input, debug=False):
        # RAGoT框架核心的推理函数，输入一条json格式的数据（多步问题），输出答案
        # 其实因为ragot是新方法，这里感觉可以考虑也实现ircot(差别只在推理器上，不把文档分开处理即可，应该可以通过config文件实现定义)
        """Main question decomposition function

        :param json_input: the input to all of the models.
        """
        
        """Run the question decomposer. The main function here is to use
        the controller to pass around inputs to the different models, then
        keep a track of the search state and terminate when the shortest path
        has been found.

        # 注意这里提到了“shortest path”
        # 但问题是ircot怎么会有这种功能？是怎么实现的？或者说为什么会产生多条路径？
        :param json_input: some input to the model
        """
        
        # 初始状态
        start_data = self.controller.init_data(json_input) # state中的data就是data_instance中定义的复杂数据类型
        # 所以state由当前步的“状态”（包括已执行的推理步骤）和下一步要执行的动作组成。这种设计是比较合理的，目前不考虑修改
        start_command = self.controller.start_state
        
        if debug:
            print(start_data)
            
        ## min-heap
        # 定义小顶堆。这里注意看一下堆结构到底是怎么发生作用的（score中间根本没用，用栈的效果应该也一样？数据结构尽量简单）
        heap = []
        init_input = json_input["question"] if json_input["question"] else "UNKNOWN"
        if debug:
            print("[START QUERY] : %s" % init_input)
            
        init_state = ReasonState(
            start_data,  ## initial input
            start_command,  ## starting point
            score=0.0,  ## starting score
        )

        ## push it to heap
        heapq.heappush(heap, init_state)
        # 堆中会存储多个state？堆顶是最近产生（并push进去的）state？
        ## start the main search
        while True:
            if len(heap) == 0:
                if debug:
                    print("[FAILED]: %s" % init_input)
                return None, []

            ## pop from heap
            current_state = heapq.heappop(heap) #然后堆中就不存在这个state了？对，所以整个过程应该是：pop一个state，然后处理，再将新得到的state依次（如果有多个）push到堆中，再重复上述过程，直到pop出的state的next_command是end_command，这时就终止整个过程并返回结果
            # 如果把框架中每个模块的执行记为一轮的话，当下一轮的第一个可能状态开始执行时，上一轮所有的可能状态一定都已经执行完了（树的性质）
            # 所以这种堆结构的基本作用相当于一个队列（先进先出），并且在上一层的所有状态执行完之前不会进入下一个状态）。但堆比队列高级的一点是引入了score：如果状态有相应的score，score会影响处理的先后顺序（不过这一点感觉没什么用）
            # score其实没什么影响（因为最后总会被处理完，或在某一步得到最终答案）
            # 所以这里用堆结构控制整个推理过程，其实也可以用队列，不过堆没有别的影响，暂时不修改
            
            if debug:
                print("[MID_STATE] command=%s" % (current_state.next)) #MIN_STATE
            # if current_state.next is None:
            # print(current_state.data.get_printable_reasoning_chain())
            #     current_state.next = current_state.data.get_last_generator()
            ## end state
            # 终止推理过程的条件
            # 此时current_state就是最终状态，包含完整的推理链，可以提取出最终答案
            if current_state.next == self.controller.end_state: #这里就是退出推理过程的模块
                if current_state.data.has_tasks(): #has tasks，这里的task可能是什么？？？不太理解，后面写模块的时候顺便注意一下
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
            # 执行controller的execute_module方法，从当前状态生成下一个状态
            # 这里实际上生成的新状态应该只有一个(return [state])，这里应该是相当于兼容了更一般的情况
            # 所以是不是也可以作为RAGoT的一种实现方式？（不需要各个thought之间的处理，而是把每个thought都作为一个new state，最后取最短路径对应的答案
            for new_state in self.controller.execute_module(current_state, debug=debug):
                ## push onto heap
                heapq.heappush(heap, new_state)