from llm_qa_systems.run import init_state_heap, run_state_step
from llm_qa_systems.data_instances import QuestionAnsweringStep, AnswerVotingStep, TempAnsweringStep, QuestionGenerationStep
from flask_utils import get_qid_for_query

qa_sys_configs = {}
qa_sys_configs["search_query"] = "When was Ludwig Gruno Of Hesse-Homburg's father born?"
qa_sys_configs["search_qid"] = get_qid_for_query()
qa_sys_configs["search_source"] = "local_2wikimultihopqa"
qa_sys_configs["search_generator"] = "gpt-3.5-turbo-instruct"
qa_sys_configs["search_mode"] = "ircot"
init_state_heap(qa_sys_configs)
run_time = 0
while True:
    # 现在有个不正常的现象是无法达到停止状态（command=[EOQ]也能输出，但最后还是无法停止）
    # 这里无法结束运行的原因是，在原来的代码里，这个方法是一直在运行的，所以self.num_calls可以累积；但在这种情况下，self.num_calls无法累积，因此每次调用都停留在和第一次一样的状态，也就是self.num_calls=0，所以每次调用的结果都一样，就是无法终止
        # 想一下怎么解决
        # 其他的类都不依赖属性值self.num_calls的全局性，但这个方法正好依赖，目前只能这样解释
        # 再看下进一步的处理
    print(f"第{run_time+1}轮")
    run_time += 1
    if run_time > 20: #发生异常，无法正常退出
        break
    state_data, done = run_state_step(qa_sys_configs)
    # 这里输出done表示的是“一次完整的问答系统调用”结束了，如果是sc-ircot，相当于完成了一次采样
    last_step = state_data.get_last_step()
    if last_step.participant == "step_by_step_bm25_retriever" and isinstance(last_step, TempAnsweringStep):
        print("TS: " + last_step.answers) # 返回给前端的好像可以就是这个？？？看下具体内容
    elif last_step.participant == "step_by_step_bm25_retriever" and isinstance(last_step, QuestionAnsweringStep):
        print("S: " + last_step.answer)
    elif last_step.participant == "step_by_step_cot_reasoning_gen" and isinstance(last_step, TempAnsweringStep):
        print("TA: " + last_step.answers) #看下这里到底怎么写
    elif last_step.participant == "step_by_step_cot_reasoning_gen" and isinstance(last_step, QuestionAnsweringStep):
        if qa_sys_configs["search_mode"] == "ragot":
            try:
                last_second_step = state_data.get_last_second_step()
                last_third_step = state_data.get_last_third_step()
                # 这里顺序应该是确定的，都是vote环节一次性新增的
                print("V: " + last_third_step.vote)
                print("S: " + last_second_step.answer)
                print("A: " + last_step.answer)
            except:
                print("#"*3 + "出现异常" + "#"*10)
                print("A: " + last_step.answer)
        else:
            print("A: " + last_step.answer)
        # 另外考虑下ragot vote怎么判断和处理
    elif last_step.participant == "step_by_step_exit_controller" and isinstance(last_step, QuestionAnsweringStep):
        print("J: " + last_step.answer)
    elif last_step.participant == "generate_main_question" and isinstance(last_step, QuestionGenerationStep):
        print("FQ: " + last_step.question)
    elif last_step.participant == "answer_main_question" and isinstance(last_step, QuestionAnsweringStep):
        print("FT: " + last_step.answer)
    elif last_step.participant == "extract_answer" and isinstance(last_step, QuestionAnsweringStep):
        print("FA: " + last_step.answer)
    else:
        print("?????" + last_step.participant)
    #if isinstance(step, TempAnsweringStep):
    if done:
        print("运行完成")
        chain = "\n" + qa_sys_configs["search_query"]
        chain += "\n" + state_data.get_printable_reasoning_chain()
        print(chain)
        break