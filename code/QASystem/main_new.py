# 后面运行的时候应该需要把Sc_IRCoT和RAGoT文件夹放在QASystem文件夹里（和main.py在同一个目录下）
from flask import Flask, render_template, request, url_for, redirect
from flask_utils import get_recommend_querys, get_qid_for_query, get_para_for_title, init_search_result_json, load_search_result_json, save_search_result_json, get_cur_docs
from llm_qa_systems.run import init_state_heap, run_state_step, self_consistency_vote
from llm_qa_systems.data_instances import QuestionAnsweringStep, AnswerVotingStep, TempAnsweringStep, QuestionGenerationStep
import threading
from pyngrok import ngrok, conf

#print("Enter your authtoken, which can be copied from https://dashboard.ngrok.com/get-started/your-authtoken")
conf.get_default().auth_token = '2ezwbmwxGCGRDAUC8K5qogVwVYz_5iQGCBn2iRbWneewX65dM'

app = Flask(__name__)

# Open a ngrok tunnel to the HTTP server
public_url = ngrok.connect(5000).public_url
print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}/\"".format(public_url, 5000))

# Update any base URLs to use the public ngrok URL
app.config["BASE_URL"] = public_url

#app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', 
                           temp_id=get_qid_for_query(), 
                           recommend_querys=get_recommend_querys()
                           )

@app.route('/search', methods = ['GET','POST'])
def search():
    print("#"*20)
    print("进入search页面")    
    qa_sys_configs = {}
    qa_sys_configs["search_query"] = request.args.get('inputs', "", type=str)
    if not qa_sys_configs["search_query"]:
        return redirect('https://www.bing.com', 302)
    qa_sys_configs["search_qid"] = request.args.get("id4query")
    qa_sys_configs["search_source"] = request.args.get("menu4searchsource", "", type=str)
    qa_sys_configs["search_generator"] = request.args.get("menu4llmgenerator", "", type=str)
    qa_sys_configs["search_mode"] = request.args.get("menu4mode", "", type=str)
    qa_prefix = "processing..."
    if qa_sys_configs["search_mode"] == "scircot":
        qa_prefix = "i.i.d. 8 times..."
    init_state_heap(qa_sys_configs)
    init_search_result_json(qa_sys_configs)
    print("页面初始化完成，渲染初始页面")
    if qa_sys_configs["search_source"].startswith("local"):
        result_page = "test_grid_local.html"
    else:
        result_page = "test_grid.html"
    return render_template(result_page,
                           temp_id=get_qid_for_query(),
                           query_id=qa_sys_configs["search_qid"],
                           search_query=qa_sys_configs["search_query"],
                           search_source=qa_sys_configs["search_source"],
                           search_generator=qa_sys_configs["search_generator"],
                           search_mode=qa_sys_configs["search_mode"],
                           qa_prefix=qa_prefix,
                           )

@app.route('/update', methods=['POST'])
def update():
    # search函数只用来做初始的页面渲染，页面后续更新是通过update这个函数（与后端问答系统联系）
    # 需要考虑极端异常情况的处理（extractor）
    print("update后端响应")
    qa_sys_configs = {}
    run_time = request.form['qd_run_time']
    qa_sys_configs["search_query"] = request.form['search_query']
    qa_sys_configs["search_qid"] = request.form['search_query_id'] #qid主要用来区分query内容相同的多次查询
    qa_sys_configs["search_source"] = request.form['search_source']
    qa_sys_configs["search_generator"] = request.form['search_generator']
    qa_sys_configs["search_mode"] = request.form['search_mode']
    print("收到调用, run_time为{}, qid为{}, query为{}".format(run_time, qa_sys_configs["search_qid"], qa_sys_configs["search_query"]))
    state_data, done = run_state_step(qa_sys_configs) # 注意这里的done是“一次问答系统的运行过程”是否完成运行
    last_step = state_data.get_last_step()
    response = {"process_step": [], "answer": "", "reason_chain": "", "docs": [], "done": 0}
    # 后面很快看下啊如果在这里加入<i>图标，html页面上能不能正常显示
    if last_step.participant == "step_by_step_bm25_retriever" and isinstance(last_step, TempAnsweringStep):
        response["process_step"].append("<b>candidate retrieved documents(title)</b>: " + last_step.answers)
    elif last_step.participant == "step_by_step_bm25_retriever" and isinstance(last_step, QuestionAnsweringStep):
        response["process_step"].append("<b>retrieved documents(title)</b>: " + last_step.answer)
    elif last_step.participant == "step_by_step_cot_reasoning_gen" and isinstance(last_step, TempAnsweringStep):
        response["process_step"].append("<b>candidate reasoning steps</b>: " + last_step.answers)
    elif last_step.participant == "step_by_step_cot_reasoning_gen" and isinstance(last_step, QuestionAnsweringStep):
        if qa_sys_configs["search_mode"] == "ragot":
            try:
                last_second_step = state_data.get_last_second_step()
                last_third_step = state_data.get_last_third_step()
                response["process_step"].append("<b>vote for best reasoning step</b>: " + last_third_step.vote)
                response["process_step"].append("<b>retrieved documents(title)</b>: " + last_second_step.answer)
                response["process_step"].append("<b>reasoning steps</b>: " + last_step.answer)
            except:
                print("#"*3 + "出现异常" + "#"*10)
                response["process_step"].append("<b>reasoning steps</b>: " + last_step.answer)
        else:
            response["process_step"].append("<b>reasoning steps</b>: " + last_step.answer)
    elif last_step.participant == "step_by_step_exit_controller" and isinstance(last_step, QuestionAnsweringStep):
        response["process_step"].append("<b>exit control</b>: " + last_step.answer)
    elif last_step.participant == "generate_main_question" and isinstance(last_step, QuestionGenerationStep):
        response["process_step"].append("<b>final question</b>: " + last_step.question)
    elif last_step.participant == "answer_main_question" and isinstance(last_step, QuestionAnsweringStep):
        response["process_step"].append("<b>final reasoning steps</b>: " + last_step.answer)
    elif last_step.participant == "extract_answer" and isinstance(last_step, QuestionAnsweringStep):
        response["process_step"].append("<b>final answer</b>: " + last_step.answer)
    else:
        response["process_step"].append("?????")
    print("\n".join(response["process_step"]));
    if done:
        sample_result = {}
        
        titles = state_data["titles"]
        paras = state_data["paras"]
        urls = state_data["urls"]
        assert len(titles) == len(paras) and len(titles) == len(urls), "the num of titles and paras are not equal"
        sample_result["docs"] = [(titles[i], paras[i], urls[i]) for i in range(len(titles))] # 这里注意顺序
        
        sample_result["answer"] = state_data.get_answers()[0] # 后面注意看下这里提取的是否正确
        sample_result["reason_chain"] = state_data.get_answers()[1]
        result_json = load_search_result_json(qa_sys_configs)
        result_json["sample_results"].append(sample_result)
        
        if len(result_json["sample_results"]) < result_json["sample_time"]:
            # 做初始化
            # 下次调用会重新开始一轮，用于sc-ircot
            response["process_step"].append("sample {}-th time...".format(len(result_json["sample_results"])+1))
            init_state_heap(qa_sys_configs)
        else:
            # 开始进行投票，生成result_json的final_result
            result_json["final_result"] = self_consistency_vote(result_json["sample_results"])
            response["docs"] = result_json["final_result"]["docs"]
            response["answer"] = result_json["final_result"]["answer"]
            response["reason_chain"] = result_json["final_result"]["reason_chain"]
            response["done"] = 1
        save_search_result_json(qa_sys_configs, result_json)
    return response

@app.route('/getdoc', methods=['POST'])
def getdoc():
    # search函数只用来做初始的页面渲染，页面后续更新是通过update这个函数（与后端问答系统联系）
    # 需要考虑极端异常情况的处理（extractor）
    print("getdoc后端响应")
    qa_sys_configs = {}
    qa_sys_configs["search_qid"] = request.form['search_query_id']
    docs = get_cur_docs(qa_sys_configs)
    response = {}
    response['docs'] = docs
    return response


@app.route('/detail')
def detail():
    # 这里注意加一下滚动条，避免极端情况下页面文本溢出
    # 结合现在的页面内容调整
    # 按现在的做法，只需要query_id就能读取文件，每次查询的文件是分开保存的
    # 从final_result读取
    qa_sys_configs = {}
    qa_sys_configs["search_qid"] = request.args.get('query_id')
    doc_corpus = request.args.get('doc_corpus')
    doc_title = request.args.get('doc_title')
    docs = load_search_result_json(qa_sys_configs)["final_result"]["docs"]
    doc_para = get_para_for_title(docs, doc_title)
    return render_template('detail.html',
                           corpus=doc_corpus,
                           title=doc_title,
                           para=doc_para,
            )

# Start the Flask server in a new thread
threading.Thread(target=app.run, kwargs={"use_reloader": False}).start()