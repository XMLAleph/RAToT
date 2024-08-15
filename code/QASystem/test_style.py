from flask import Flask, render_template, request, redirect
from flask_utils import get_recommend_querys, get_qid_for_query, get_para_for_title, init_search_result_json, load_search_result_json, save_search_result_json, get_cur_docs
#from llm_qa_systems.run import init_state_heap, run_state_step, self_consistency_vote
#from llm_qa_systems.data_instances import QuestionAnsweringStep, AnswerVotingStep, TempAnsweringStep, QuestionGenerationStep
app = Flask(__name__)

"""
position: absolute;
margin-left: 80px;
margin-bottom: 10px;
"""

# padding: 20px;
# 先实现完整的功能（调整detail）
# 调整loading的位置（loading和最终页面上的答案是不同的类，位置可以分别定义
@app.route('/')
def index():
    return render_template('index.html', 
                           temp_id=get_qid_for_query(), 
                           recommend_querys=get_recommend_querys()
                           )

@app.route('/search', methods = ['GET','POST'])
def search():
    # 先测试下当前页面上的ajax动态更新数据（看下button类型作为文本框，超链接文本能不能生效，先不和后端代码结合
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
    # 先看下实现
    print("页面初始化完成，渲染初始页面")
    return render_template('test_style.html',
                           temp_id=get_qid_for_query(),
                           query_id=qa_sys_configs["search_qid"],
                           search_query=qa_sys_configs["search_query"],
                           search_source=qa_sys_configs["search_source"],
                           search_generator=qa_sys_configs["search_generator"],
                           search_mode=qa_sys_configs["search_mode"],
                           qa_prefix=qa_prefix,
                           )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)