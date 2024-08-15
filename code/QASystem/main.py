# 后面运行的时候应该需要把Sc_IRCoT和RAGoT文件夹放在QASystem文件夹里（和main.py在同一个目录下）
from flask import Flask, render_template, request, url_for, redirect
import getpass
import threading
from flask_paginate import Pagination
from llm_qa_systems.Sc_IRCoT.inference import perform_qa_1
#from llm_qa_systems.RAGoT.main import perform_qa_2
from flask_utils import get_recommend_querys, get_qid_for_query, search_from_temp_json, save_temp_result_json, get_para_for_title
app = Flask(__name__)
from pyngrok import ngrok, conf

#print("Enter your authtoken, which can be copied from https://dashboard.ngrok.com/get-started/your-authtoken")
conf.get_default().auth_token = '2ezwbmwxGCGRDAUC8K5qogVwVYz_5iQGCBn2iRbWneewX65dM'
conf.get_default().request_timeout = 999

app = Flask(__name__)

# Open a ngrok tunnel to the HTTP server
public_url = ngrok.connect(5000).public_url
print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}/\"".format(public_url, 5000))

# Update any base URLs to use the public ngrok URL
app.config["BASE_URL"] = public_url

#app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', temp_id=get_qid_for_query(), recommend_querys=get_recommend_querys())

@app.route('/search', methods = ['GET','POST'])
def search():
    ####
    #query = request.form['inputs']
    qa_sys_configs = {}
    query_id = request.args.get("id4query") # 这个是必定返回的一个值，只有当通过搜索框按钮跳转页面时，才会改变id4query的值。在翻页等其他情况下，这个值不会变化
    query = request.args.get('inputs', "", type=str) #根据query是否为空来判断，如果query为空，从临时文件中读取数据；如果query不为空，调用问答系统进行回答（重新检索和新检索都是这种情况）
    # 另外也考虑query为空的特殊情况（可以重定向到首页，不是目前主要的问题）
    if not query:
        return redirect('https://www.bing.com', 302)
    qa_sys_configs["searchsource"] = request.args.get("menu4searchsource", "", type=str)
    qa_sys_configs["llmgenerator"] = request.args.get("menu4llmgenerator", "", type=str)
    qa_sys_configs["mode"] = request.args.get("menu4mode", "", type=str)
    page = request.args.get('page', 1, type=int) # 这里是分页方法现在主要的问题，想一下怎么办 # 看下直接使用的话能不能生效（主要是要和其他部分一起）
    # 先进行查询，如果查询不到就重新调用问答系统
    result = search_from_temp_json(query_id, query)
    if result:
        answer = result["answer"]
        reason_chain = result["reason_chain"]
        relevant_docs = result["docs"]
    else: # 从index.html或result.html的搜索框输入新查询或重新查询的情况
        if qa_sys_configs["mode"] == "ragot":
            answer, reason_chain, relevant_docs = perform_qa_1(query, qa_sys_configs)
        else:
            answer, reason_chain, relevant_docs = perform_qa_1(query, qa_sys_configs)
        save_temp_result_json(query_id, query, answer, reason_chain, relevant_docs)
    ####
    
    pageSize = 3 # 超参数
    
    if page > len(relevant_docs) or page < 1:
        page = 1

    # 对获取到的数据进行切片
    start = (page - 1) * pageSize	# 开始，每一页开始位置
    end = start + pageSize			# 结束，每一页结束位置
    slices = slice(start, end)
    sli_docs = relevant_docs[slices]

    # 分页对象
    current_page = Pagination(relevant_docs, page=page, per_page=pageSize, total=len(relevant_docs), items=relevant_docs[page-1],prev_label="<i class='iconfont' style='font-size: 15px'>&#xe67c;</i>",next_label="<i class='iconfont' style='font-size: 15px'>&#xe68b;</i>")
    
    if qa_sys_configs["searchsource"].startswith("local"):
        page_html = "localresult.html"
    elif qa_sys_configs["searchsource"].startswith("web"):
        page_html = "webresult.html"
        
    return render_template(page_html,
                           query_id=query_id,
                           search_source=qa_sys_configs["searchsource"],
                           search_query=query,
                           search_answer=answer,
                           answer_chain=reason_chain, 
                           search_docs=sli_docs,
                           current_page=current_page,
                           temp_id=get_qid_for_query(),
            )


@app.route('/detail')
def detail():
    query_id = request.args.get('query_id')
    doc_corpus = request.args.get('doc_corpus')
    doc_title = request.args.get('doc_title')
    doc_query = request.args.get('doc_query')
    docs = search_from_temp_json(query_id, doc_query)["docs"]
    doc_para = get_para_for_title(docs, doc_title)
    return render_template('detail.html',
                           corpus=doc_corpus,
                           title=doc_title,
                           para=doc_para,
            )

# Start the Flask server in a new thread
threading.Thread(target=app.run, kwargs={"use_reloader": False}).start()