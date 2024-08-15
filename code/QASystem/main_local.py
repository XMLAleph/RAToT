"""
# 后续安排
    1.实现动态刷新（非常重要，Sc-IRCoT和RAGoT按现在单次请求的方式基本上会超时（重复发两次请求应该就是超时导致的），必须尽快实现）
        必须“最快速度”实现
    2.优化网页模板（现在不清楚怎么在模块宽度或高度固定的情况下保证模块大小合适（比如文本非常长的情况下会超出模块里的外部框）
    3.做答辩PPT（根据论文内容整理思路，在之前PPT和已有图表的基础上完善，不另外做汇报PPT，直接做答辩用的PPT）
    
    4.整理LLM RAG后续思路（应该涉及到模型接口的改变，需要考虑），最优先必须尽快完成毕业论文剩余的修改（1.问答系统调整；2.PPT）
    
# 目前完成：

# 需要：
    1.实现网页的动态刷新
        这种动态刷新应该是通过ajax实现的，具体运行流程还不清楚，猜测是点击搜索后，先从前端向后端发一次初始请求，后端响应后生成初始的前端页面，然后前端页面再通过ajax向后端发送请求，后端再响应，生成第一次更新的前端页面，以此类推
        # 这个过程如果设计成通过点按钮往下进行的应该也能解释，主要是自动往下进行不知道具体怎么操作
            自动进行应该相当于一收到后端响应就立刻发下一次请求
            后端不等得到最后结果才一次性返回，而是每生成一段结果就传给前端
            另外考虑页面样式
            依次生成结果在后端可以是主动进行的，但实现上应该是前端发送请求，后端根据请求生成结果
            不过前端请求应该不需要传数据，只是后端每一步需要额外等前端的信号（前端每发一次请求，就往后执行一步）
            执行一步就是检索文档+生成推理步骤 但后端运行过程并不是完全规律的（最后qa环节的逻辑可能和前面的过程有区别），再看下后端的代码
            动态更新（流式输出）要能体现出框架在做什么（独特之处）（实际的运行过程）
            另外Sc-IRCoT的流程到底怎么呈现，需要考虑
            目前的思路是：首先从首页输入一个query，进行第一步处理（初始化state，返回值可以是[图标：结果]）后直接跳转到结果页面。结果页面自动调用js函数，向后端发送第一次请求，后端往后执行一步操作（检索或推理），将结果返回给前端；前端success后，首先向页面添加后端新返回的元素，然后if 后端传来的结果未终止（设法判断），就迭代调用，继续向后端发请求；else 不再进行操作，页面动态加载完成
                关键是js函数里怎么嵌套调用函数
                按现在的想法大概测试一下中间不确定的点
                先测试这种调用js函数的方式（顺便测试在页面中新增元素），然后进一步换成基于ajax前后端通信的函数
                至少js迭代调用函数是可以实现的(test_flask_ajax.py)
        # 另外这个过程怎么终止？感觉按问答系统的实际运行过程，应该由后端发送信号（生成过程结束）来决定，但不清楚能不能实现这种操作
        # 另外每次从前端动态获取的数据具体怎么处理（怎么变成html页面的一部分）目前还不清楚
    
        # 每次新增一条flex布局的数据，比如第一次新增一条（图标，推理步骤），第二次新增一条（图标，[检索结果]）
        
"""
from flask import Flask, render_template, request, redirect
from flask_paginate import Pagination
#from RAGoT.main import perform_qa
from flask_utils import get_recommend_querys, get_qid_for_query, search_from_temp_json, save_temp_result_json, perform_qa, get_para_for_title
app = Flask(__name__)

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
        answer, reason_chain, relevant_docs = perform_qa(query)
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
    
    #没有其他元素的情况下，跳转后页面的渐变色背景能生效。原始页面用纯色背景也能生效，但换成渐变色背景就无法显示了
    
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


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)