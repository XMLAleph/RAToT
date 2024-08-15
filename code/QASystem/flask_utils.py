import random
import json
import os
import hashlib

def get_recommend_querys():
    recommend_query_num = 2
    local_querys_files = ["llm_qa_systems/processed_data/hotpotqa/dev_subsampled.jsonl","llm_qa_systems/processed_data/2wikimultihopqa/dev_subsampled.jsonl","llm_qa_systems/processed_data/musique/dev_subsampled.jsonl"] # 各个数据集文件的路径，视最后具体的文件结构而定
    random_file_index = random.randint(0,len(local_querys_files)-1)
    local_querys_file = local_querys_files[random_file_index]
    random_item_indexs = []
    item_indexs = range(1,100-1)
    random_item_indexs = random.sample(item_indexs, recommend_query_num)
    random_items = []
    with open(local_querys_file, 'r+') as f:
        line = f.readline()
        item_index = 0
        while line:
            for random_item_index in random_item_indexs:
                if item_index == random_item_index:
                    random_items.append(json.loads(line)["question_text"])
            if len(random_items) == len(random_item_indexs):
                break
            line = f.readline()
            item_index += 1
    return random_items

'''
def get_recommend_querys():
    return ["Who is the founder of the company which published Big Picture (Magazine)?", "Are Antoine Jean-Baptiste Thomas and Canardo (Rapper) of the same nationality?"]
'''

def get_qid_for_query():
    # def index: return render_template('index.html', random_function())
    # 看下这种方式能不能在每次打开网页时显示随机内容
    # 每调用一次这个函数，就会生成一个“唯一”的哈希值（同一个query多次调用生成的哈希值也不一样）
    # 如果是同一次查询中翻页，get到的qid不会变化（qid每次进入新页面都会生成，但只有提交搜索框表单时才会返回，这样就无法从临时文件里查到数据
    hash_qid = "___".join(
        [
            "qid",
            hashlib.md5(str(random.random()).encode("utf-8")).hexdigest(),
        ]
    )
    return hash_qid

def search_from_temp_json(qid, query):
    temp_json_path = "res.jsonl" # 固定的位置
    if not os.path.exists(temp_json_path):
        return None
    with open(temp_json_path, "r") as temp_f:
        for line in temp_f:
            if not line.strip():
                continue
            item = json.loads(line)
            if (item["qid"] == qid) and (item["query"] == query):
                return item
    return None

def save_temp_result_json(qid, query, answer, reason_chain, relevant_docs):
    # 将问答系统返回的结果写入临时文件
    temp_json_path = "res.jsonl" # 固定的位置
    item = {
        "qid":qid,
        "query":query,
        "answer":answer,
        "reason_chain":reason_chain,
        "docs":relevant_docs,
        } # 追加写
    with open(temp_json_path, "a+") as temp_f:
        json.dump(item, temp_f)
        temp_f.write('\n')
    return

def init_search_result_json(qa_sys_configs):
    # 固定位置保存
    query = qa_sys_configs["search_query"]
    qid = qa_sys_configs["search_qid"]
    searchsource = qa_sys_configs["search_source"].split('_') # ['local','hotpotqa'] web_bing
    #llmgenerator = qa_sys_configs["llmgenerator"]
    mode = qa_sys_configs["search_mode"] # ircot scircot ragot
    sample_time = 1
    if mode == "scircot":
        sample_time = 8
    json_result = {
        "qid":qid,
        "query":query,
        "search_source":searchsource,
        "mode":mode,
        "sample_time":sample_time,
        "sample_results":[],
        "final_result":[],
        } # 一轮运行结束后，将结果（answer, docs）append到sample_results中，再通过len(sample_results)判断是否达到指定的运行轮数。如果达到，对前端发送终止信号，前端除了更新process的内容，也会更新result的内容；如果没达到，对前端发送非终止信号，并重新init heap，重新开始运行
    #print("初始化:{}".format(json_result))
    save_search_result_json(qa_sys_configs, json_result)
    return

def load_search_result_json(qa_sys_configs):
    qid = qa_sys_configs["search_qid"]
    json_path = str(qid) + ".json"
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r") as json_f:
        for line in json_f:
            if not line.strip():
                continue
            json_result = json.loads(line)
    return json_result #字典类型

def save_search_result_json(qa_sys_configs, json_result):
    qid = qa_sys_configs["search_qid"]
    json_path = str(qid) + ".json"
    with open(json_path, "w") as json_f: # 覆盖原有的内容
        json.dump(json_result, json_f)
        json_f.write('\n')
    return

def get_cur_docs(qa_sys_configs):
    qid = qa_sys_configs["search_qid"]
    json_path = str(qid) + ".json"
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r") as json_f:
        for line in json_f:
            if not line.strip():
                continue
            json_result = json.loads(line)
    try:
        return json_result["sample_results"][0]["docs"]
    except:
        return []
    
def perform_qa(query):
    answer = f"I am the answer for {query}"
    reason_chain = f"I am the reasoning chain for {query}"
    titles = ["Big Picture (magazine)", "Picture Play (magazine)", "Warren Publishing","The Big Picture", "Current Publishing (UK)", "Highline Big Picture", "James Brown (publisher)", "James Warren (politician)", "Diane Warren", "1984 (magazine)", "Jamin Warren"]
    relevant_docs = [(titles[i], f"I am the {i}-th content for {query}\n","https://www.bing.com") for i in range(len(titles))]
    return answer, reason_chain, relevant_docs

def get_para_for_title(docs, doc_title):
    for doc in docs:
        if doc[0] == doc_title:
            return doc[1] #docs是嵌套列表，docs[i] = [title, para, url]，这个再和问答系统统一一下
    return f"content4{doc_title}..."*20