import os
import json
import copy
from collections import Counter, defaultdict
import re

from diskcache import Cache
from tqdm import tqdm
import ftfy
import hashlib

def get_pid_for_title_paragraph_text(title: str, paragraph_text: str) -> str:
    title = ftfy.fix_text(title.strip())
    paragraph_text = ftfy.fix_text(paragraph_text.strip())

    if paragraph_text.startswith("Wikipedia Title: " + title + "\n"):
        paragraph_text = paragraph_text.replace("Wikipedia Title: " + title + "\n", "").strip()

    if paragraph_text.startswith("Wikipedia Title: " + title + " \n"):
        paragraph_text = paragraph_text.replace("Wikipedia Title: " + title + " \n", "").strip()

    if paragraph_text.startswith("Title: " + title + "\n"):
        paragraph_text = paragraph_text.replace("Title: " + title + "\n", "").strip()

    if paragraph_text.startswith("Title: " + title + " \n"):
        paragraph_text = paragraph_text.replace("Title: " + title + " \n", "").strip()

    title = "".join([i if ord(i) < 128 else " " for i in title]).lower()
    paragraph_text = "".join([i if ord(i) < 128 else " " for i in paragraph_text]).lower()

    title = re.sub(r" +", " ", title)
    paragraph_text = re.sub(r" +", " ", paragraph_text)

    # NOTE: This is more robust, but was done after V2 big exploration.
    # So uncomment it for rerunning evals for those experiments.
    title = re.sub(r" +", "", title)
    paragraph_text = re.sub(r" +", "", paragraph_text)

    pid = "___".join(
        [
            "pid",
            hashlib.md5(title.encode("utf-8")).hexdigest(),
            hashlib.md5(paragraph_text.encode("utf-8")).hexdigest(),
        ]
    )

    return pid

def format_drop_answer(answer_json):
    # 应该是从answer_object中获取答案文本（三种类型中的一种）
    if answer_json["number"]:
        return answer_json["number"]
    if len(answer_json["spans"]):
        return answer_json["spans"]
    # only date possible
    date_json = answer_json["date"]
    if not (date_json["day"] or date_json["month"] or date_json["year"]):
        print("Number, Span or Date not set in {}".format(answer_json))
        return None
    return date_json["day"] + "-" + date_json["month"] + "-" + date_json["year"]

cache = Cache(os.path.expanduser("~/.cache/title_queries"))

class BasicReader:
    def __init__(self, add_paras=False, add_gold_paras=False):
        self.add_paras = add_paras
        self.add_gold_paras = add_gold_paras

    def read_examples(self, file):
        return NotImplementedError("read_examples not implemented by " + self.__class__.__name__)

class DatasetReader(BasicReader):
    def __init__(
        self,
        add_paras=False,
        add_gold_paras=False,
        add_pinned_paras=False,
        pin_position="no_op",
        remove_pinned_para_titles=False,
        max_num_words_per_para=None,
        retriever_host=None,
        retriever_port=None,
    ):
        super().__init__(add_paras, add_gold_paras)
        self.add_paras = add_paras
        self.add_gold_paras = add_gold_paras
        self.add_pinned_paras = add_pinned_paras
        self.pin_position = pin_position
        self.remove_pinned_para_titles = remove_pinned_para_titles
        self.max_num_words_per_para = max_num_words_per_para
        self.retriever_host = retriever_host
        self.retriever_port = retriever_port

        self.qid_to_external_paras = defaultdict(list)
        self.qid_to_external_titles = defaultdict(list)

    def read_examples(self, file):
        # 这里就是reader的主函数。关键是看一下返回的结果“是用来做什么”的，其中每个属性有什么作用
        # 每种数据集的基本格式是一致的（文件的每行是一条数据，数据是一个问题、答案和相关文档组成的字典
        # 看下这里的外部知识库是怎么设计的（不过外部知识库没必要改，应该是有另外的处理过程
        # 推理过程中根据投票选出的thought确定的"gold文档"还是应该想办法记录一下（就算保存所有文档也要记录）
        with open(file, "r") as input_fp:

            for line in tqdm(input_fp):

                if not line.strip():
                    continue

                input_instance = json.loads(line)

                qid = input_instance["question_id"]
                query = question = input_instance["question_text"]
                answers_objects = input_instance["answers_objects"]

                formatted_answers = [  # List of potentially validated answers. Usually it's a list of one item.
                    tuple(format_drop_answer(answers_object)) for answers_object in answers_objects
                ]
                answer = Counter(formatted_answers).most_common()[0][0]
                # 数据集中给出的answer应该也就是最后算评价指标时的ground truth？
                
                output_instance = {
                    "qid": qid,
                    "query": query,
                    "answer": answer,
                    "question": question,
                }
                # 为什么要把query和question分开？？？
                # qid：多步问题的唯一id；pid：应该是document的唯一id
                # paragraph和document应该是同一个东西
                # 不知道为什么要放这一步
                for paragraph in input_instance.get("pinned_contexts", []) + input_instance["contexts"]:
                    assert not paragraph["paragraph_text"].strip().startswith("Title: ")
                    assert not paragraph["paragraph_text"].strip().startswith("Wikipedia Title: ")
                
                #title后面有用吗？？？为什么还有检索title的做法？
                title_paragraph_tuples = []
                
                # pinned_paras只有iirc数据集才需要
                if self.add_pinned_paras:
                    for paragraph in input_instance["pinned_contexts"]:
                        title = paragraph["title"]
                        paragraph_text = paragraph["paragraph_text"]
                        if (title, paragraph_text) not in title_paragraph_tuples:
                            title_paragraph_tuples.append((title, paragraph_text))
                
                if self.add_paras:
                    assert not self.add_gold_paras, "enable only one of the two: add_paras and add_gold_paras."
                    for paragraph in input_instance["contexts"]:
                        title = paragraph["title"]
                        paragraph_text = paragraph["paragraph_text"]
                        if (title, paragraph_text) not in title_paragraph_tuples:
                            title_paragraph_tuples.append((title, paragraph_text))

                if self.add_gold_paras:
                    assert not self.add_paras, "enable only one of the two: add_paras and add_gold_paras."
                    for paragraph in input_instance["contexts"]:
                        if not paragraph["is_supporting"]:
                            continue
                        title = paragraph["title"]
                        paragraph_text = paragraph["paragraph_text"]
                        if (title, paragraph_text) not in title_paragraph_tuples:
                            title_paragraph_tuples.append((title, paragraph_text))

                if self.max_num_words_per_para is not None:
                    title_paragraph_tuples = [
                        (title, " ".join(paragraph_text.split(" ")[: self.max_num_words_per_para]))
                        for title, paragraph_text in title_paragraph_tuples
                    ]

                output_instance["titles"] = [e[0] for e in title_paragraph_tuples]
                output_instance["paras"] = [e[1] for e in title_paragraph_tuples]
                #也就是说正常情况下这里titles和paras都是空的（初始输入，titles和paras需要推理过程中供从外部检索）
                # titles和paras是一一对应的
                
                if self.remove_pinned_para_titles and "pinned_contexts" in input_instance:
                    for paragraph in input_instance["pinned_contexts"]:
                        while paragraph["title"] in output_instance["titles"]:
                            index = output_instance["titles"].index(paragraph["title"])
                            output_instance["titles"].pop(index)
                            output_instance["paras"].pop(index)
                    
                # 这部分设置output_instance中的titles和paras以及pids的意义是什么？？？
                pids = [
                    get_pid_for_title_paragraph_text(title, paragraph_text)
                    for title, paragraph_text in zip(output_instance["titles"], output_instance["paras"])
                ]
                output_instance["pids"] = pids

                output_instance["real_pids"] = [
                    paragraph["id"]
                    for paragraph in input_instance["contexts"]
                    if paragraph["is_supporting"] and "id" in paragraph
                ]

                # 这里算是一种预处理，保留即可
                for para in output_instance["paras"]:
                    assert not para.strip().startswith("Title: ")
                    assert not para.strip().startswith("Wikipedia Title: ")

                # Backup Paras and Titles are set so that we can filter from the original set
                # of paragraphs again and again.
                if "paras" in output_instance:
                    output_instance["backup_paras"] = copy.deepcopy(output_instance["paras"])
                    output_instance["backup_titles"] = copy.deepcopy(output_instance["titles"])
                
                # 这里valid_titles是什么意思？？？好像只有IIRC有
                if "valid_titles" in input_instance:
                    output_instance["valid_titles"] = input_instance["valid_titles"]
                
                # 用于实验的每条数据的元数据（属性）
                output_instance["metadata"] = {}
                output_instance["metadata"]["level"] = input_instance.get("level", None)
                output_instance["metadata"]["type"] = input_instance.get("type", None)
                output_instance["metadata"]["answer_type"] = input_instance.get("answer_type", None)
                output_instance["metadata"]["simplified_answer_type"] = input_instance.get(
                    "simplified_answer_type", None
                )

                if self.add_pinned_paras:
                    assert len(input_instance["pinned_contexts"]) == 1
                    output_instance["metadata"]["pinned_para"] = input_instance["pinned_contexts"][0]["paragraph_text"]
                    output_instance["metadata"]["pinned_title"] = input_instance["pinned_contexts"][0]["title"]
                output_instance["metadata"]["pin_position"] = self.pin_position

                # 注意下面这部分会把数据集给定的问题的contexts中标为"is_supporting"的文档作为gold文档保存
                output_instance["metadata"]["gold_titles"] = []
                output_instance["metadata"]["gold_paras"] = []
                output_instance["metadata"]["gold_ids"] = []
                for paragraph in input_instance["contexts"]:
                    if not paragraph["is_supporting"]:
                        continue
                    title = paragraph["title"]
                    paragraph_text = paragraph["paragraph_text"]
                    output_instance["metadata"]["gold_titles"].append(title)
                    output_instance["metadata"]["gold_paras"].append(paragraph_text)
                    output_instance["metadata"]["gold_ids"].append(paragraph.get("id", None))

                yield output_instance