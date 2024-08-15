"""
# 这里相当于IRCoT的ircot.py，完成框架中每个模块的定义
# 需要的基本模块包括：1.检索器；2.生成器（会生成多个thought，采样多次）；3.投票器；4.控制器（控制每一轮结束后是否退出）

# 之前检索到的相关文档的处理有3种可能的方式：
    1.不保留之前检索到的文档（cumulate_former_results=False，choose_correspond_doc=False）
    2.只保留之前每一步检索到的文档中的"gold文档"（投票选出的thought对应的文档）（cumulate_former_results=False，choose_correspond_doc=False）
    3.保留之前检索到的所有文档（cumulate_former_results=True，choose_correspond_doc=False）
    
相关的参数：
    1.retriever的cumulate_former_results（=True：保留之前的相关文档；=False：覆盖之前的相关文档）
    2.voter的choose_correspond_doc（=True：需要assert cumulate_former_results也是true，只保留state.data['titles/paras']中得分最高的thought对应的相关文档。=False：不修改state.data中的相关文本
    这个参数主要是根据voter的结果，修改retriever得到的state.data['paras']

voting的输入是多个当前的thought，输出是经过投票或打分后重新排序的thought，并更新state
另外还有一个实际问题，实际中一次性检索到的文档可能有多种，如果只保留其中一个的话，感觉会影响效果
而且之后是用生成的内容作为query来检索，而不是原问题，所以之前检索到但没有被保留下来的相关文档后面可能很难再被检索到了，这可能也会影响效果

不过其实这种推迟到vote环节再将相应的answer类型数据加到state.data中，可能并不会打乱顺序（因为retriever和generator的结果都需要在vote环节处理，应该可以按原来的顺序加）
投票的对象可以是从开始到目前为止生成的内容（而不仅是新生成的一句内容），这样理论上越到后面选择的合理性越高

如果voter的结果出现问题（没有正常地选出一个结果）怎么办？一种可能的后备方案是这种情况就换成ircot的原始方法（这种情况下，所有文档都会加入state.data中
如果需要的话后面再实现这种情况，先尽快看LLM实际返回的结果

这个方法主要的问题就是，如果一步生成需要用到两个以上的相关文档，就会出问题（理想情况下每一步推理只需要用到一个，但模型不一定能进行这种非常理想化的生成

把ircot的基本做法也作为一个备选项，尽量设法处理在一步生成中需要用到两个以上的相关文档的情况，这样理论上效果至少和ircot接近

至少调试阶段可以考虑这种做法，看下怎么实现(可以考虑把ircot基本做法的结果放到固定的位置，如果投票出现问题就直接选择这个结果)

特别注意state和new_state！！！！！！已经引入了new_state的情况下，有些操作仍然是对state进行的，会导致出错！
"""

import re
import copy
import json
from typing import List
from functools import lru_cache
import random
from tqdm import tqdm

from rapidfuzz import fuzz

from prompt_reader import read_prompt, read_prompt_easy
from data_instances import QuestionAnsweringStep, AnswerVotingStep, TempAnsweringStep, QuestionGenerationStep, Task

from Retriever.ela_retriever import UnifiedRetriever
from Generators.gpt3generator import GPT3Generator
from Generators.llmgenerator import LLMGenerator

from Dataset_readers import get_pid_for_title_paragraph_text

random.seed(100)  # Don't change.

@lru_cache(maxsize=None)
def get_spacy_object():
    import spacy

    return spacy.load("en_core_web_sm")

def is_reasoning_sentence(sentence: str) -> bool:
    starters = ["thus ", "thus,", "so ", "so,", "that is,", "therefore", "hence"] # 这里依赖一个人工指定的列表来判断是否是推理步骤
    for starter in starters:
        if sentence.lower().startswith(starter):
            return True

    regex = re.compile("(.*)(\d[\d,]*\.?\d+|\d+) ([+-]) (\d[\d,]*\.?\d+|\d+) = (\d[\d,]*\.?\d+|\d+)(.*)")
    # 这个正则匹配式是什么意思？？？
    match = bool(re.match(regex, sentence))
    if match:
        return True

    return False

def remove_reasoning_sentences(sentences: List[str]) -> List[str]:
    return [sentence for sentence in sentences if not is_reasoning_sentence(sentence)]

def remove_wh_words(text: str) -> str:
    # 这里为什么要把这些词去掉???
    wh_words = {"who", "what", "when", "where", "why", "which", "how", "does", "is"}
    words = [word for word in text.split(" ") if word.strip().lower() not in wh_words]
    text = " ".join(words)
    return text

# 框架中模块的基本类
class BasicModule(object):
    def run(self, state, debug=False):
        """The main function that interfaces with the overall search and
        model controller, and manipulates the incoming data.

        :param state: the state of controller and model flow.
        :type state: launchpadqa.question_search.model_search.SearchState
        :rtype: list
        """
        raise NotImplementedError("Must implement to work inside of controller!")

    def return_model_calls(self):
        # 这个函数是返回当前模块已经执行的次数
        """
        :return: a dict of <model_name, number of calls> made by this participant
        """
        raise NotImplementedError("Must implement to work inside of controller!")

def is_para_closely_matching(
    existing_titles: List[str],
    existing_paras: List[str],
    new_title: str,
    new_para: str,
    match_threshold: float = 90,
) -> bool:

    if new_title in existing_titles and new_para in existing_paras:
        return True

    assert match_threshold > 1.0, "The threshold is 0-100 scaled."

    assert len(existing_titles) == len(existing_paras)
    for existing_title, existing_para in zip(existing_titles, existing_paras):
        condition_1 = fuzz.ratio(existing_title, new_title) >= match_threshold
        condition_2 = fuzz.ratio(existing_para, new_para) >= match_threshold
        if condition_1 and condition_2:
            return True
    return False

def para_to_text(title: str, para: str, max_num_words: int) -> int:
    # Note: the split and join must happen before the attaching title+para.
    # also don't split() because that disrupts the new lines.
    para = " ".join(para.split(" ")[:max_num_words])
    para = (
        para.strip()
        if para.strip().startswith("Wikipedia Title: ")
        else "Wikipedia Title: " + title + "\n" + para.strip()
    )
    return para

def add_and_reorder_if_pinned(titles, paras, pinned_title, pinned_para, pin_position):

    if pinned_title is not None or pinned_para is not None:
        assert pinned_title is not None and pinned_para is not None

        if pinned_para not in paras:
            titles.insert(0, pinned_title)
            paras.insert(0, pinned_para)

        pin_index = paras.index(pinned_para)
        assert titles[pin_index].lower().strip() == pinned_title.lower().strip()

        if pin_position == "no_op":
            return titles, paras

        elif pin_position == "top":
            titles.pop(pin_index)
            paras.pop(pin_index)
            titles.insert(0, pinned_title)
            paras.insert(0, pinned_para)

        elif pin_position == "bottom":
            titles.pop(pin_index)
            paras.pop(pin_index)
            titles.append(pinned_title)
            paras.append(pinned_para)

        else:
            raise Exception(f"Unknown pin_position {pin_position}")

    return titles, paras

def assert_unique_titles_paras(titles: List[str], paras: List[str]) -> bool:
    titles_paras = [(title, para) for title, para in zip(titles, paras)]
    assert len(titles_paras) == len(set(titles_paras))
    
class Retriever(BasicModule):
    def __init__(
        self,
        retrieval_type,
        retrieval_count=None,
        query_source="last_answer",
        cumulate_former_results=False,
        document_type="title",
        source_corpus_name=None,
        dont_add_to_state=False,
        allowed_paragraph_types=None,
        dont_skip_long_paras=False,
        return_pids=False,
        return_paras=False,
        valid_titles_are_allowed_titles=False,
        set_result_as_valid_titles=False,
        global_max_num_paras=100,
        next_model=None,
        end_state="[EOQ]",
    ):
        assert retrieval_type in (
            "map_generated_to_valid_titles",
            "bm25"
        ), f"retrieval_type {retrieval_type} not among the valid choices."

        # 实际运行的时候取值是question_or_last_generated_sentence
        assert query_source in (
            "original_question",
            "last_answer",
            "question_or_last_generated_sentence",
        ), f"query_source {query_source} not among the valid choices."
        
        # 实际运行的时候取值是title_paragraph_text
        assert document_type in (
            "title", 
            "paragraph_text", 
            "title_paragraph_text"
        ), f"document_type {document_type} not among the valid choices."
        
        self.valid_titles_are_allowed_titles = valid_titles_are_allowed_titles
        if valid_titles_are_allowed_titles:
            assert (
                retrieval_type == "bm25"
            ), "valid_titles_are_allowed_titles is applicable only when retrieval_type is bm25."

        if set_result_as_valid_titles:
            assert (
                retrieval_type == "map_generated_to_valid_titles"
            ), "set_result_as_valid_titles is only available for map_generated_to_valid_titles retrieval type."
        self.set_result_as_valid_titles = set_result_as_valid_titles

        self.global_max_num_paras = global_max_num_paras
        self.retrieval_type = retrieval_type
        self.next_model = next_model
        self.end_state = end_state # 这些信息这里也需要吗？
        self.retrieval_count = retrieval_count # 这里想一下后面怎么调合适
        self.document_type = document_type
        self.query_source = query_source
        self.cumulate_former_results = cumulate_former_results
        self.source_corpus_name = source_corpus_name
        self.dont_add_to_state = dont_add_to_state
        self.dont_skip_long_paras = dont_skip_long_paras
        self.return_pids = return_pids
        self.return_paras = return_paras
        self.num_calls = 0
        
        # 具体实现的时候还有个问题，每一步找到的文档要不要一直叠加，如果叠加的话怎么叠加（因为理论上能找到相关文档，就一直只保留相关文档即可？
        # 这个方法对模型本身能力的要求其实也比较高
        if self.return_pids and self.return_paras:
            raise Exception("Only one of return_pids or return_paras should be true.")

        if allowed_paragraph_types:
            assert isinstance(allowed_paragraph_types, list)
            self.allowed_paragraph_types = allowed_paragraph_types
        else:
            self.allowed_paragraph_types = [None] #这里到底是什么情况？？？确实是none，不知道是什么作用，但这里[None]能保证整个检索过程正好运行一次，没有其他意义

        if retrieval_type == "bm25":
            if self.retrieval_count is None:
                raise Exception(f"retrieval_count is needed for the retrieval_type {retrieval_type}.")
            if self.source_corpus_name is None:
                raise Exception(f"source_corpus_name is needed for the retrieval_type {retrieval_type}.")

        self.retrieval_failures_so_far = 0
        self.retrieval_failures_max = 9
        
        self.retriever = UnifiedRetriever()
        self.retrieval_method = "retrieve_from_elasticsearch"
        assert self.retrieval_method in ("retrieve_from_elasticsearch")
        
    def return_model_calls(self):
        return {"paragraph_retrieve_and_reset": self.num_calls}

    def run(self, state, debug=False):
        # run方法根据输入的查询query(第一步是问题，后面是“最近一步的推理步骤”)检索k篇相关文档
        # 看下query_source这里具体是怎么定义的
        # 每次用于检索的query到底是什么？？？（第一次是原始问题，后面是什么？）这个特别注意打印一下
        if self.query_source == "original_question":
            input_query = state.data["question"]

        elif self.query_source == "last_answer":
            input_query = state.data.get_last_answer() # 有初始值吗？

        elif self.query_source == "question_or_last_generated_sentence":
            # 从配置来看实际中query_source是这一种（这一点必须确认清楚）
            # add question to query only if generated sentences are empty. O/w use last_generated_sentence.
            # 这里有个问题，从第二步开始用于检索的last generated sentence是“非推理步骤”的句子中最近生成的一个，为什么可以是非推理步骤？
            # 感觉这种做法也不一定合理，看下实际情况
            question = state.data["question"]
            generated_sentences = state.data.get("generated_sentences", [])
            # 看下generated_sentences具体是什么，在推理过程中是怎么变化的
            generated_sentences = remove_reasoning_sentences(generated_sentences)
            last_generated_sentence_str = generated_sentences[-1].strip() if generated_sentences else ""
            input_query = last_generated_sentence_str if last_generated_sentence_str else question
        else:
            raise Exception(f"Unknown query_source: {self.query_source}.")
        
        if debug:
            print("Input Query: {}".format(input_query))
            
        if not self.cumulate_former_results:
            state.data["titles"] = []
            state.data["paras"] = []
            
        selected_titles = state.data["titles"]
        selected_paras = state.data["paras"]
        
        # 还有一个问题是，对cumulate_former_results的设置也会影响map_generated_to_valid_titles这种IIRC特定的处理过程            
        if self.retrieval_type == "map_generated_to_valid_titles":
            try:
                # Assuming input_query will be of form: '["title3", "title7"]''
                generated_titles = json.loads(input_query)
                # 这里是不是把问题作为关键词来做关键词检索？
            except:
                generated_titles = [
                    e.strip() for e in input_query.strip().replace('"', "").replace("[", "").replace("]", "").split(",")
                ]

            for generated_title in generated_titles:

                assert self.document_type == "title"
                params = {
                    "query_text": generated_title,
                    "max_hits_count": self.retrieval_count,
                    "document_type": "title",
                    "corpus_name": self.source_corpus_name,
                }

                locally_mapped_titles = set()
                
                results = getattr(self.retriever, self.retrieval_method)(**params)

                for result in results:
                    selected_title = result["title"]
                    selected_para = result.get("paragraph_text", "")  # backoff for natcq

                    locally_mapped_titles.add(selected_title)

                    if len(selected_para.split(" ")) > 600 and not self.dont_skip_long_paras:
                        print("WARNING: Discarding a retrieved paragraph as it's excessively long.")
                        continue

                    if result["corpus_name"] != self.source_corpus_name:
                        raise Exception(
                            f"The retrieved corpus name {result['corpus_name']} "
                            f"doesn't match {self.source_corpus_name}."
                        )

                    if selected_title not in selected_titles and len(selected_paras) < self.global_max_num_paras:
                        selected_titles.append(selected_title)
                        selected_paras.append(selected_para)

            if self.set_result_as_valid_titles:
                state.data["valid_titles"] = selected_titles
            
            answer = json.dumps(selected_titles)
            if self.return_pids:
                pids = [
                    get_pid_for_title_paragraph_text(title, paragraph_text)
                    for title, paragraph_text in zip(selected_titles, selected_paras)
                ]
                answer = json.dumps(pids)
            if self.return_paras:
                answer = json.dumps(
                    [{"title": title, "paragraph_text": para} for title, para in zip(selected_titles, selected_paras)]
                )
            
            new_state = state.copy()
            new_state.data.add_answer(QuestionAnsweringStep(answer=answer, score=0, participant=state.next)) #这里的State是原来的state，原来state的next command也就是new state的participant
            new_state.next = self.next_model if self.next_model else self.end_state

            if not self.dont_add_to_state:
                new_state.data["paras"] = selected_paras
                new_state.data["titles"] = selected_titles

            
        elif self.retrieval_type == "bm25":
            
            temp_titles = []
            temp_paras = []
            
            input_query = remove_wh_words(input_query)

            params = {
                "query_text": input_query,
                "max_hits_count": self.retrieval_count,
                "corpus_name": self.source_corpus_name,
            }

            for allowed_paragraph_type in self.allowed_paragraph_types:
                if allowed_paragraph_type is not None:
                    params["allowed_paragraph_types"] = [allowed_paragraph_type]
                
                if not input_query.strip():
                    # can happen when query is based on last cot gen
                    # but it's a reasoning (non-factual) sentence.
                    continue

                if self.retrieval_type == "bm25":
                    params["document_type"] = self.document_type

                if self.valid_titles_are_allowed_titles:
                    params["allowed_titles"] = state.data["valid_titles"]
                
                results = getattr(self.retriever, self.retrieval_method)(**params)
                
                for result in results:

                    if result["corpus_name"] != self.source_corpus_name:
                        raise Exception(
                            f"The retrieved corpus name {result['corpus_name']} "
                            f"doesn't match {self.source_corpus_name}."
                        )

                    if len(result["paragraph_text"].split(" ")) > 600 and not self.dont_skip_long_paras:
                        print("WARNING: Discarding a retrieved paragraph as it's excessively long.")
                        continue

                    if is_para_closely_matching(
                        selected_titles,
                        selected_paras,
                        result["title"],
                        result["paragraph_text"],
                    ):
                        continue

                    if len(selected_paras) >= self.global_max_num_paras:
                        continue

                    if self.valid_titles_are_allowed_titles: # 这个应该是IIRC数据集用的？？？
                        assert result["title"].lower().replace(" ", "") in [
                            valid_title.lower().replace(" ", "") for valid_title in state.data["valid_titles"]
                        ]
                    temp_titles.append(result["title"])
                    temp_paras.append(result["paragraph_text"])
            
            new_state = state.copy()
            if "temp_titles" not in state.data:
                new_state.data['temp_titles'] = []
                new_state.data['temp_paras'] = []
            # 还有个问题，participant这里也不太好控制（因为在vote那里再做这一步的话，paticipant已经变化了
            new_state.next = self.next_model if self.next_model else self.end_state

            if not self.dont_add_to_state: # dont_add_to_state=False时，才会将这一次检索到的文档加入到state的data部分中
                new_state.data["temp_paras"] = temp_paras
                new_state.data["temp_titles"] = temp_titles
                new_state.data['temp_retriever_participant'] = state.next
                    
        else:
            raise Exception(
                f"retrieval_type must be one of 'map_generated_to_valid_titles', 'bm25'. Found {self.retrieval_type}."
            )

        self.num_calls += 1
        
        return new_state
    
        
class TitleGenerator(BasicModule):
    """Goes with StepByStepCOTGenParticipant"""

    def __init__(
        self,
        retrieval_count,
        prompt_file,
        prompt_reader_args,
        show_so_far_titles,
        show_so_far_paras,
        show_so_far_cot,
        prompt_question="",
        max_para_num_words=350,
        gen_model="gpt3",
        next_model=None,
        question_prefix="",
        end_state="[EOQ]",
        **kwargs,
    ) -> None:

        self.num_calls = 0
        self.next_model = next_model
        self.end_state = end_state

        assert isinstance(retrieval_count, int)
        assert isinstance(show_so_far_titles, bool)
        assert isinstance(show_so_far_paras, bool)
        assert isinstance(show_so_far_cot, bool)

        self.retrieval_count = retrieval_count
        self.show_so_far_titles = show_so_far_titles
        self.show_so_far_paras = show_so_far_paras
        self.show_so_far_cot = show_so_far_cot

        tpc_combination = "".join(
            ("Y" if show_so_far_titles else "N", "Y" if show_so_far_paras else "N", "Y" if show_so_far_cot else "N")
        )
        valid_tpc_combinations = ("NNN", "NYN", "NNY", "YNY", "YNN", "YYN", "YYY")
        # The NNN and NYN are only for the base condition, when no contextual info is available
        # NNN when paras are not pinned, NYN when they are pinned.
        assert tpc_combination in valid_tpc_combinations, f"given tpc_combination ({tpc_combination}) is not valid."

        if prompt_file:
            prompt_reader_args = prompt_reader_args or {}
            prompt_reader_args["file_path"] = prompt_file
            self.prompt = read_prompt(**prompt_reader_args)
        else:
            print("WARNING: Using StepByStepLLMTitleGenParticipant without any prompt.")
            self.prompt = ""
        self.prompt_question = prompt_question
        self.question_prefix = question_prefix

        self.max_para_num_words = max_para_num_words
        if gen_model == "gpt3":
            self.generator = GPT3Generator(**kwargs)
        elif gen_model == "llm_api":
            self.generator = LLMGenerator(**kwargs)
        else:
            raise ValueError("Unknown gen_model: " + gen_model)

    def return_model_calls(self):
        return {"step_by_step_retrieve": self.num_calls}

    def run(self, state, debug=False):

        paras_text = ""
        if self.show_so_far_paras:
            zipped_titles_paras = list(zip(state.data["titles"], state.data["paras"]))
            paragraphs = [para_to_text(title, para, self.max_para_num_words) for title, para in zipped_titles_paras]
            paras_text = "\n\n".join(paragraphs).strip()
            if not paragraphs:
                print(
                    "WARNING: Found a case of non-contexual question on contextual prompt. Prompt isn't 'trained' for it."
                )

        titles_text = ""
        if self.show_so_far_titles:
            so_far_titles = list(dict.fromkeys(state.data.get("titles", [])).keys())
            if so_far_titles:
                titles_text = "So far collected Wikipedia page titles: "
                titles_text += "[" + ", ".join(['"' + str(e) + '"' for e in so_far_titles]) + "]"
            else:
                print(
                    "WARNING: Found a case of non-contexual question on contextual prompt. Prompt isn't 'trained' for it."
                )
                titles_text = "So far no wikipedia page titles have been collected."

        cot_text = ""
        if self.show_so_far_cot:
            so_far_cot = " ".join(state.data.get("generated_sentences", [])).strip()
            if so_far_cot:
                cot_text = f"So far collected evidence: {so_far_cot}"
            else:
                print(
                    "WARNING: Found a case of non-contexual question on contextual prompt. Prompt isn't 'trained' for it."
                )
                cot_text = "So far no evidence has been collected."

        multihop_question = state.data["question"]
        question_text = f"Q: {self.question_prefix}The question is: '{multihop_question}'. "

        if self.prompt_question:
            question_text += self.prompt_question
        else:
            question_text += (
                f"Read the information given above to answer this question, and "
                f"generate titles of {self.retrieval_count} additional Wikipedia pages that have relevant information to answer this question."
            )

        test_example_str = "\n\n".join([paras_text, titles_text + "\n" + cot_text, question_text]).strip()
        test_example_str += "\n" + "A: "
        test_example_str = re.sub(r"\n\n+", "\n\n", test_example_str)

        prompt = "\n\n\n".join([self.prompt, test_example_str]).strip()

        output_text_scores = self.generator.generate_text_sequence(prompt)

        if len(output_text_scores) > 1:
            print("Can not handle more than one answer for this model yet" + "\n" + str(output_text_scores))

        generated_titles_str = output_text_scores[0][0].strip()

        new_state = state.copy()
        new_state.data.add_answer(QuestionAnsweringStep(answer=generated_titles_str, score=0, participant=state.next))
        new_state.next = self.next_model if self.next_model else self.end_state

        self.num_calls += 1

        return new_state

class QuestionCopyer(BasicModule):
    """
    Generates question by copying the question field from the data json.
    """

    def __init__(
        self,
        next_model=None,
        end_state="[EOQ]",
        eoq_after_n_calls=1,
    ):
        self.next_model = next_model
        self.end_state = end_state
        self.num_calls = 0
        self.eoq_after_n_calls = eoq_after_n_calls

    def return_model_calls(self):
        return {"copy_question": self.num_calls}

    def run(self, state, debug=False):

        if (self.num_calls + 1) % (self.eoq_after_n_calls + 1) == 0:
            output = self.end_state
        else:
            output = state.data["question"].strip()

        self.num_calls += 1

        new_state = state.copy()

        new_state.data.add_qgen(QuestionGenerationStep(question=output, score=0, participant=state.next))

        if output == self.end_state:
            new_state.next = self.end_state
        else:
            new_state.data.add_task(Task(task_question=None, task_participant=new_state.next))
            new_state.next = self.next_model

        return [new_state]

class ReasoningGenerator(BasicModule):
    """
    Keeps a state of generated COT, and continues it with one sentence at a time.
    The context fed to the COT generator can be changed by changing state.data["titles"]
    """
    # 这里的one sentence不应该都是推理步骤吗？为什么retriever中形成query时能出现筛选掉推理步骤，保留其他语句的做法？
    def __init__(
        self,
        prompt_file="",
        prompt_reader_args=None,
        add_context=True,
        answer_extractor_regex=".* answer is (.*)",
        answer_extractor_remove_last_fullstop=True,
        terminal_return_type="titles",
        generation_type="sentences",
        reason_base_former_retrieval=True,
        max_num_sentences=10,
        terminal_state_next_model=None,
        shuffle_paras=False,
        disable_exit=False,
        max_para_num_words=350,
        question_prefix="",
        gen_model="gpt3",
        next_model=None,
        end_state="[EOQ]",
        **kwargs,
    ):
        # config中可以一并配置llm_generator的超参数，通过**kwargs传给generator()。
        # 包括n也可以直接传
        # config中配置的参数分为两部分，一类是module要用的参数，一类**kwargs是module调用的其他函数(generator)要用的参数
        import spacy  # Kept here because it's almost always not required, and it's slow.

        self.num_calls = 0
        self.next_model = next_model
        self.end_state = end_state

        if prompt_file:
            # ircot的prompt中包含的distractor(干扰项)就是相似文档。这一点比较严重，后面应该尽快验证
            # 所以prompt可能也需要重新设计，至少样例中不应该有干扰项（还是也可以有？应该是多种情况都试一下）
            # 目前不能确定采用哪种做法，考虑每种情况都试一下（有干扰文档/无干扰文档）
            # 一个可能的“前提条件”或“限制条件”是“上下文长度受限的情况”（把一次性检索到的文档分开也可以说是为了解决这个问题）
            # few-shot learning的做法应该还是比较有用的，这一点不修改，但是prompt的内容应该调整（是否需要distractor）
            prompt_reader_args = prompt_reader_args or {}
            prompt_reader_args["file_path"] = prompt_file
            self.prompt = read_prompt(**prompt_reader_args)
        else:
            print("WARNING: Using ReasoningGenerator without any prompt.")
            self.prompt = ""

        self.max_para_num_words = max_para_num_words
        if gen_model == "gpt3":
            self.generator = GPT3Generator(**kwargs)
        elif gen_model == "llm_api":
            self.generator = LLMGenerator(**kwargs)
        else:
            raise ValueError("Unknown gen_model: " + gen_model)

        if disable_exit:
            assert terminal_return_type is None, "When disable_exit is True, terminal_return_type must be None."
        else:
            if terminal_return_type not in ("answer", "titles", "pids"):
                raise Exception(
                    f"When disable_exit is False, terminal_return_type has to be one of answer or titles."
                    f"Found {terminal_return_type}."
                )

        assert generation_type in ("sentences", "queries") #queries具体是什么情况？

        self.add_context = add_context
        self.answer_extractor_regex = re.compile(answer_extractor_regex)
        self.answer_extractor_remove_last_fullstop = answer_extractor_remove_last_fullstop
        self.terminal_return_type = terminal_return_type
        self.generation_type = generation_type
        self.max_num_sentences = max_num_sentences
        self.terminal_state_next_model = terminal_state_next_model
        self.shuffle_paras = shuffle_paras
        self.disable_exit = disable_exit
        self.question_prefix = question_prefix
        self.reason_base_former_retrieval = reason_base_former_retrieval
        # Run 'python -m spacy download en_core_web_sm' if not downloaded already.
        self.spacy_object = spacy.load("en_core_web_sm")

    def return_model_calls(self):
        return {"step_by_step_cot": self.num_calls}

    def run(self, state, debug=False):

        exit_generation = False

        if f"generated_{self.generation_type}" not in state.data:
            state.data[f"generated_{self.generation_type}"] = []
        if f"temp_generated_{self.generation_type}" not in state.data:
            state.data[f"temp_generated_{self.generation_type}"] = []
        if len(state.data['temp_titles']) == 0:
            print("当前步没有新检索到的文档，基于已有的上下文文档进行生成")
            state.data['temp_titles'] = [""]
            state.data['temp_paras'] = [""]
        # 如果已经生成的内容长度超过最大长度，就不再进行生成
        
        if len(state.data[f"generated_{self.generation_type}"]) >= self.max_num_sentences:
            exit_generation = True
            
        new_state = state.copy()
        
        return_answer = "EMPTY"
        
        # Don't bother wasting expensive llm call if we're already going to exist afterwards.
        if not exit_generation:
            # 如果可以正常进行生成，先用检索到的相关文档组成prompt，再输出模型进行生成
            if self.reason_base_former_retrieval:
                former_titles = new_state.data["titles"]
                former_paras = new_state.data["paras"]
            else:
                former_titles = []
                former_paras = []
                
            new_state.data[f"temp_generated_{self.generation_type}"] = [] #初始化一个中间存储器
            question = new_state.data["question"] #question就是原始question
            if self.question_prefix:
                assert self.question_prefix.endswith("\n") or self.question_prefix.endswith(" ")
                question = self.question_prefix + question
                
            assert len(new_state.data['temp_titles']) == len(new_state.data['temp_titles']) and len(new_state.data['temp_titles']) != 0, "No retrieved docs from the current step"
            for i in range(len(new_state.data['temp_titles'])): 
                #将这一步检索到的k个文档分开处理，返回一组备选thought，再经过vote mechanism选出最优的thought
                # 这个循环执行完之后，应该会改变state.data['temp_generated_{}']存储的内容，但stata.data['generated_{}']存储的内容还没有变化
                titles, paras = add_and_reorder_if_pinned(
                    former_titles + [new_state.data['temp_titles'][i]],
                    former_paras + [new_state.data['temp_paras'][i]],
                    new_state.data["metadata"].get("pinned_title", None),
                    new_state.data["metadata"].get("pinned_para", None),
                    new_state.data["metadata"].get("pin_position", None),
                )
                # 但应该还有一种情况，虽然保留每一步检索的文档，但推理时只使用当前步检索到的文档
                # pinned_context是直接存储在metadata里的（metadata里也存储了gold文档、答案等信息，这里应该是唯一用到metadata的地方
                # 这里的意思好像就是，如果pinned context不在被检索到的文本里，就在这一步将其从metadata里直接加入进来
                # 是只有IIRC需要的操作
                zipped_titles_paras = list(zip(titles, paras))
                if self.shuffle_paras:
                    random.shuffle(zipped_titles_paras)
    
                context = "\n\n".join(
                    [para_to_text(title, para, self.max_para_num_words) for title, para in zipped_titles_paras]
                )
                generation_so_far = " ".join(state.data[f"generated_{self.generation_type}"])
    
                if self.add_context:
                    test_example_str = context + "\n\n" + f"Q: {question}" + "\n" + f"A: {generation_so_far}"
                else:
                    test_example_str = f"Q: {question}" + "\n" + f"A: {generation_so_far}"
    
                prompt = "\n\n\n".join([self.prompt, test_example_str]).strip()
    
                output_text_scores = self.generator.generate_text_sequence(prompt)
                if len(output_text_scores) > 1:
                    print("Can not handle more than one answer for this model yet" + "\n" + str(output_text_scores))
    
                new_generation = output_text_scores[0][0].strip()
                new_sents = list(self.spacy_object(new_generation).sents)
                # config设置中注意把generator原先的next_model和终止机制都转移给voter
                if new_sents:
                    new_generation = new_sents[0].text #就是在这里实现了每次只选择新生成的第一句话加入思维链
                    # 不是在这里实现的，还需要继续找
                    # 就是在这里实现的，new_sents就是生成内容组成的多个句子。voter的情况是因为只生成了一句话（The best choice is...），如果生成内容超过一句话，这里就能达到提取其中第一句的效果
                    #new_state.data[f"generated_{self.generation_type}"].append(new_generation)
                    new_state.data[f"temp_generated_{self.generation_type}"].append(new_generation)
                    if self.answer_extractor_regex.match(new_generation):
                        # 能从这一步新生成的句子中提取出答案，就终止生成
                        # 主要是多个备选状态的情况怎么办？假如多个备选状态都得到了答案？
                        return_answer = self.answer_extractor_regex.match(new_generation).group(1)
                        if self.answer_extractor_remove_last_fullstop and return_answer.endswith("."):
                            return_answer = return_answer[:-1]
                        exit_generation = True
    
                else:
                    # generator真的会出现完全不生成任何内容的情况吗？？？
                    if self.disable_exit:  # Add just empty sentence so exit controller can exit.
                        #new_state.data[f"generated_{self.generation_type}"].append("")
                        new_state.data[f"temp_generated_{self.generation_type}"].append("")
                    exit_generation = True

        if self.disable_exit: # 强制要求不能退出
            exit_generation = False
            
        if exit_generation:
            if self.terminal_state_next_model is not None:
                new_state.next = self.terminal_state_next_model
            else:
                new_state.next = self.end_state

        else:
            # It should output full COT so far, not just what's generated in this round.
            new_state.next = self.next_model
        
        # assert isinstance(output, str)
        
        new_state.data['temp_generator_participant'] = state.next
        
        self.num_calls += 1
        
        # 这里实际修改的只有state.data['temp_generated_{}']，data['generated_{}']需要vote选出thought后才会正式修改
        # 主要是需要考虑各种情况，参数设置也需要调整（有些参数这一步可能不需要）
        # vote环节也是调用llm，尽量和generator的接口保持一致（需要看实际输出的情况）
        return new_state

class Voter(BasicModule):
    """
    处理上一步（ReasoningGenerator）生成的多个备选thought（存储在state.data['temp_generated_{}']里），并根据结果正式更新state.data['paras']和state.data['generated_{}']的内容（voter执行完止之后，state的状态就和IRCoT一样了）
    """
    # voter相当于是generator的延续，只是又调用了一次模型，所以disable_exit也是true，不会在vote环节之后就退出（至少会经过exit_contoller）
    # 这里的one sentence不应该都是推理步骤吗？为什么retriever中形成query时能出现筛选掉推理步骤，保留其他语句的做法？
    # generator的next_model是voter
    # voter的next_model是exit_controller
    # voter和generator会分别实例化generator，并不是generator实例化好之后，voter也继续使用
    def __init__(
        self,
        prompt_file="",
        prompt_reader_args=None,
        vote_extractor_regex=".*best CHOICE is .*(\d+).*",
        answer_extractor_remove_last_fullstop=True,
        terminal_return_type="titles",
        generation_type="sentences",
        reset_queries_as_sentences=False,
        choose_correspond_doc=False,
        choose_base_former_generation=False,
        max_num_sentences=10,
        terminal_state_next_model=None,
        return_pids=False,
        return_paras=False,
        # shuffle_paras=False,
        disable_exit=False,
        # max_para_num_words=350,
        question_prefix="",
        gen_model="gpt3",
        next_model=None,
        
        backup_prompt_file="",
        backup_prompt_reader_args=None,
        backup_gen_model="gpt3",
        backup_reason_base_former_retrieval=True,
        backup_question_prefix="",
        backup_shuffle_paras=False,
        backup_max_para_num_words=350,
        backup_add_context=True,
        backup_generator_params=None,
        end_state="[EOQ]",
        **kwargs,
    ):
        # choose_correspond_doc是新加的参数，=True：只保留投票选出的thought对应的检索文档；=False：保留全部文档（用于以后的检索）
        # choose_correspond_doc如果完全设置成false，投票的意义就不大了
        # choose_base_former是新加的参数，=True：在投票时把之前生成的内容也作为上下文的一部分，选择对象是former_context+choice1,former_context+choice2,...
        # 无论cumulate_retrieved_result是true还是false，这里都可以保存（因为vote环节处理的仍然属于“当前步”的内容）
        import spacy

        self.num_calls = 0
        self.next_model = next_model
        self.end_state = end_state

        if prompt_file:
            # 这里需要重新配置路径和读取prompt的做法，应该不用像generator中一样读取，prompt_read_args取默认值none即可
            prompt_reader_args = prompt_reader_args or {}
            prompt_reader_args["file_path"] = prompt_file
            self.prompt = read_prompt_easy(**prompt_reader_args)
        else:
            print("WARNING: Using Voter without any prompt.")
            self.prompt = ""
            
        #self.max_para_num_words = max_para_num_words #这个好像不需要，是zero-shot prompt
        if gen_model == "gpt3":
            self.generator = GPT3Generator(**kwargs)
        elif gen_model == "llm_api":
            self.generator = LLMGenerator(**kwargs)
        else:
            raise ValueError("Unknown gen_model: " + gen_model)
    
            
        if disable_exit: #这个参数保持不变
            assert terminal_return_type is None, "When disable_exit is True, terminal_return_type must be None."
        else:
            if terminal_return_type not in ("answer", "titles", "pids"):
                raise Exception(
                    f"When disable_exit is False, terminal_return_type has to be one of answer or titles."
                    f"Found {terminal_return_type}."
                )
        assert generation_type in ("sentences", "queries") # voter中的generation_type主要还是用来确保上一步generator的正常执行
        # 这里先默认选generation_type=sentences？
        self.vote_extractor_regex = re.compile(vote_extractor_regex,re.DOTALL)
        self.answer_extractor_remove_last_fullstop = answer_extractor_remove_last_fullstop
        self.terminal_return_type = terminal_return_type
        self.generation_type = generation_type
        self.reset_queries_as_sentences = reset_queries_as_sentences
        self.max_num_sentences = max_num_sentences
        self.terminal_state_next_model = terminal_state_next_model
        self.disable_exit = disable_exit
        self.return_pids = return_pids
        self.return_paras = return_paras
        self.question_prefix = question_prefix
        self.choose_correspond_doc = choose_correspond_doc
        self.choose_base_former_generation = choose_base_former_generation
        self.spacy_object = spacy.load("en_core_web_sm")
        
        self.backup_prompt_file = backup_prompt_file
        self.backup_prompt_reader_args = backup_prompt_reader_args
        self.backup_gen_model = backup_gen_model
        self.backup_reason_base_former_retrieval = backup_reason_base_former_retrieval
        self.backup_question_prefix = backup_question_prefix
        self.backup_shuffle_paras = backup_shuffle_paras
        self.backup_max_para_num_words = backup_max_para_num_words
        self.backup_add_context = backup_add_context
        self.backup_generator_params = backup_generator_params
        
    def return_model_calls(self):
        return {"step_by_step_vote": self.num_calls}

    def run(self, state, debug=False):

        # exit_generation = False
        
        choose_correspond_doc = self.choose_correspond_doc
        
        new_state = state.copy()
        
        if "vote_result" not in new_state.data:
            new_state.data["vote_result"] = [] #data['vote_result']并不会被使用，只是作为一种记录
        new_state.data["vote_result"].append([])
        
        return_answer = "EMPTY"

        #if not exit_generation:
        # 这里先就考虑一次生成的情况，后面如果能让LLM一次性返回多个结果（不用通过for循环多次调用），就进一步实现多次投票的做法，稳定性应该能更好
        assert self.generation_type == "sentences"
        
        # assert len(state.data["temp_generated_sentences"]) > 0, "Must have generated several thoughts for voter to choose inside."
        """
        可能的输入情况：
            1.1 temp_paras为空，temp_generated只有一个元素（没有检索到新文档（可能是检索到的文档之前都出现过），根据之前的文档进行生成）
                // 主要是怎么把之前加进去的空元素""筛选出来？或者有什么更好的处理方法？？？
            1.2 temp_paras不为空（检索到了新文档）
                1.2.1 temp_paras只有一个元素（只有一篇新文档），temp_generated只有一个元素（考虑这种情况主要是因为thought也只有一个）
                1.2.2 temp_paras有多个元素，temp_generated有多个元素
        
        可能的投票情况：
            2.1 正常结果，在temp_generated_{}下标范围内
                正常选出了一个thought，可以正常执行投票流程
            2.2 无效结果，模型回答I don't know
                无法选出thought，认为这种情况说明模型无法根据一篇相关文档生成下一个推理步骤（比如可能需要融合多个相关文档），因此重新做一次生成（用这一步检索到的全部文档，不再进行筛选）
            2.3 错误结果，在temp_generated_{}下标范围外
                完全错误的结果，暂时没想到处理方法，目前的异常处理方法是将投票结果改成默认的thought1
        1.2.2：2.1、2.2、2.3都可以正常处理
        1.2.1：不需要进行投票（因为只有一个选项，强行投票有可能导致越界）
        1.1：不需要进行投票
        投票结束后，正常情况需要把投票选出的thought和对应的相关文档加入到state.data中(temp_para[vote_index], temp_generated[vote_index])
            如果不需要投票的话（因为只有一个选项），thought直接加入即可，主要是对应的相关文档怎么加
            // 应该也是直接加入就行
        是否越界是只有执行投票才需要考虑的问题
        首先保证一点，无论是什么情况，temp_paras和temp_generated都至少有一个元素
        voter模块执行完之后，如果投票选出的thought还没得到答案，会继续进入下一轮的retriever-generator-voter的流程
        # 其实这种默认vote的保底做法应该也可以，因为下标为0的thought对应的实际上是相关度最高的文档，这个文档就是相关文档的几率应该是比较高的
        """
        #assert len(state.data['temp_paras']) == len(state.data[f"temp_generated_{self.generation_type}"]), 
        if len(new_state.data['temp_paras']) != len(new_state.data[f"temp_generated_{self.generation_type}"]):
            print("the num of context and generated thought must equal")
            print(new_state.data['temp_paras'],len(new_state.data['temp_paras']))
            print(new_state.data[f"temp_generated_{self.generation_type}"], len(new_state.data[f"temp_generated_{self.generation_type}"]))
        # 用这部分异常处理避免出现空集的情况，但按理说正常运行不可能出现空集
        # （可能会出现空结果（temp_paras = [""](当前步没检索到新文档)或temp_generated_{} = [""](当前步没有生成新的内容)(这种情况还不知道为什么会出现，但之前好像确实出现过)，但应该不会有空集）
        if len(new_state.data['temp_paras']) == 0:
            print("当前步检索到的文档数为0")
            new_state.data['temp_paras'] = [""]
            new_state.data['temp_titles'] = [""]
        if len(new_state.data[f"temp_generated_{self.generation_type}"]) == 0:
            print("当前步生成的thought数量为0")
            new_state.data[f"temp_generated_{self.generation_type}"] = [""]
        
        backup_module = False
        
        if len(new_state.data[f"temp_generated_{self.generation_type}"]) > 1:
            # 只有当存在两个以上的选项时，才进行投票
            # 如果只有一个选项，投票是不是相当于自我纠正？感觉不太可能，即使选项确实是错的也没法重新生成
            llm_votes = []
            choices = ""
            question = new_state.data["question"]
            if self.question_prefix:
                # generator这里的prefix是"answer the question by thinking step by step"，这里不知道是不是也需要
                assert self.question_prefix.endswith("\n") or self.question_prefix.endswith(" ")
                question = self.question_prefix + question
            if self.choose_base_former_generation:
                generation_so_far = " ".join(new_state.data[f"generated_{self.generation_type}"])
            else:
                generation_so_far = ""
            for i, candidate_thought in enumerate(new_state.data[f"temp_generated_{self.generation_type}"],1):
                choices += "CHOICE {}: {}\n".format(i, generation_so_far+candidate_thought)
                # 这种做法相当于是对“单个推理步骤的效果难以判断”的缓解方案（在投票时加上之前的thought）
                # 这种情况下才进行投票，否则投票是没有意义的
            question_thoughts_str = f"QUESTON: {question}" + "\n\n" + f"{choices}" + "\n\n" + "YOUR OUTPUT: "
            # 感觉在这个prompt里，各个choice的顺序也会对结果有影响，是不是应该每个choice前面都加一次question？但prompt内部的表述？
            prompt = "\n\n\n".join([self.prompt, question_thoughts_str]).strip()
            
            output_text_scores = self.generator.generate_text_sequence(prompt)
            #print(output_text_scores) # 这里先看下llm输出的到底是什么内容 # [(' The best CHOICE is 1', 0)]
            
            vote_sents = [list(self.spacy_object(output_text_score[0].strip()).sents) for output_text_score in output_text_scores]
            #new_generation = output_text_scores[0][0].strip()
            # 这里看下实际输出的情况

            for vote_sent in vote_sents:
                new_generation = ""
                vote = 0
                if vote_sent:
                    new_generation = vote_sent[0].text # 应该就是"the best choice is ..."
                    if self.vote_extractor_regex.match(new_generation):
                        return_answer = self.vote_extractor_regex.match(new_generation).group(1)
                        if self.answer_extractor_remove_last_fullstop and return_answer.endswith("."):
                            return_answer = return_answer[:-1]
                        vote = int(return_answer) - 1 # 转换成thought对应的下标
                    else:
                        print("voter的生成内容出现问题，无法正确提取结果")
                        backup_module = True
                        #exit_generation = True
    
                else:
                    print("voter的生成过程出现问题，无法正确生成结果")
                    backup_module = True
                    #exit_generation = True
                    
                llm_votes.append(vote)
                new_state.data['vote_result'][-1].append(new_generation)
            
            voted_index = max(set(llm_votes),key=llm_votes.count)
            # 应该只有这种情况才会出现越界的问题
            
        else:
            # 只有一个选项，不进行投票，直接选择唯一的一个选项
            print("只有一个备选thought: {}, 直接作为投票结果，跳过投票".format(new_state.data[f"temp_generated_{self.generation_type}"][0]))
            voted_index = 0
            
        print("投票结果为:{}(数组下标)".format(voted_index))
        
        # 得到投票结果后再进行进一步的判定
        # 是不是可以在这里判定temp_generated和temp_paras是否为空？
        # temp_generated为空的情况前面会定义成[""]（有一个空元素），temp_generated有一个元素的情况vote_index会直接选择唯一的一个元素，所以可能产生vote_index越界的情况应该只有temp_generated有多个选项的正常情况
        # 如果出现这种情况，这里会保证vote_index在下标范围内，并重新生成一个被选出的thought
        if voted_index >= len(new_state.data[f"temp_generated_{self.generation_type}"]):
            print("voted_index out of range")
            voted_index = 0
            backup_module = True
        
        # 需要考虑有票数相同的thought等极端情况
        if backup_module:
            print("无法有效地投票出结果，重新进行生成") #并把生成结果作为投票结果
            choose_correspond_doc = False
            backup_generator_params = self.backup_generator_params
            new_temp_generation = self.backup_generator_run(new_state, backup_generator_params)
            new_state.data[f"temp_generated_{self.generation_type}"][voted_index] = new_temp_generation
        
        if len(new_state.data['temp_paras']) == 1 and not new_state.data['temp_paras'][0]:
            print("当前步检索到的文档为空，跳过加入到state.data中的过程")
        else:
            if choose_correspond_doc:
                # 按照之前的处理，即使没有检索到新的文档（temp_paras实际上是空的），为了保证generator模块仍然能正常运行，会指定temp_paras = [""]
                # 但这种情况下，如果voted_index不对的话，就会发生错误（out of range）
                # 所以需要考虑voted_index错误的情况（按理说不会出现这种情况
                # 如果某一步没有新检索到的文档，但仍然执行了generator模块，得到了答案（因为是根据之前检索到的文档生成的，只有一个thought），并执行了后面的投票环节（还是不太清楚如果只有一个元素，投票结果会是什么情况）
                # 如果这一步没有新检索到的文档，但仍然进行了生成和投票，这里就会出问题
                new_state.data["titles"].append(new_state.data['temp_titles'][voted_index])
                new_state.data["paras"].append(new_state.data['temp_paras'][voted_index])
            else:
                new_state.data["titles"] += new_state.data['temp_titles'] # 因为是列表
                new_state.data["paras"] += new_state.data['temp_paras']
        
        new_state.data[f"generated_{self.generation_type}"].append(new_state.data[f"temp_generated_{self.generation_type}"][voted_index])
        
        # 这里完成retriever环节没有执行的操作
        output = json.dumps(new_state.data["temp_titles"])
        new_state.data.add_temp_answers(TempAnsweringStep(answers=output, score=0, participant=state.next))
        
        # 这里完成generator和voter环节的操作（把temp_generated_{}和投票结果加入到reasoning chain中
        # temp_answer实际上在generator环节就能完成，统一在这里完成主要是保证reasoning chain的顺序不被打乱
        output = "; ".join(new_state.data[f"temp_generated_{self.generation_type}"])
        new_state.data.add_temp_answers(TempAnsweringStep(answers=output, score=0, participant=state.next))
        
        output = "; ".join(new_state.data['vote_result'][-1])
        new_state.data.add_vote(AnswerVotingStep(vote=output, score=0, participant=state.next))
        
        answer = json.dumps(new_state.data["titles"])
        if self.return_pids:
            pids = [
                get_pid_for_title_paragraph_text(title, paragraph_text)
                for title, paragraph_text in zip(new_state.data["titles"], new_state.data["paras"])
            ]
            answer = json.dumps(pids)
        if self.return_paras:
            answer = json.dumps(
                [{"title": title, "paragraph_text": para} for title, para in zip(new_state.data["titles"], new_state.data["paras"])]
            )
        new_state.data.add_answer(QuestionAnsweringStep(answer=answer, score=0, participant=new_state.data["temp_retriever_participant"])) #这里的State是原来的state，原来state的next command也就是new state的participant

        # 这里完成generator环节没有执行的操作
        output = " ".join(new_state.data[f"generated_{self.generation_type}"])
        if self.reset_queries_as_sentences:
            # deepcopy is necessary
            new_state.data["generated_queries"] = copy.deepcopy(new_state.data["generated_sentences"])
        new_state.data.add_answer(QuestionAnsweringStep(answer=output, score=0, participant=new_state.data["temp_generator_participant"]))
        
        new_state.next = self.next_model
        
        self.num_calls += 1
        
        return new_state
    
    def backup_generator_run(
        self, 
        state,
        backup_generator_params,      
    ):
        # 这里也算一种回溯机制（如果投票失败
        # 如果最后投票成功的情况（例如2wikimqa的第一个问题）比投票失败的情况（投票没能从幻觉thought中选出正确的thought）
        # 在这一点上，distrator确实是有用的，与完全没有distractor相比，模型更有可能输出（问题的答案无法从上下文得出）
        # （后面可以继续调整generator的prompt，比如要求think step by step或不要强行生成答案
        # 后面还是看下llm生成的置信度分数，按理说如果确实是相似文本的话，置信度分数不可能比根据相关文本生成的更高（因为“根据相似文本生成答案的过程可能比根据相关文本生成更困难”）
        # 也就是score机制（投票机制只能根据文本内容判断哪个选项更合适，如果是人为判断也确实可能选错）
        if self.backup_prompt_file:
            # 这里需要重新配置路径和读取prompt的做法，应该不用像generator中一样读取，prompt_read_args取默认值none即可
            backup_prompt_reader_args = self.backup_prompt_reader_args or {}
            backup_prompt_reader_args["file_path"] = self.backup_prompt_file
            backup_prompt = read_prompt(**backup_prompt_reader_args)
        else:
            print("WARNING: Using Backup generator without any prompt.")
            backup_prompt = ""
            
        if self.backup_gen_model == "gpt3":
            backup_generator = GPT3Generator(**backup_generator_params)
        elif self.backup_gen_model == "llm_api":
            backup_generator = LLMGenerator(**backup_generator_params)
        else:
            raise ValueError("Unknown backup_gen_model: " + self.backup_gen_model)
        
        if self.backup_reason_base_former_retrieval:
            # 这里和generator的设置保持一致即可
            former_titles = state.data["titles"]
            former_paras = state.data["paras"]
        else:
            former_titles = []
            former_paras = []
            
        # new_generation = ""
        question = state.data["question"]
        if self.backup_question_prefix:
            assert self.backup_question_prefix.endswith("\n") or self.question_prefix.endswith(" ")
            question = self.backup_question_prefix + question
        
        titles, paras = add_and_reorder_if_pinned(
            former_titles + state.data['temp_titles'],
            former_paras + state.data['temp_paras'],
            state.data["metadata"].get("pinned_title", None),
            state.data["metadata"].get("pinned_para", None),
            state.data["metadata"].get("pin_position", None),
        )
        zipped_titles_paras = list(zip(titles, paras))
        if self.backup_shuffle_paras:
            random.shuffle(zipped_titles_paras)

        context = "\n\n".join(
            [para_to_text(title, para, self.backup_max_para_num_words) for title, para in zipped_titles_paras]
        )
        
        generation_so_far = " ".join(state.data[f"generated_{self.generation_type}"])

        if self.backup_add_context:
            test_example_str = context + "\n\n" + f"Q: {question}" + "\n" + f"A: {generation_so_far}"
        else:
            test_example_str = f"Q: {question}" + "\n" + f"A: {generation_so_far}"

        prompt = "\n\n\n".join([backup_prompt, test_example_str]).strip()

        output_text_scores = backup_generator.generate_text_sequence(prompt)
        
        if len(output_text_scores) > 1:
            print("Can not handle more than one answer for this model yet" + "\n" + str(output_text_scores))

        new_generation = output_text_scores[0][0].strip()
        
        new_sents = list(self.spacy_object(new_generation).sents)
        # config设置中注意把generator原先的next_model和终止机制都转移给voter
        if new_sents:
            new_generation = new_sents[0].text
        else:
            print("backup_generator未生成新内容，已生成的内容是:{}".format(generation_so_far))
            print(output_text_scores) #看下这种具体是什么情况
            # 主要是为什么会出现完全不生成内容的情况？？？
            new_generation = ""
        return new_generation
        
class ExitController(BasicModule):
    """
    controls whether to exit or not.
    """

    def __init__(
        self,
        answer_extractor_regex=".* answer is (.*)",
        answer_extractor_remove_last_fullstop=True,
        terminal_return_type="titles",
        max_num_sentences=10,
        terminal_state_next_model=None,
        global_max_num_paras=100,
        generation_key="generated_sentences",
        next_model=None,
        end_state="[EOQ]",
    ):
        if terminal_return_type not in ("answer", "titles", "pids"):
            raise Exception(f"terminal_return_type has to be one of answer or titles. Found {terminal_return_type}.")

        self.num_calls = 0
        self.answer_extractor_regex = re.compile(answer_extractor_regex)
        self.answer_extractor_remove_last_fullstop = answer_extractor_remove_last_fullstop
        self.terminal_return_type = terminal_return_type
        self.max_num_sentences = max_num_sentences
        self.terminal_state_next_model = terminal_state_next_model
        self.global_max_num_paras = global_max_num_paras
        self.generation_key = generation_key
        self.next_model = next_model
        self.end_state = end_state

    def return_model_calls(self):
        return {"step_by_step_exit_controller": self.num_calls}

    def run(self, state, debug=False):

        if self.generation_key not in state.data: #就是generation_sentences这种generator生成的类型
            state.data[self.generation_key] = []
        generated_sentences = state.data[self.generation_key]
        
        new_state = state.copy()
        return_answer = "EMPTY"
        return_titles = json.dumps(state.data["titles"])
        return_pids = json.dumps(
            [  # keep using these as we don't want pinned to be part of returned titiles
                get_pid_for_title_paragraph_text(title, paragraph_text)
                for title, paragraph_text in zip(state.data["titles"], state.data["paras"])
            ]
        )

        assert_unique_titles_paras(state.data["titles"], state.data["paras"]) #保证state.data['paras']中的每个文档都是唯一的
        if len(state.data["paras"]) > self.global_max_num_paras:
            print("发生异常，state.data[paras]数量超出最大数量")
            state.data["paras"] = state.data["paras"][:self.global_max_num_paras]
        
        assert len(state.data["paras"]) <= self.global_max_num_paras

        # 为什么会出现这种情况？？？
        # 出现这种情况的问题回答的方向完全错了，所以一直无法结束，检索的文档一直累积
        # 应该也算一种极端情况，暂时不考虑其他专门的处理方法
        exit_generation = False

        if state.data[self.generation_key] and not state.data[self.generation_key][-1]:
            exit_generation = True

        if len(state.data[self.generation_key]) >= self.max_num_sentences:
            exit_generation = True

        # backup if regex doesn't match but we need to exit.
        # 特别注意看下这里的backup是怎么做的
        if self.generation_key in ("generated_sub_answers", "generated_sub_questions"):
            return_answer = generated_sentences[-1]
        else:
            return_answer = " ".join(generated_sentences)

        if generated_sentences and self.answer_extractor_regex.match(generated_sentences[-1]):
            return_answer = self.answer_extractor_regex.match(generated_sentences[-1]).group(1)
            if self.answer_extractor_remove_last_fullstop and return_answer.endswith("."):
                return_answer = return_answer[:-1]
            exit_generation = True

        if exit_generation:

            if self.terminal_return_type == "answer":  # answer
                output = return_answer
            elif self.terminal_return_type == "pids":  # pids
                output = return_pids
            else:  # titles
                assert self.terminal_return_type == "titles"
                output = return_titles

            if self.terminal_state_next_model is not None:
                new_state.next = self.terminal_state_next_model
            else:
                new_state.next = self.end_state

        else:
            output = "Exit? No."
            new_state.next = self.next_model
            # terminal_state_next_model（终止）是answer_extractor，next_model（不终止）是retriever（开始下一轮）
        
        assert isinstance(output, str)
        new_state.data.add_answer(QuestionAnsweringStep(answer=output, score=0, participant=state.next))

        self.num_calls += 1

        return new_state


class AnswerExtractor(BasicModule):
    def __init__(
        self,
        regex,
        next_model="[EOQ]",
        match_all_on_failure=False,
        query_source="last_question",
        remove_last_fullstop=False,
    ):
        self.regex = re.compile(regex)
        self.next_model = next_model
        self.num_calls = 0
        self.match_all_on_failure = match_all_on_failure
        self.query_source = query_source
        self.remove_last_fullstop = remove_last_fullstop
        assert query_source in (
            "last_question",
            "last_answer",
        ), f"query_source must be either last_question or last_answer. Found {query_source}."

    def return_model_calls(self):
        return {"extract": self.num_calls}

    def run(self, state, debug=False):
        # 提取答案文本("the answer is")
        self.num_calls += 1 #每调用一次，这个属性就+1（通过属性记录整个运行过程的参数）

        new_state = state.copy()

        if self.query_source == "last_answer":
            query = new_state.data.get_last_answer()
            # query = new_state.data.get_last_reason_answer()
            # 这里因为按各个模块执行的顺序，最后一步的answer就是新生成的推理步骤(retriever生成的结果也会作为answer类型保存，但在推理步骤之前)
        else:
            query = new_state.data.get_last_question()

        if query.startswith('"') and query.endswith('"'):
            query = query[1:-1]

        m = self.regex.match(query)
        if self.match_all_on_failure and not self.regex.match(query):
            m = re.compile(r"(.*)").match(query)

        if m:
            answer = m.group(1)

            if self.remove_last_fullstop and answer.endswith("."):
                answer = answer[:-1]
            
            if debug:
                print("ANS: " + answer)

            try:  # Hacky. Fix later. This is to handle '[\\"1,450 miles\\"]' to '["1,450 miles"]'
                json.loads(answer)
            except:
                try:
                    answer = json.dumps(json.loads(answer.encode("utf-8").decode("unicode_escape")))
                except:
                    pass

            new_state.data.add_answer(QuestionAnsweringStep(answer=answer, score=0, participant=state.next))
            # 基本上这里生成的就是最终答案了
            new_state.last_output = answer
            new_state.next = self.next_model
            return new_state
        else:
            print("Answer Extractor did not find a match for input regex in {}".format(query))
            return []