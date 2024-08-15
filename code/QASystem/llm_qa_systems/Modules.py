import re
import copy
import json
from typing import List, Dict
from functools import lru_cache
import random
from rapidfuzz import fuzz

from llm_qa_systems.prompt_reader import read_prompt, read_prompt_easy
from llm_qa_systems.data_instances import QuestionAnsweringStep, AnswerVotingStep, TempAnsweringStep, QuestionGenerationStep, Task
from llm_qa_systems.Dataset_readers import get_pid_for_title_paragraph_text
from llm_qa_systems.Generators.gpt3generator import GPT3Generator
from llm_qa_systems.Retriever.ela_retriever import UnifiedRetriever
from llm_qa_systems.Retriever.api_retriever import SearchEngineRetriever

random.seed(100)  # Don't change.

@lru_cache(maxsize=None)
def get_spacy_object():
    import spacy

    return spacy.load("en_core_web_sm")


def is_reasoning_sentence(sentence: str) -> bool:
    starters = ["thus ", "thus,", "so ", "so,", "that is,", "therefore", "hence"]
    for starter in starters:
        if sentence.lower().startswith(starter):
            return True

    regex = re.compile("(.*)(\d[\d,]*\.?\d+|\d+) ([+-]) (\d[\d,]*\.?\d+|\d+) = (\d[\d,]*\.?\d+|\d+)(.*)")
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

def get_real_pid_for_title_paragraph_text(
    source_corpus_name: str, retriever_host: str, retriever_port: str, title, paragraph_text
) -> str:
    query_text = " ".join(paragraph_text.split(" ")[:30])
    retriever = UnifiedRetriever(host="http://localhost/", port=9200)
    retrieval_method = "retrieve_from_elasticsearch"
    params = {
        "allowed_titles": [title],
        "query_text": query_text,
        "max_hits_count": 20,
        "corpus_name": source_corpus_name,
        "document_type": "paragraph_text",
    }
    result = getattr(retriever, retrieval_method)(**params)
    result = result.json()
    
    retrieval = result["retrieval"]

    if not retrieval:
        print("WARNING: Not para with the same title retrieved.")
        return ""

    def para_similarity_func(retrieval_):
        return (
            float(retrieval_["title"].lower() == title.lower())
            + get_token_similarity(retrieval_["paragraph_text"], paragraph_text) / 100
        )

    retrieval = sorted(retrieval, key=para_similarity_func, reverse=True)[0]

    retrieved_title = retrieval["title"]
    retrieved_para = retrieval.get("paragraph_text", "")  # backoff for natcq
    retrieved_id = retrieval["id"]  # has to be there.
    assert retrieved_id

    if retrieved_title != title:
        print("WARNING: Para with the same title couldn't be identified.")
        retrieved_id = ""
    if retrieved_para != paragraph_text:
        print("WARNING: Para with the same paragraph_text couldn't be identified.")
        retrieved_id = ""

    return retrieved_id


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


def assert_unique_titles_paras(titles: List[str], paras: List[str]) -> bool:
    titles_paras = [(title, para) for title, para in zip(titles, paras)]
    assert len(titles_paras) == len(set(titles_paras))


def get_token_similarity(str_1: str, str_2: str) -> float:
    return fuzz.token_sort_ratio(str_1.lower(), str_2.lower())


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

        
# ragot
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
        
# scircot: class ParticipantModel(object):
    
# ragot
class RAGoTRetriever(BasicModule):
    def __init__(
        self,
        retrieval_type,
        retriever_type="local",
        retrieval_method="retrieve_from_elasticsearch",
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
        
        if retriever_type == "local":
            self.retriever = UnifiedRetriever(host="http://localhost/", port=9200) #这里改一下，考虑web的情况
        elif retriever_type == "web":
            self.retriever = SearchEngineRetriever()
        
        self.retrieval_method = retrieval_method
        assert self.retrieval_method in ("retrieve_from_elasticsearch", "retrieve_from_bing", "retrieve_from_google")
        
    def return_model_calls(self):
        return {"paragraph_retrieve_and_reset": self.num_calls}

    def run(self, state, debug=False):
        if self.query_source == "original_question":
            input_query = state.data["question"]

        elif self.query_source == "last_answer":
            input_query = state.data.get_last_answer() # 有初始值吗？实际运行的时候是第三种情况

        elif self.query_source == "question_or_last_generated_sentence":
            question = state.data["question"]
            generated_sentences = state.data.get("generated_sentences", [])
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
            state.data["urls"] = []
            
        selected_titles = state.data["titles"]
        selected_paras = state.data["paras"]
        selected_urls = state.data["urls"]
        
        if self.retrieval_type == "bm25":
            
            temp_titles = []
            temp_paras = []
            temp_urls = []
            
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
                    '''
                    if result["corpus_name"] != self.source_corpus_name:
                        raise Exception(
                            f"The retrieved corpus name {result['corpus_name']} "
                            f"doesn't match {self.source_corpus_name}."
                        )
                    '''
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
                        # 总的检索文档数最大是global_max_num_paras，超过就不再增加新的文档
                        continue

                    if self.valid_titles_are_allowed_titles: # 这个应该是IIRC数据集用的？？？
                        assert result["title"].lower().replace(" ", "") in [
                            valid_title.lower().replace(" ", "") for valid_title in state.data["valid_titles"]
                        ]
                    temp_titles.append(result["title"])
                    temp_paras.append(result["paragraph_text"])
                    temp_urls.append(result.get("url", result["title"]))
        
            new_state = state.copy()
            if "temp_titles" not in state.data:
                new_state.data['temp_titles'] = []
                new_state.data['temp_paras'] = []
                new_state.data['temp_urls'] = []
            # 还有个问题，participant这里也不太好控制（因为在vote那里再做这一步的话，paticipant已经变化了
            new_state.next = self.next_model if self.next_model else self.end_state

            if not self.dont_add_to_state: # dont_add_to_state=False时，才会将这一次检索到的文档加入到state的data部分中
                new_state.data["temp_paras"] = temp_paras
                new_state.data["temp_titles"] = temp_titles
                new_state.data["temp_urls"] = temp_urls
                new_state.data['temp_retriever_participant'] = state.next
                
            output = json.dumps(new_state.data["temp_titles"],ensure_ascii=False) #这里改一下，不要在结果页面上直接显示成list的形式
            new_state.data.add_temp_answers(TempAnsweringStep(answers=output, score=0, participant=state.next))
            
        else:
            raise Exception(
                f"retrieval_type must be one of 'map_generated_to_valid_titles', 'bm25'. Found {self.retrieval_type}."
            )

        self.num_calls += 1
        
        return new_state
    
class IRCoTRetriever(BasicModule):
    def __init__(
        self,
        retrieval_type,
        retriever_type="local",
        retrieval_method="retrieve_from_elasticsearch",
        retrieval_count=None,
        query_source="last_answer",
        cumulate_titles=False,
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
            "bm25",
            "dense"
        ), f"retrieval_type {retrieval_type} not among the valid choices."

        assert query_source in (
            "original_question",
            "last_answer",
            "question_or_last_generated_sentence",
        ), f"query_source {query_source} not among the valid choices."

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
        self.end_state = end_state
        self.retrieval_count = retrieval_count
        self.document_type = document_type
        self.query_source = query_source
        self.cumulate_titles = cumulate_titles
        self.source_corpus_name = source_corpus_name
        self.dont_add_to_state = dont_add_to_state
        self.dont_skip_long_paras = dont_skip_long_paras
        self.return_pids = return_pids
        self.return_paras = return_paras
        self.num_calls = 0

        if self.return_pids and self.return_paras:
            raise Exception("Only one of return_pids or return_paras should be true.")

        if allowed_paragraph_types:
            assert isinstance(allowed_paragraph_types, list)
            self.allowed_paragraph_types = allowed_paragraph_types
        else:
            self.allowed_paragraph_types = [None]

        if retrieval_type == "bm25":
            if self.retrieval_count is None:
                raise Exception(f"retrieval_count is needed for the retrieval_type {retrieval_type}.")
            if self.source_corpus_name is None:
                raise Exception(f"source_corpus_name is needed for the retrieval_type {retrieval_type}.")

        self.retrieval_failures_so_far = 0
        self.retrieval_failures_max = 9
        
        if retriever_type == "local":
            self.retriever = UnifiedRetriever(host="http://localhost/", port=9200) #这里改一下，考虑web的情况
        elif retriever_type == "web":
            self.retriever = SearchEngineRetriever()
        
        self.retrieval_method = retrieval_method
        assert self.retrieval_method in ("retrieve_from_elasticsearch", "retrieve_from_bing", "retrieve_from_google")
        
    def return_model_calls(self):
        return {"paragraph_retrieve_and_reset": self.num_calls}

    def run(self, state, debug=False):

        if self.query_source == "original_question":
            input_query = state.data["question"]

        elif self.query_source == "last_answer":
            input_query = state.data.get_last_answer()

        elif self.query_source == "question_or_last_generated_sentence":
            # add question to query only if generated sentences are empty. O/w use last_generated_sentence.
            question = state.data["question"]
            generated_sentences = state.data.get("generated_sentences", [])
            generated_sentences = remove_reasoning_sentences(generated_sentences)
            last_generated_sentence_str = generated_sentences[-1].strip() if generated_sentences else ""
            input_query = last_generated_sentence_str if last_generated_sentence_str else question

        else:
            raise Exception(f"Unknown query_source: {self.query_source}.")

        if self.cumulate_titles:
            selected_titles = state.data["titles"]
            selected_paras = state.data["paras"]
            selected_urls = state.data["urls"]
        else:
            selected_titles = []
            selected_paras = []
            selected_urls = []
                
        if self.retrieval_type == "bm25":
            
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

                result = getattr(self.retriever, self.retrieval_method)(**params)
                
                for retrieval_item in result:
                    if len(retrieval_item["paragraph_text"].split(" ")) > 600 and not self.dont_skip_long_paras:
                        print("WARNING: Discarding a retrieved paragraph as it's excessively long.")
                        continue
                    
                    if is_para_closely_matching(
                        selected_titles,
                        selected_paras,
                        retrieval_item["title"],
                        retrieval_item["paragraph_text"],
                    ):
                        continue

                    if len(selected_paras) >= self.global_max_num_paras:
                        continue

                    if self.valid_titles_are_allowed_titles:
                        assert retrieval_item["title"].lower().replace(" ", "") in [
                            valid_title.lower().replace(" ", "") for valid_title in state.data["valid_titles"]
                        ]

                    selected_titles.append(retrieval_item["title"])
                    selected_paras.append(retrieval_item["paragraph_text"])
                    selected_urls.append(retrieval_item.get("url", retrieval_item["title"]))
                    
        elif self.retrieval_type == "dense":
            return NotImplementedError("retrive method : {self.retrieval_type} not implemented yet")
        
        else:
            raise Exception(
                f"retrieval_type must be one of 'map_generated_to_valid_titles', 'bm25'. Found {self.retrieval_type}."
            )

        self.num_calls += 1

        answer = json.dumps(selected_titles,ensure_ascii=False)

        if self.return_pids:
            pids = [
                get_pid_for_title_paragraph_text(title, paragraph_text)
                for title, paragraph_text in zip(selected_titles, selected_paras)
            ]
            answer = json.dumps(pids,ensure_ascii=False)

        if self.return_paras:
            answer = json.dumps(
                [{"title": title, "paragraph_text": para} for title, para in zip(selected_titles, selected_paras)],
                ensure_ascii=False
            )

        new_state = state.copy()
        new_state.data.add_answer(QuestionAnsweringStep(answer=answer, score=0, participant=state.next))
        new_state.next = self.next_model if self.next_model else self.end_state

        if not self.dont_add_to_state:
            new_state.data["paras"] = selected_paras
            new_state.data["titles"] = selected_titles
            new_state.data["urls"] = selected_urls

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
        #print(self.num_calls)
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

class RAGoTReasoningGenerator(BasicModule):
    """
    Keeps a state of generated COT, and continues it with one sentence at a time.
    The context fed to the COT generator can be changed by changing state.data["titles"]
    """
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
        import spacy  # Kept here because it's almost always not required, and it's slow.

        self.num_calls = 0
        self.next_model = next_model
        self.end_state = end_state

        if prompt_file:
            prompt_reader_args = prompt_reader_args or {}
            prompt_reader_args["file_path"] = prompt_file
            self.prompt = read_prompt(**prompt_reader_args)
        else:
            print("WARNING: Using ReasoningGenerator without any prompt.")
            self.prompt = ""

        self.max_para_num_words = max_para_num_words
        if gen_model == "gpt3":
            self.generator = GPT3Generator(**kwargs)
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
            state.data['temp_urls'] = [""]
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
                former_urls = new_state.data["urls"]
            else:
                former_titles = []
                former_paras = []
                former_urls = []
                
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
        
        output = "; ".join(new_state.data[f"temp_generated_{self.generation_type}"])
        new_state.data.add_temp_answers(TempAnsweringStep(answers=output, score=0, participant=state.next))
        
        self.num_calls += 1
        
        return new_state
    
class IRCoTReasoningGenerator(BasicModule):
    """
    Keeps a state of generated COT, and continues it with one sentence at a time.
    The context fed to the COT generator can be changed by changing state.data["titles"]
    """

    def __init__(
        self,
        prompt_file="",
        prompt_reader_args=None,
        add_context=True,
        answer_extractor_regex=".* answer is (.*)",
        answer_extractor_remove_last_fullstop=True,
        terminal_return_type="titles",
        generation_type="sentences",
        reset_queries_as_sentences=False,
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

        import spacy  # Kept here because it's almost always not required, and it's slow.

        self.num_calls = 0
        self.next_model = next_model
        self.end_state = end_state

        if prompt_file:
            prompt_reader_args = prompt_reader_args or {}
            prompt_reader_args["file_path"] = prompt_file
            self.prompt = read_prompt(**prompt_reader_args)
        else:
            print("WARNING: Using StepByStepCOTGenParticipant without any prompt.")
            self.prompt = ""

        self.max_para_num_words = max_para_num_words
        if gen_model == "gpt3":
            self.generator = GPT3Generator(**kwargs)
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

        assert generation_type in ("sentences", "queries")

        self.add_context = add_context
        self.answer_extractor_regex = re.compile(answer_extractor_regex)
        self.answer_extractor_remove_last_fullstop = answer_extractor_remove_last_fullstop
        self.terminal_return_type = terminal_return_type
        self.generation_type = generation_type
        self.reset_queries_as_sentences = reset_queries_as_sentences
        self.max_num_sentences = max_num_sentences
        self.terminal_state_next_model = terminal_state_next_model
        self.shuffle_paras = shuffle_paras
        self.disable_exit = disable_exit
        self.question_prefix = question_prefix

        # Run 'python -m spacy download en_core_web_sm' if not downloaded already.
        self.spacy_object = spacy.load("en_core_web_sm")

    def return_model_calls(self):
        return {"step_by_step_cot": self.num_calls}

    def run(self, state, debug=False):

        exit_generation = False

        if f"generated_{self.generation_type}" not in state.data:
            state.data[f"generated_{self.generation_type}"] = []

        if len(state.data[f"generated_{self.generation_type}"]) >= self.max_num_sentences:
            exit_generation = True

        new_state = state.copy()
        return_answer = "EMPTY"
        return_titles = json.dumps(state.data["titles"],ensure_ascii=False)
        return_pids = json.dumps(
            [  # use this (^|v) as we don't want pinned to be part of returned titles/paras.
                get_pid_for_title_paragraph_text(title, paragraph_text)
                for title, paragraph_text in zip(state.data["titles"], state.data["paras"])
            ],
            ensure_ascii=False
        )

        # Don't bother wasting expensive llm call if we're already going to exist afterwards.
        if not exit_generation:

            question = state.data["question"]
            titles, paras = add_and_reorder_if_pinned(
                state.data["titles"],
                state.data["paras"],
                state.data["metadata"].get("pinned_title", None),
                state.data["metadata"].get("pinned_para", None),
                state.data["metadata"].get("pin_position", None),
            )
            zipped_titles_paras = list(zip(titles, paras))
            if self.shuffle_paras:
                random.shuffle(zipped_titles_paras)

            context = "\n\n".join(
                [para_to_text(title, para, self.max_para_num_words) for title, para in zipped_titles_paras]
            )
            generation_so_far = " ".join(state.data[f"generated_{self.generation_type}"])

            if self.question_prefix:
                assert self.question_prefix.endswith("\n") or self.question_prefix.endswith(" ")
                question = self.question_prefix + question

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
            if new_sents:
                new_generation = new_sents[0].text
                new_state.data[f"generated_{self.generation_type}"].append(new_generation)

                if self.answer_extractor_regex.match(new_generation):
                    return_answer = self.answer_extractor_regex.match(new_generation).group(1)
                    if self.answer_extractor_remove_last_fullstop and return_answer.endswith("."):
                        return_answer = return_answer[:-1]
                    exit_generation = True

            else:
                if self.disable_exit:  # Add just empty sentence so exit controller can exit.
                    new_state.data[f"generated_{self.generation_type}"].append("")
                exit_generation = True

        if self.disable_exit:
            exit_generation = False

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

            # It should output full COT so far, not just what's generated in this round.
            output = " ".join(new_state.data[f"generated_{self.generation_type}"])
            new_state.next = self.next_model

        if self.reset_queries_as_sentences:
            # deepcopy is necessary
            new_state.data["generated_queries"] = copy.deepcopy(new_state.data["generated_sentences"])

        assert isinstance(output, str)
        new_state.data.add_answer(QuestionAnsweringStep(answer=output, score=0, participant=state.next))

        self.num_calls += 1

        return new_state
    
class Voter(BasicModule):
    """
    处理上一步（ReasoningGenerator）生成的多个备选thought（存储在state.data['temp_generated_{}']里），并根据结果正式更新state.data['paras']和state.data['generated_{}']的内容（voter执行完止之后，state的状态就和IRCoT一样了）
    """
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
        
        choose_correspond_doc = self.choose_correspond_doc
        
        new_state = state.copy()
        
        if "vote_result" not in new_state.data:
            new_state.data["vote_result"] = [] #data['vote_result']并不会被使用，只是作为一种记录
        new_state.data["vote_result"].append([])
        
        return_answer = "EMPTY"

        #if not exit_generation:
        # 这里先就考虑一次生成的情况，后面如果能让LLM一次性返回多个结果（不用通过for循环多次调用），就进一步实现多次投票的做法，稳定性应该能更好
        assert self.generation_type == "sentences"
    
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
            new_state.data['temp_urls'] = [""]
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
                new_state.data["titles"].append(new_state.data['temp_titles'][voted_index])
                new_state.data["paras"].append(new_state.data['temp_paras'][voted_index])
                new_state.data["urls"].append(new_state.data['temp_urls'][voted_index])
            else:
                new_state.data["titles"] += new_state.data['temp_titles'] # 因为是列表
                new_state.data["paras"] += new_state.data['temp_paras']
                new_state.data["urls"] += new_state.data['temp_urls']
                
        new_state.data[f"generated_{self.generation_type}"].append(new_state.data[f"temp_generated_{self.generation_type}"][voted_index])
        
        # 这里完成retriever环节没有执行的操作
        # 这些操作看下为什么要放在这里执行？按理说在retriever和reader模块就确定了？
        # 应该是为了处理无法正常投票出结果的情况
        # 看下放到前面可不可行，主要是是否影响实际效果
        '''
        output = json.dumps(new_state.data["temp_titles"])
        new_state.data.add_temp_answers(TempAnsweringStep(answers=output, score=0, participant=state.next))
        '''
        
        # 这里完成generator和voter环节的操作（把temp_generated_{}和投票结果加入到reasoning chain中
        # temp_answer实际上在generator环节就能完成，统一在这里完成主要是保证reasoning chain的顺序不被打乱 #？？？
        # 感觉应该不会被打乱？？？
        # 确实会打乱，除非投票结果这一步不保存
        # 好像也不会打乱，step的数据类型不一样，看下代码细节
        # 只是tempanswering类型应该不会打乱
        # 应该不会打乱，tempanswering类型的step不会被作为模块的输入，只是记录状态  顺序上应该也不会打乱
        # vote这一步仍然存在添加多个step的问题，但现在retriever和generator应该也有输出了，看下实际运行有没有问题
        '''
        output = "; ".join(new_state.data[f"temp_generated_{self.generation_type}"])
        new_state.data.add_temp_answers(TempAnsweringStep(answers=output, score=0, participant=state.next))
        '''
        
        output = "; ".join(new_state.data['vote_result'][-1])
        new_state.data.add_vote(AnswerVotingStep(vote=output, score=0, participant=state.next))
        
        answer = json.dumps(new_state.data["titles"],ensure_ascii=False)
        if self.return_pids:
            pids = [
                get_pid_for_title_paragraph_text(title, paragraph_text)
                for title, paragraph_text in zip(new_state.data["titles"], new_state.data["paras"])
            ]
            answer = json.dumps(pids,ensure_ascii=False)
        if self.return_paras:
            answer = json.dumps(
                [{"title": title, "paragraph_text": para} for title, para in zip(new_state.data["titles"], new_state.data["paras"])],
                ensure_ascii=False
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
        else:
            raise ValueError("Unknown backup_gen_model: " + self.backup_gen_model)
        
        if self.backup_reason_base_former_retrieval:
            # 这里和generator的设置保持一致即可
            former_titles = state.data["titles"]
            former_paras = state.data["paras"]
            former_urls = state.data["urls"]
        else:
            former_titles = []
            former_paras = []
            former_urls = []
            
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
        return_titles = json.dumps(state.data["titles"],ensure_ascii=False)
        return_pids = json.dumps(
            [  # keep using these as we don't want pinned to be part of returned titiles
                get_pid_for_title_paragraph_text(title, paragraph_text)
                for title, paragraph_text in zip(state.data["titles"], state.data["paras"])
            ],
            ensure_ascii=False
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
            output = "Exit? Yes."
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
                    answer = json.dumps(json.loads(answer.encode("utf-8").decode("unicode_escape")),ensure_ascii=False)
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