from typing import Dict

from llm_qa_systems.Dataset_readers import DatasetReader
from llm_qa_systems.participant_qa import LLMQAGenerator
from llm_qa_systems.Modules import (
    AnswerExtractor,
    QuestionCopyer,
    RAGoTRetriever,
    IRCoTRetriever,
    RAGoTReasoningGenerator,
    IRCoTReasoningGenerator,
    TitleGenerator,
    ExitController,
    Voter,
)

MODEL_NAME_CLASS = {
    "answer_extractor": AnswerExtractor,
    "llmqa": LLMQAGenerator,
    "ircot_retrieve_and_reset_paragraphs": IRCoTRetriever,
    "ragot_retrieve_and_reset_paragraphs": RAGoTRetriever,
    "ircot_step_by_step_cot_gen": IRCoTReasoningGenerator,
    "ragot_step_by_step_cot_gen": RAGoTReasoningGenerator,
    "step_by_step_exit_controller": ExitController,
    "copy_question": QuestionCopyer,
    "question_copyer": QuestionCopyer,
    "step_by_step_llm_title_gen": TitleGenerator,
    "step_by_step_vote": Voter,
}

READER_NAME_CLASS: Dict[str, DatasetReader] = {
    "multi_para_rc": DatasetReader,
}

PREDICTION_TYPES = {"answer", "titles", "pids"}