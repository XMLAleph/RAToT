from typing import Dict

from Dataset_readers import DatasetReader
from participant_qa import LLMQAGenerator
from Modules import (
    AnswerExtractor,
    QuestionCopyer,
    Retriever,
    ReasoningGenerator,
    TitleGenerator,
    ExitController,
    Voter,
)

MODULE_NAME_CLASS = {
    "answer_extractor": AnswerExtractor,
    "question_copyer": QuestionCopyer,
    "llmqa": LLMQAGenerator,
    "retrieve_and_reset_paragraphs": Retriever,
    "step_by_step_cot_gen": ReasoningGenerator,
    "step_by_step_llm_title_gen": TitleGenerator,
    "step_by_step_vote": Voter,
    "step_by_step_exit_controller": ExitController,
}

READER_NAME_CLASS: Dict[str, DatasetReader] = {
    "multi_para_rc": DatasetReader,
}

PREDICTION_TYPES = {"answer", "titles", "pids"}