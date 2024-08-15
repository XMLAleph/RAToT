from typing import Dict

from modules.Dataset_Readers import DatasetReader, MultiParaRCReader
from modules.participant_qa import LLMQAParticipantModel
from modules.Scircot import (
    AnswerExtractor,
    RetrieveParticipant,
    StepByStepCOTGenParticipant,
    StepByStepExitControllerParticipant,
    CopyQuestionParticipant,
    StepByStepLLMTitleGenParticipant,
)
#开始在这部分修改模型

MODEL_NAME_CLASS = {
    "answer_extractor": AnswerExtractor,
    "llmqa": LLMQAParticipantModel,
    "retrieve_and_reset_paragraphs": RetrieveParticipant,
    "step_by_step_cot_gen": StepByStepCOTGenParticipant,
    "step_by_step_exit_controller": StepByStepExitControllerParticipant,
    "copy_question": CopyQuestionParticipant,
    "step_by_step_llm_title_gen": StepByStepLLMTitleGenParticipant,
}

READER_NAME_CLASS: Dict[str, DatasetReader] = {
    "multi_para_rc": MultiParaRCReader,
}

PREDICTION_TYPES = {"answer", "titles", "pids"}