# Set dataset:
local dataset = "2wikimultihopqa";
local retrieval_corpus_name = dataset;
local add_pinned_paras = if dataset == "iirc" then true else false;
local valid_qids = ["028eaef60bdb11eba7f7acde48001122", "8727d1280bdc11eba7f7acde48001122", "79a863dc0bdc11eba7f7acde48001122", "4724c54e08e011ebbda1ac1f6bf848b6", "e5150a5a0bda11eba7f7acde48001122", "35bf3490096d11ebbdafac1f6bf848b6", "a5995da508ab11ebbd82ac1f6bf848b6", "228546780bdd11eba7f7acde48001122", "97954d9408b011ebbd84ac1f6bf848b6", "f44939100bda11eba7f7acde48001122", "1ceeab380baf11ebab90acde48001122", "f86b4a28091711ebbdaeac1f6bf848b6", "c6f63bfb089e11ebbd78ac1f6bf848b6", "af8c6722088b11ebbd6fac1f6bf848b6", "5897ec7a086c11ebbd61ac1f6bf848b6"];
local prompt_reader_args = {
    "filter_by_key_values": {
        "qid": valid_qids
    },
    "order_by_key": "qid",
    "estimated_generation_length": 0, # don't drop in reading phase.
    "shuffle": false,
    "model_length_limit": 1000000, # don't drop in reading phase.
    "tokenizer_model_name": "google/flan-t5-large",
};

# (Potentially) Hyper-parameters:
# null means it's unused.
local llm_retrieval_count = null;
local llm_map_count = null;
local bm25_retrieval_count = 6;
local rc_context_type_ = "gold_with_n_distractors"; # Choices: no, gold, gold_with_n_distractors
local distractor_count = "2"; # Choices: 1, 2, 3
local rc_context_type = (
    if rc_context_type_ == "gold_with_n_distractors"
    then "gold_with_" + distractor_count + "_distractors"  else rc_context_type_
);
local multi_step_show_titles = null;
local multi_step_show_paras = null;
local multi_step_show_cot = null;
local rc_qa_type = "direct"; # Choices: direct, cot
local qa_question_prefix = (
    if std.endsWith(rc_context_type, "cot")
    then "Answer the following question by reasoning step-by-step.\n"
    else "Answer the following question.\n"
);

{
    "start_state": "step_by_step_bm25_retriever",
    "end_state": "[EOQ]",
    "models": {
        "step_by_step_bm25_retriever": {
            "name": "retrieve_and_reset_paragraphs",
            "next_model": "step_by_step_cot_reasoning_gen",
            "retrieval_type": "bm25",
            "retrieval_count": bm25_retrieval_count,
            "global_max_num_paras": 15,
            "query_source": "question_or_last_generated_sentence",
            "source_corpus_name": retrieval_corpus_name,
            "document_type": "title_paragraph_text",
            "return_pids": false,
            "cumulate_titles": true,
            "end_state": "[EOQ]",
        },
        "step_by_step_cot_reasoning_gen": {
            "name": "step_by_step_cot_gen",
            "next_model": "step_by_step_exit_controller",
            "prompt_file": "prompts/"+dataset+"/"+rc_context_type+"_context_cot_qa_flan_t5.txt",
            "question_prefix": "Answer the following question by reasoning step-by-step.\n",
            "prompt_reader_args": prompt_reader_args,
            "generation_type": "sentences",
            "reset_queries_as_sentences": false,
            "add_context": true,
            "shuffle_paras": false,
            "terminal_return_type": null,
            "disable_exit": true,
            "end_state": "[EOQ]",
            "gen_model": "llm_api",
            "model_name": "google/flan-t5-large",
            "model_tokens_limit": 6000,
            "max_length": 200,
        },
        "step_by_step_exit_controller": {
            "name": "step_by_step_exit_controller",
            "next_model": "step_by_step_bm25_retriever",
            "answer_extractor_regex": ".* answer is:? (.*)\\.?",
            "answer_extractor_remove_last_fullstop": true,
            "terminal_state_next_model": "extract_answer",
            "terminal_return_type": "pids",
            "global_max_num_paras": 15,
            "end_state": "[EOQ]",
        },
        "extract_answer": {
            "name": "answer_extractor",
            "query_source": "last_answer",
            "regex": ".* answer is:? (.*)\\.?",
            "match_all_on_failure": true,
            "remove_last_fullstop": true,
        }
    },
    "reader": {
        "name": "multi_para_rc",
        "add_paras": false,
        "add_gold_paras": false,
        "add_pinned_paras": add_pinned_paras,
    },
    "prediction_type": "answer",
}