# Set dataset:
local dataset = "iirc";
local retrieval_corpus_name = dataset;
local add_pinned_paras = if dataset == "iirc" then true else false;
local valid_qids = ["q_9499", "q_10236", "q_2466", "q_10270", "q_8776", "q_9591", "q_10227", "q_8981", "q_9518", "q_3290", "q_8173", "q_8736", "q_10344", "q_389", "q_1672"];
local prompt_reader_args = {
    "filter_by_key_values": {
        "qid": valid_qids
    },
    "order_by_key": "qid",
    "estimated_generation_length": 0, # don't drop in reading phase.
    "shuffle": false,
    "model_length_limit": 1000000, # don't drop in reading phase.
    "tokenizer_model_name": "google/flan-t5-base",
};

# (Potentially) Hyper-parameters:
# null means it's unused.
local llm_retrieval_count = 3;
local llm_map_count = 1;
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
    "start_state": "base_llm_title_gen",
    "end_state": "[EOQ]",
    "models": {
        "base_llm_title_gen": {
            "name": "step_by_step_llm_title_gen",
            "next_model": "base_llm_title_map",
            "prompt_question": "Generate titles of " + llm_retrieval_count + " Wikipedia pages that have relevant information to answer this question.",
            "prompt_file": "prompts/"+dataset+"/no_context_open_llm_retrieval_flan_t5.txt",
            "question_prefix": "Answer the following question.\n",
            "prompt_reader_args": prompt_reader_args,
            "retrieval_count": llm_retrieval_count,
            "show_so_far_titles": false,
            "show_so_far_paras": add_pinned_paras,
            "show_so_far_cot": false,
            "end_state": "[EOQ]",
            "gen_model": "llm_api",
            "model_name": "google/flan-t5-base",
            "model_tokens_limit": 6000,
            "max_length": 200,
        },
        "base_llm_title_map": {
            "name": "retrieve_and_reset_paragraphs",
            "next_model": "step_by_step_bm25_retriever",
            "retrieval_type": "map_generated_to_valid_titles",
            "retrieval_count": llm_map_count,
            "global_max_num_paras": 15,
            "source_corpus_name": retrieval_corpus_name,
            "dont_add_to_state": true,
            "cumulate_titles": false,
            "set_result_as_valid_titles": true,
            "end_state": "[EOQ]",
        },
        "step_by_step_bm25_retriever": {
            "name": "retrieve_and_reset_paragraphs",
            "next_model": "step_by_step_cot_reasoning_gen",
            "retrieval_type": "bm25",
            "retrieval_count": bm25_retrieval_count,
            "global_max_num_paras": 15,
            "query_source": "question_or_last_generated_sentence",
            "source_corpus_name": retrieval_corpus_name,
            "document_type": "title_paragraph_text",
            "valid_titles_are_allowed_titles": add_pinned_paras, # New (only present for IIRC)
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
            "model_name": "google/flan-t5-base",
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
        "pin_position": "bottom",
    },
    "prediction_type": "answer",
}