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
    "estimated_generation_length": 300,
    "shuffle": false,
    "model_length_limit": 4000,
};

# (Potentially) Hyper-parameters:
# null means it's unused.
local llm_retrieval_count = null;
local llm_map_count = null;
local bm25_retrieval_count = 6; # 这里应该可以考虑适当增大检索的数量 # 这里改一下，调试阶段检索数量尽量小，避免api调用花费太大
local rc_context_type_ = "gold_with_n_distractors"; # Choices: no, gold, gold_with_n_distractors #这里和生成Prompt的参数配置不一样
local distractor_count = "2"; # Choices: 0, 1, 2, 3
local rc_context_type = (
    if rc_context_type_ == "gold_with_n_distractors"
    then "gold_with_" + distractor_count + "_distractors"  else rc_context_type_
);
local multi_step_show_titles = null;
local multi_step_show_paras = null;
local multi_step_show_cot = null;
local rc_qa_type = "cot"; # Choices: direct, cot
# 注意核对超参数(尤其是voter的参数)

{
    "start_state": "step_by_step_bm25_retriever",
    "end_state": "[EOQ]",
    "modules": {
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
            "cumulate_former_results": true,
            "end_state": "[EOQ]",
        },
        "step_by_step_cot_reasoning_gen": {
            "name": "step_by_step_cot_gen",
            "next_model": "step_by_step_vote",
            "prompt_file": "prompts/"+dataset+"/"+rc_context_type+"_context_cot_qa_gpt_3.txt",
            "prompt_reader_args": prompt_reader_args,
            "gen_model": "gpt3",
            "engine": "gpt-3.5-turbo-instruct",
            "retry_after_n_seconds": 50,
            "generation_type": "sentences",
            "add_context": true,
            "shuffle_paras": false,
            "terminal_return_type": null,
            "disable_exit": true,
            "reason_base_former_retrieval": true,
            "end_state": "[EOQ]",
        },
        "step_by_step_vote": {
            "name": "step_by_step_vote",
            "next_model": "step_by_step_exit_controller",
            "prompt_file": "prompts/" + "vote/" + "llm_vote_2.txt",
            "gen_model": "gpt3",
            "engine": "gpt-3.5-turbo-instruct",
            "llm_n": 6,
            "temperature": 0.7,
            "retry_after_n_seconds": 50,
            "generation_type": "sentences",
            "terminal_return_type": null,
            "disable_exit": true,
            "reset_queries_as_sentences": false,
            "return_pids": false,
            "choose_correspond_doc": true,
            "choose_base_former_generation": true,
            "end_state": "[EOQ]",
            "backup_prompt_file": "prompts/"+dataset+"/"+rc_context_type+"_context_cot_qa_gpt_3.txt",
            "backup_prompt_reader_args": prompt_reader_args,
            "backup_gen_model": "gpt3",
            "backup_reason_base_former_retrieval": true,
            "backup_shuffle_paras": false,
            "backup_add_context": true,
            "backup_generator_params": {
                "engine": "gpt-3.5-turbo-instruct",
                "retry_after_n_seconds": 50,
            },
        },
        "step_by_step_exit_controller": {
            "name": "step_by_step_exit_controller",
            "next_model": "step_by_step_bm25_retriever",
            "answer_extractor_regex": ".* answer is:? (.*)\\.?",
            "answer_extractor_remove_last_fullstop": true,
            "terminal_state_next_model": "generate_main_question",
            "terminal_return_type": "pids",
            "global_max_num_paras": 15,
            "end_state": "[EOQ]",
        },
        "generate_main_question": {
            "name": "question_copyer",
            "next_model": "answer_main_question",
            "eoq_after_n_calls": 1,
            "end_state": "[EOQ]",
        },
        "answer_main_question": {
            "name": "llmqa",
            "next_model": "extract_answer",
            "prompt_file": "prompts/"+dataset+"/"+rc_context_type+"_context_cot_qa_gpt_3.txt",
            "prompt_reader_args": prompt_reader_args,
            "end_state": "[EOQ]",
            "gen_model": "gpt3",
            "engine": "gpt-3.5-turbo-instruct",
            "retry_after_n_seconds": 50,
            "add_context": true,
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