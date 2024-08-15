# Set dataset:
local dataset = "hotpotqa";
local retrieval_corpus_name = dataset;
local add_pinned_paras = if dataset == "iirc" then true else false;
local valid_qids = ["5abb14bd5542992ccd8e7f07", "5ac2ada5554299657fa2900d", "5a758ea55542992db9473680", "5ae0185b55429942ec259c1b", "5a8ed9f355429917b4a5bddd", "5abfb3435542990832d3a1c1", "5ab92dba554299131ca422a2", "5a835abe5542996488c2e426", "5a89c14f5542993b751ca98a", "5a90620755429933b8a20508", "5a7bbc50554299042af8f7d0", "5a8f44ab5542992414482a25", "5add363c5542990dbb2f7dc8", "5a7fc53555429969796c1b55", "5a790e7855429970f5fffe3d"];
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
    "models": {
        "step_by_step_bm25_retriever": {
            "name": "ragot_retrieve_and_reset_paragraphs",
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
            "name": "ragot_step_by_step_cot_gen",
            "next_model": "step_by_step_vote",
            "prompt_file": "llm_qa_systems/prompts/"+dataset+"/"+rc_context_type+"_context_cot_qa_gpt_3.txt",
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
            "prompt_file": "llm_qa_systems/prompts/" + "vote/" + "llm_vote_2.txt",
            "gen_model": "gpt3",
            "engine": "gpt-3.5-turbo-instruct",
            "n": 6,
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
            "backup_prompt_file": "llm_qa_systems/prompts/"+dataset+"/"+rc_context_type+"_context_cot_qa_gpt_3.txt",
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
            "prompt_file": "llm_qa_systems/prompts/"+dataset+"/"+rc_context_type+"_context_cot_qa_gpt_3.txt",
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