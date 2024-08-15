import os
from typing import Dict
import time
from functools import lru_cache

from diskcache import Cache
from modules.prompt_reader import fit_prompt_into_given_limit

import torch
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from transformers.utils.import_utils import is_torch_bf16_gpu_available
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5Tokenizer,
    T5ForConditionalGeneration,
)

# cache = Cache(os.path.expanduser("~/.cache/llmcalls"))

# @lru_cache(maxsize=None)
def get_model_and_tokenizer():

    model_shortname = os.environ["MODEL_NAME"] #这里的做法改一下

    valid_model_shortnames = [
        "gpt-j-6B",
        "opt-66b",
        "gpt-neox-20b",
        "T0pp",
        "opt-125m",
        "flan-t5-base",
        "flan-t5-large",
        "flan-t5-xl",
        "flan-t5-xxl",
        "flan-t5-base-bf16",
        "flan-t5-large-bf16",
        "flan-t5-xl-bf16",
        "flan-t5-xxl-bf16",
        "gpt-3.5-turbo-instruct"
    ]
    assert model_shortname in valid_model_shortnames, f"Model name {model_shortname} not in {valid_model_shortnames}"

    if model_shortname == "gpt-j-6B":

        model_name = "EleutherAI/gpt-j-6B"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision="sharded",
            device_map="auto",  # torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    elif model_shortname == "gpt-3.5-turbo-instruct":
        return
        
    elif model_shortname == "opt-66b":

        model_name = "facebook/opt-66b"
        model = AutoModelForCausalLM.from_pretrained(model_name, revision="main", device_map="auto")
        # the fast tokenizer currently does not work correctly
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    elif model_shortname == "gpt-neox-20b":

        model_name = "EleutherAI/gpt-neox-20b"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision="main",
            device_map="auto",  # torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    elif model_shortname == "T0pp":

        model_name = "bigscience/T0pp"
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            revision="sharded",
            device_map="auto",  # torch_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    elif model_shortname == "opt-125m":

        model_name = "facebook/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(model_name, revision="main", device_map="auto")
        # the fast tokenizer currently does not work correctly
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    elif model_shortname.startswith("flan-t5") and "bf16" not in model_shortname:
        model_name = "google/" + model_shortname
        if torch.cuda.device_count() == 2:
            hf_device_map = {"shared": 1, "encoder": 0, "decoder": 1, "lm_head": 1}
        else:
            hf_device_map = "auto"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, revision="main", device_map=hf_device_map)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    elif model_shortname.startswith("flan-t5") and model_shortname.endswith("-bf16"):

        assert torch.cuda.is_bf16_supported()
        assert is_torch_bf16_gpu_available()
        model_name = "google/" + model_shortname.replace("-bf16", "")
        if torch.cuda.device_count() == 2:
            hf_device_map = {"shared": 1, "encoder": 0, "decoder": 1, "lm_head": 1}
        else:
            hf_device_map = "auto"
        model = T5ForConditionalGeneration.from_pretrained(
            model_name, device_map=hf_device_map, torch_dtype=torch.bfloat16
        )
        tokenizer = T5Tokenizer.from_pretrained(model_name)

    elif model_shortname == "ul2":
        model_name = "google/" + model_shortname
        model = T5ForConditionalGeneration.from_pretrained(
            # Don't use auto here. It's slower cpu loading I guess.
            "google/ul2"  # , low_cpu_mem_usage=True, torch_dtype=torch.bfloat16
        ).cuda()
        tokenizer = AutoTokenizer.from_pretrained("google/ul2")

    return model, tokenizer

class EOSReachedCriteria(StoppingCriteria):
    # Use this when EOS is not a single id, but a sequence of ids, e.g. for a custom EOS text.
    def __init__(self, tokenizer: AutoTokenizer, eos_text: str):
        self.tokenizer = tokenizer
        self.eos_text = eos_text
        assert (
            len(self.tokenizer.encode(eos_text)) < 10
        ), "EOS text can't be longer then 10 tokens. It makes stopping_criteria check slow."

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        decoded_text = self.tokenizer.decode(input_ids[0][-10:])
        condition1 = decoded_text.endswith(self.eos_text)
        condition2 = decoded_text.strip().endswith(self.eos_text.strip())
        return condition1 or condition2
    
def generate(
    prompt: str,
    max_input: int = None,
    max_length: int = 200,
    min_length: int = 1,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    num_return_sequences: int = 1,
    repetition_penalty: float = None,
    length_penalty: float = None,
    eos_text: str = None,
    keep_prompt: bool = False,
):
    start_time = time.time()

    model_shortname = os.environ["MODEL_NAME"]

    model, tokenizer = get_model_and_tokenizer()
    inputs = tokenizer.encode(prompt, max_length=max_input, return_tensors="pt").cuda()
    # 这里max_input看下具体是在哪设定的
    stopping_criteria_list = StoppingCriteriaList()
    if eos_text:
        stopping_criteria = EOSReachedCriteria(tokenizer=tokenizer, eos_text=eos_text)
        stopping_criteria_list = StoppingCriteriaList([stopping_criteria])

    # T0pp, ul2 and flan are the only encoder-decoder model, and so don't have prompt part in its generation.
    is_encoder_decoder = model_shortname in ["T0pp", "ul2"] or model_shortname.startswith("flan-t5")
    # max_length_ = max_length if is_encoder_decoder else inputs.shape[1]+max_length
    generated_output = model.generate(
        inputs,
        # max_length=max_length_,
        max_new_tokens=max_length,
        min_length=min_length,
        do_sample=True,
        temperature=0.8,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        return_dict_in_generate=True,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        stopping_criteria=stopping_criteria_list,
        output_scores=False,  # make it configurable later. It turns in generated_output["scores"]
    )
    
    # 直接在这里修改参数设置即可调整生成的多样性(参考https://huggingface.co/blog/zh/how-to-generate)，先打印看下原先的参数设置
    generated_ids = generated_output["sequences"]
    #generated_scores = generated_output["sequences_scores"]
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    generated_num_tokens = [len(generated_ids_) for generated_ids_ in generated_ids]
    if not keep_prompt and not is_encoder_decoder:
        generated_texts = [
            generated_text[generated_text.index(prompt) + len(prompt) :] for generated_text in generated_texts
        ]
    elif keep_prompt and is_encoder_decoder:
        generated_texts = [prompt + generated_text for generated_text in generated_texts]

    end_time = time.time()
    run_time_in_seconds = end_time - start_time
    return {
        "generated_num_tokens": generated_num_tokens,
        "generated_texts": generated_texts,
        "run_time_in_seconds": run_time_in_seconds,
        "model_name": model_shortname,
    }

def non_cached_llm_call(  # kwargs doesn't work with caching.
    prompt,
    model_name,
    max_input=None,
    max_length=100,
    min_length=1,
    do_sample=False,
    temperature=1.0,
    top_k=50,
    top_p=1.0,
    num_return_sequences=1,
    repetition_penalty=None,
    length_penalty=None,
    keep_prompt=False,
) -> Dict:

    params = {
        "prompt": prompt,
        "max_input": max_input,
        "max_length": max_length,
        "min_length": min_length,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "num_return_sequences": num_return_sequences,
        "repetition_penalty": repetition_penalty,
        "length_penalty": length_penalty,
        "keep_prompt": keep_prompt,
    }

    
    if "/" in model_name:
        assert model_name.count("/", 1)
        model_name = model_name.split("/")[1]

    result = generate(**params) #.json()

    model_name_ = result.get("model_name", "")  # To assure that response is from the right model.

    if model_name_.replace("-bf16", "").replace("-dsbf16", "").replace("-8bit", "") != model_name:
        raise Exception(f"Looks like incorrect LLM server is ON: {model_name_} != {model_name}.")

    return result


# @cache.memoize()
def cached_llm_call(  # kwargs doesn't work with caching.
    prompt,
    model_name,
    max_input=None,
    max_length=100,
    min_length=1,
    do_sample=False,
    temperature=1.0,
    top_k=50,
    top_p=1.0,
    num_return_sequences=1,
    repetition_penalty=None,
    length_penalty=None,
    keep_prompt=False,
) -> Dict:
    return non_cached_llm_call(
        prompt,
        model_name,
        max_input=max_input,
        max_length=max_length,
        min_length=min_length,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        keep_prompt=keep_prompt,
    )


def llm_call(
    prompt,
    model_name,
    max_input=None,
    max_length=100,
    min_length=1,
    do_sample=False,
    temperature=1.0,
    top_k=50,
    top_p=1.0,
    num_return_sequences=1,
    repetition_penalty=None,
    length_penalty=None,
    keep_prompt=False,
):
    function = cached_llm_call if not do_sample and temperature > 0 else non_cached_llm_call
    return function(
        prompt,
        model_name,
        max_input=max_input,
        max_length=max_length,
        min_length=min_length,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        keep_prompt=keep_prompt,
    )


class LLMGenerator:

    # Instructions to start the LLM Server are in the README here:
    # https://github.com/harshTrivedi/llm_server

    def __init__(
        self,
        model_name,
        max_input=None,
        max_length=100,
        min_length=1,
        do_sample=False,
        eos_text="\n",
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        num_return_sequences=1,
        repetition_penalty=None,
        length_penalty=None,
        model_tokens_limit=2000,
        remove_method="first",
    ):

        valid_model_names = [
            "gpt-j-6B",
            "opt-66b",
            "gpt-neox-20b",
            "T0pp",
            "flan-t5-base",
            "flan-t5-large",
            "flan-t5-xl",
            "flan-t5-xxl",
            "ul2",
        ] #这些后面也都可以测试一下
        model_name_ = model_name
        if "/" in model_name:
            assert model_name.count("/", 1)
            model_name_ = model_name.split("/")[1]
        assert model_name_ in valid_model_names, f"Model name {model_name_} not in {valid_model_names}"

        self.model_name = model_name
        self.max_input = max_input
        self.max_length = max_length
        self.min_length = min_length
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.eos_text = eos_text
        self.num_return_sequences = num_return_sequences
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.model_tokens_limit = model_tokens_limit
        self.remove_method = remove_method

    def generate_text_sequence(self, prompt):
        """
        :param input_text:
        :return: returns a sequence of tuples (string, score) where lower score is better
        """
        prompt = prompt.rstrip()

        prompt = fit_prompt_into_given_limit(
            original_prompt=prompt,
            model_length_limit=self.model_tokens_limit,
            estimated_generation_length=self.max_length,
            demonstration_delimiter="\n\n\n",
            shuffle=False,
            remove_method=self.remove_method,
            tokenizer_model_name=self.model_name,
            last_is_test_example=True,
        )

        # Note: Don't pass eos_text. Doesn't seem to work right.
        
        params = {
          "prompt": prompt,
          "model_name": self.model_name,
          "max_input": self.max_input,
          "max_length": self.max_length,
          "min_length": self.min_length,
          "do_sample": self.do_sample,
          "temperature": self.temperature,
          "top_k": self.top_k,
          "top_p": self.top_p,
          "num_return_sequences": self.num_return_sequences,
          "repetition_penalty": self.repetition_penalty,
          "length_penalty": self.length_penalty,
          "keep_prompt": False,
        }
        
        result = llm_call(**params)
        '''
        result = llm_call(
            prompt=prompt,
            model_name=self.model_name,
            max_input=self.max_input,
            max_length=self.max_length,
            min_length=self.min_length,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            num_return_sequences=self.num_return_sequences,
            repetition_penalty=self.repetition_penalty,
            length_penalty=self.length_penalty,
            keep_prompt=False,
        )
        '''
        generated_texts = result["generated_texts"]
        modified_texts = []
        for text in generated_texts:
            # remove the prompt
            if text.startswith(prompt):
                text = text[len(prompt) :]
            if self.eos_text and self.eos_text in text:
                text = text[: text.index(self.eos_text)]
            modified_texts.append(text)
        generated_texts = modified_texts

        output_seq_score = [(text, 1 / (index + 1)) for index, text in enumerate(generated_texts)]
        # print(prompt)
        # print("------------")
        # print(output_seq_score[0][0])

        # TODO: Deal with output-probabilities if needed.

        return sorted(output_seq_score, key=lambda x: x[1])
