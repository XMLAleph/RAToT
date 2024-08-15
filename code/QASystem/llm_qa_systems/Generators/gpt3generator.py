import logging
import time
import os
from functools import lru_cache

from openai import OpenAI
from diskcache import Cache
from llm_qa_systems.prompt_reader import fit_prompt_into_given_limit

logger = logging.getLogger(__name__)

def openai_call(
    prompt,
    engine,
    temperature,
    max_tokens,
    top_p,
    frequency_penalty,
    presence_penalty,
    stop,
    n,
    best_of,
    logprobs,
):
    engine = "gpt-3.5-turbo-instruct"
    # function = cached_openai_call if temperature == 0 else OpenAI().completions.create
    function = OpenAI().completions.create
    # 只有temperature设置为0的时候才通过上面的cache_call函数调用openai api，否则直接在这里定义
    # 但这两种做法有什么区别?
    return function(
        prompt=prompt,
        model=engine,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
        n=n,
        logprobs=logprobs,
    )


@lru_cache(maxsize=1)
def get_gpt_tokenizer():
    from transformers import GPT2Tokenizer

    return GPT2Tokenizer.from_pretrained("gpt2")


class GPT3Generator:
    def __init__(
        self,
        engine="text-davinci-002",
        temperature=0,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n"],
        retry_after_n_seconds=None,
        n=1,
        best_of=1,
        logprobs=0,
        remove_method="first",
    ):
        self.engine = "gpt-3.5-turbo-instruct"
        self.logprobs = logprobs
        self.n = n
        self.best_of = best_of
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.stop = stop
        self.temperature = temperature
        self.retry_after_n_seconds = None #这里先试下codex(应该已经不能调用了)，用其他模型的话不能retry
        # 先试下codex能不能正常调用
        self.remove_method = remove_method

        self.model_tokens_limit = 3500
        
    def generate_text_sequence(self, prompt):
        prompt = prompt.rstrip()

        prompt = fit_prompt_into_given_limit(
            original_prompt=prompt,
            model_length_limit=self.model_tokens_limit,
            estimated_generation_length=self.max_tokens,
            demonstration_delimiter="\n\n\n",
            shuffle=False,
            remove_method=self.remove_method,
            tokenizer_model_name="gpt2",  # did this before tiktoken was released.
            last_is_test_example=True,
        )

        arguments = {
            "engine": self.engine,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "n": self.n,
            "best_of": self.best_of,
            "logprobs": self.logprobs,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stop": self.stop,
        }
        if self.best_of is not None:
            arguments["best_of"] = self.best_of

        success = False
        for index in range(3): #原始的设置是500
            try:
                response = openai_call(**arguments)
                success = True
                break
            except Exception as exception:

                success = False

                tokenizer = get_gpt_tokenizer()
                prompt_num_tokens = len(tokenizer.tokenize(prompt))
                if prompt_num_tokens + arguments["max_tokens"] > self.model_tokens_limit > prompt_num_tokens:
                    last_used_max_tokens = arguments["max_tokens"]
                    updated_max_tokens = self.model_tokens_limit - prompt_num_tokens
                    arguments["max_tokens"] = updated_max_tokens
                    if last_used_max_tokens == updated_max_tokens:
                        break
                    print(
                        f"WARNING: (Round {index}) Decreasing max_tokens from "
                        f"{last_used_max_tokens} to {updated_max_tokens} and retrying."
                    )
                    continue

                if self.retry_after_n_seconds is None:
                    import traceback

                    print(traceback.format_exc())
                    exit()

                print(f"Encountered exception of class: {exception.__class__}")
                if hasattr(exception, "user_message"):
                    print(exception.user_message)
                print(f"Potentially reached OpenAI rate limit. Will try again in {self.retry_after_n_seconds}s.")
                time.sleep(self.retry_after_n_seconds)
                pass

        if not success:
            raise Exception("Could not complete OpenAI call")

        output_seq_score = []

        for index, choice in enumerate(response.choices):
            if "logprobs" in choice and "token_logprobs" in choice["logprobs"]:
                probs = []
                for prob, tok in zip(choice["logprobs"]["token_logprobs"], choice["logprobs"]["tokens"]):
                    if tok not in self.stop and tok != "<|endoftext|>":
                        probs.append(prob)
                    else:
                        probs.append(prob)
                        break

                score = -sum(probs) / len(probs) if len(probs) else 100.0
                output_seq_score.append((choice.text, score))
            else:
                output_seq_score.append((choice.text, index))

        return sorted(output_seq_score, key=lambda x: x[1])
