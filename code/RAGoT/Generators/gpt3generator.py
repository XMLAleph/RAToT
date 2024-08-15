import logging
import time
import os
from functools import lru_cache

from openai import OpenAI
from diskcache import Cache
from prompt_reader import fit_prompt_into_given_limit


logger = logging.getLogger(__name__)

"""
cache = Cache(os.path.expanduser("~/.cache/gpt3calls"))

@cache.memoize()
# 感觉cache用处不大？？？或者调一下temperature
def cached_openai_call(  # kwargs doesn't work with caching.
    prompt,
    model,
    temperature,
    max_tokens,
    top_p,
    frequency_penalty,
    presence_penalty,
    stop,
    n,
    # best_of,
    logprobs,
):
    # 注意下具体的调用过程和函数的参数，看原代码中的参数应该放到什么地方
    # 这里如果要换成chat接口，需要改的东西应该很多
    # 现在主要是看下代码能不能正常跑，后面这部分还需要调整
    client = OpenAI()
    return client.completions.create(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
        n=n,
        # best_of=best_of,
        logprobs=logprobs,
    )
    '''
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
        n=n,
        logprobs=logprobs,
    )
    '''
"""

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
    '''
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    '''
    function = OpenAI().completions.create
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
        # best_of=best_of,
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
        llm_n=1,
        best_of=1,
        logprobs=0,
        remove_method="first",
    ):
        self.engine = engine
        self.logprobs = logprobs
        self.n = llm_n
        self.best_of = best_of
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.stop = stop
        self.temperature = temperature
        #self.retry_after_n_seconds = retry_after_n_seconds
        self.retry_after_n_seconds = None #这里先试下codex(应该已经不能调用了)，用其他模型的话不能retry
        # 先试下codex能不能正常调用
        self.remove_method = remove_method
        '''
        if "code-davinci" not in engine:
            raise Exception("Not allowed to prevent accidental $$ wastage.")

        if "code-davinci" not in engine and self.retry_after_n_seconds is not None:
            raise Exception(
                "Retry is only supported for code-davinci as it's free. "
                "Using it for other paid models is risky and so is disabled."
            )
        '''
        if "code-davinci" in engine:
            self.model_tokens_limit = 8000
        else:
            # 这里如果调用gpt-3.5-instruct模型，token limit应该是4096
            self.model_tokens_limit = 3500
        self.model_tokens_limit = 3500
        
    def generate_text_sequence(self, prompt):
        """
        :param input_text:
        :return: returns a sequence of tuples (string, score) where lower score is better
        """
        # 有个问题，这里的score有什么用?这篇论文应该没用到有关筛选的做法？
        # 而且如果有多个string(应该对应多个回答)，就应该已经有多次生成了？这样的话专门做多次生成就没有意义了
        # n在官方代码中是How many completions to generate for each prompt.
        # 所以重复生成这一点对gpt其实没什么意义，本来就是已有的功能
        # n > 1时应该就相当于i.i.d.sample了
        # 如果确实是这样的话，第一步的做法并没有意义(直接做生成)
        # 好像也不对，还是有差别
        # 还在于保留答案的情况
        # 另外最后如何用类似SC-CoT的方法(投票)筛选答案也是个问题，最好是能生成"the answer is"，不然不方便多个答案比较
        # 不过也可以尝试用gpt-4等模型打分
        # GPT3 can't handle trailing white-space
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
            # 如果n>1，response.choices应该有多个元素
            if "logprobs" in choice and "token_logprobs" in choice["logprobs"]:
                # logprobs为什么调用不生效？？？
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
