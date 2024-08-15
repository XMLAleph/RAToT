# 测试gpt-3.5接口调用时的一些问题
# 这里调整一下，后面用来调试ragot的prompt的效果

from openai import OpenAI
from prompt_reader import read_prompt

'''
prompt = """Q:"Who was president of the U.S. when superconductivity was discovered?"
A:Let's think step by step.
"""

prompt = "In what country was Lost Gravity manufactured?"
'''
def get_prompt(prompt_file, prompt_reader_args):
    prompt_reader_args = prompt_reader_args or {}
    prompt_reader_args["file_path"] = prompt_file
    prompt = read_prompt(**prompt_reader_args)
    return prompt

def get_response(prompt):
    client = OpenAI(
        api_key="sk-cJ5OVoJbe957yGuIENMvea6fpBnIYlnNUYwfWeHiFLzdrT0C",
    # 这里将官方的接口访问地址，替换成aiproxy的入口地址
        base_url="https://api.aiproxy.io/v1"
    )
    
    response = client.completions.create(
        prompt=prompt,
        model="gpt-3.5-turbo-instruct",
        temperature=0.8,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        n=1,
        best_of=1,
        logprobs=0,
    )
    stop=["\n"],
    '''
    completions = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        max_tokens=500,
        prompt=prompt,
        temperature=0.8,
        n=2,
        logprobs=0,
    )
    for choice in response.choices:
        print(choice.text)
        print(choice.logprobs)
    '''
    output_seq_score = []
    
    for index, choice in enumerate(response.choices):
        # 如果n>1，response.choices应该有多个元素
        if "logprobs" in choice and "token_logprobs" in choice["logprobs"]:
            probs = []
            for prob, tok in zip(choice["logprobs"]["token_logprobs"], choice["logprobs"]["tokens"]):
                if tok not in stop and tok != "<|endoftext|>":
                    probs.append(prob)
                else:
                    probs.append(prob)
                    break
    
            score = -sum(probs) / len(probs) if len(probs) else 100.0
            output_seq_score.append((choice.text, score))
        else:
            output_seq_score.append((choice.text, index))
    
    return output_seq_score

valid_qids = ["028eaef60bdb11eba7f7acde48001122", "8727d1280bdc11eba7f7acde48001122", "79a863dc0bdc11eba7f7acde48001122", "4724c54e08e011ebbda1ac1f6bf848b6", "e5150a5a0bda11eba7f7acde48001122", "35bf3490096d11ebbdafac1f6bf848b6", "a5995da508ab11ebbd82ac1f6bf848b6", "228546780bdd11eba7f7acde48001122", "97954d9408b011ebbd84ac1f6bf848b6", "f44939100bda11eba7f7acde48001122", "1ceeab380baf11ebab90acde48001122", "f86b4a28091711ebbdaeac1f6bf848b6", "c6f63bfb089e11ebbd78ac1f6bf848b6", "af8c6722088b11ebbd6fac1f6bf848b6", "5897ec7a086c11ebbd61ac1f6bf848b6"];
prompt_file = ""
prompt_reader_args = {
    "filter_by_key_values": {
        "qid": valid_qids
    },
    "order_by_key": "qid",
    "estimated_generation_length": 300,
    "shuffle": False,
    "model_length_limit": 4000,
}
prompt = get_prompt(prompt_file, prompt_reader_args) # 也可以改成一般的prompt

prompt = "\n\n\n".join([self.prompt, test_example_str]).strip()
llm_api_output = get_response(prompt)
print(llm_api_output)