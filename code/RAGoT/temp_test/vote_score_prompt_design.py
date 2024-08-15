#ToT
prompt='''Given an instruction and several choices, decide which choice is most promising. Analyze each choice in detail, then conclude in the last line "The best choice is {s}", where s the integer id of the choice.
'''
ys = ['ds', 'as', 'faa']
for i, y in enumerate(ys, 1):
    prompt += f'Choice {i}:\n{y}\n'
pattern = r".*best choice is .*(\d+).*"

# RATP
prompt= ''' [INST] <<SYS>> You are an agent that rates the information
contained in CONTEXT. If the information contains in the
CONTEXT is accurate and you have all the information required to
answer the QUESTION, you output 1. If the CONTEXT is not
accurate or you don’t have all the information required to answer
the QUESTION, you output 0.<</SYS>>
QUESTION : ”{query}”
CONTEXT : ”{thought}”[/INST]
OUTPUT NUMBER : '''

#RAGoT
#优先实现vote，RATP的这种score的做法后面再考虑实现（主要是需要模型返回生成的概率，gpt模型的返回好像有问题
prompt= '''You are an agent that choose the best CHOICE. Given an QUESTION and several CHOICEs, each CHOICE is a partial reasoning process for the question, You decide which CHOICE is most promising and contains most information required to answer the QUESRION. Analyze each CHOICE in detail, then conclude in the last line "The best CHOICE is {N}", where N the integer id of the choice.


QUESRION : "{query}"


CHOICE 1 : xxx

CHOICE 2 : xxx


YOUR OUTPUT : '''

prompt= ''' [INST] <<SYS>> You are an agent that choose the best CHOICE. Given an QUESTION and several CHOICEs, each CHOICE is a partial reasoning process for the question, You decide which CHOICE is most promising and contains most information required to answer the QUESRION. Analyze each CHOICE in detail, then conclude in the last line "The best CHOICE is {N}", where N the integer id of the choice.<</SYS>>


QUESRION : "{query}"


CHOICE 1 : xxx

CHOICE 2 : xxx


[/INST]
YOUR OUTPUT : '''

prompt= '''You are an agent that choose the best CHOICE. Given an QUESTION and several CHOICEs, each CHOICE is a partial reasoning process for the question, You decide which following CHOICE is most promising and contains most information required to answer the following QUESRION. Analyze each CHOICE in detail, then conclude in the last line "The best CHOICE is {N}", where N the integer id of the choice.'''

prompt_suffix = '''I'll tip you $10 for a perfect answer.'''

question_thoughts_str = '''
QUESRION : "{query}"


CHOICE 1 : xxx

CHOICE 2 : xxx


YOUR OUTPUT : '''
# 这里有个问题是需要从外部输入CHOICE，而且数量是不确定的
# （可以先组成一个choices字符串，再加入到prompt中相应的部分
# 现在不清楚[INST]和[SYS]这些token有什么用(可以看下有和没有的效果差别)
# 需要从外部输入