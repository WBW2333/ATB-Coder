import json

from setuptools.sandbox import save_path

from bot_pipeline import BoT
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

import torch
import time
from datetime import datetime
from human_eval.data import write_jsonl, read_problems
import torch.nn as nn
from FlagEmbedding import BGEM3FlagModel
import re
from joblib.testing import timeout
from func_timeout import func_set_timeout


class Logger(object):
    def __init__(self, fileN="output.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


import sys

 # 创建解析器对象
parser = argparse.ArgumentParser(description="读取终端命令参数")

# 添加命令行参数
parser.add_argument('--model', type=str, help="LLM", default=" ")
parser.add_argument('--model_path', type=str, help="LLM路径", default=" ")
parser.add_argument('--problem', type=str, help="测试集路径")
parser.add_argument('--ATP', type=str, help="算法模版文件路径", default='')
parser.add_argument('--tag', type=str, help="测试集题目标签", default="Unknow")
parser.add_argument('--info', type=str, help="测试方法", default="Unknow")

# 解析命令行参数
args = parser.parse_args()

# model_name = "E:/Work/1/pythonProject/ChatGLM-main/model/ZhipuAI/codegeex2-6b"
# model_name = "E:/Work/1/pythonProject/ChatGLM-main/model/ZhipuAI/codegeex4-all-9b"
# model_name = "E:/Work/1/pythonProject/Qwen-main/model/qwen/CodeQwen1.5-7B-Chat"
model_name = args.model_path


# problems = read_problems('E:/Work/1/pythonProject/humaneval-x/codegeex/benchmark/humaneval-x/python/data/humaneval_python.jsonl.gz')

# 定义读取 JSON 文件的函数
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


# 读取 JSON 文件
# print("easy")
# file_path = 'E:/Work/1/pythonProject/humaneval-x/leetcode-benchmark/leetcode_passk/easy-bench.json'
# print("medium")
# file_path = 'E:/Work/1/pythonProject/humaneval-x/leetcode-benchmark/leetcode_passk/medium-bench.json'
# print("hard")
# file_path = 'E:/Work/1/pythonProject/humaneval-x/leetcode-benchmark/leetcode_passk/hard-bench.json'
print(args.tag)
file_path = args.problem

# print("无")
# print("ATP")
# print("全面AT")
# print("CoT-baseline")
# print("ont-shot-baseline")
# print("kare-baseline")
# print("scot-baseline")
# print("小AT")
# print("仅CoT")
# print("仅AT")
print(args.info)
problems = read_json_file(file_path)


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 100

# 读取模型参数和tokenizer
model = 100
tokenizer = 100
if args.model == 'CodeQwen':
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
elif args.model == 'CodeGeeX':
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


# BGE，读取模板
sentences = []
dic = []
# 从JSONL文件逐行读取数据
# with open('E:/Work/1/pythonProject/humaneval-x/leetcode-benchmark/thought_templates/thought_templates_small.jsonl', 'r') as f:
# with open('E:/Work/1/pythonProject/humaneval-x/leetcode-benchmark/thought_templates/thought_templates.jsonl', 'r') as f:
# with open('E:/Work/1/pythonProject/humaneval-x/leetcode-benchmark/thought_templates/output.jsonl', 'r', encoding='utf-8') as f:
if not args.ATP == '':
    jsons = read_json_file(args.ATP)
    for item in jsons:
        dic.append(item)
        sentences.append(item['thought_question'])

# 加载嵌入模型
bgeModel = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
output_templates = bgeModel.encode(sentences, return_colbert_vecs=True)
# bgeModel = 100
# output_templates = 100


def get_func_name(text):
    code_match = re.search(r'python\n(.*?)\n', text, re.DOTALL)
    if code_match:
        code = code_match.group(1)
        return code
    return ""

def get_func_define(text):
    function_match = re.search(r'def\s+(\w+)\s*', text)
    if function_match:
        return function_match.group(1)
    return ""

def find_between_strings(s):
    # 使用正则表达式来匹配 /box 和 /ass 之间的内容
    pattern = r'```python(.*?)\```'
    matches = re.findall(pattern, s, re.DOTALL)
    return matches

load_path = (args.model + '_' + args.problem.replace('.json', '') + '_' + args.ATP.replace('.json', '') + '_' +  args.tag + '_' + args.info + '.json').replace('/', '-')
# 检查文件是否存在
if not os.path.exists(load_path):
    # 如果文件不存在，写入一个空数组
    with open(load_path, 'w', encoding='utf-8') as file:
        json.dump([], file)

run_temp = {}

@func_set_timeout(3) # 设定函数执行时间
def run_func(problem, func_define, result):
    print(result)
    if result is None:
        return -1
    print("Begin Func ---------------------------------")
    correct = 1
    result = process_code(result)
    # print("结果：")
    # print(result)
    for input in problem['example']:
        code_run = '\nresult = ' + func_define + '(' + input['input'] + ")"
        # print(code_run)
        # 定义一个空的字典来存储执行代码后的变量
        exec_globals = {}
        ans = "空结果"
        run_temp['run'] = 'from typing import List\n' + result + code_run
        print('---------------------------------------------------------------------------------------------')
        print(run_temp['run'])

        try:
            # 使用exec函数执行代码字符串
            exec(run_temp['run'], exec_globals)
            # 获取结果
            ans = exec_globals['result']
        except Exception as e:
            # 捕获异常并输出异常信息
            ans = f"An error occurred: {e}"
            correct = -1
        try:
            print(f"输出结果：{ans}，样例答案：{eval(input['output'])}，监测结果：{ans == eval(input['output'])}")
            if not ans == eval(input['output']):
                correct = 0
                break
        except Exception as e:
            correct = -1
            break
    return correct

def process_code(code_str):
    # 检查字符串是否以 'class Solution:' 开头
    if code_str.strip().startswith("class Solution:"):
        # 去掉 'class Solution:' 这一行
        lines = code_str.splitlines()
        processed_lines = []
        for line in lines[1:]:  # 跳过第一行
            # 去掉每行开头的一个缩进符（假设缩进为4个空格或一个制表符）
            if line.startswith("    "):  # 如果缩进是4个空格
                processed_lines.append(line[4:])
            elif line.startswith("\t"):  # 如果缩进是一个制表符
                processed_lines.append(line[1:])
            elif line.startswith("    ```python"):
                continue
            else:
                processed_lines.append(line)

        # 将处理后的代码行重新组合成字符串
        processed_code = "\n".join(processed_lines)
        return processed_code
    else:
        return code_str  # 如果不以 'class Solution:' 开头，返回原始字符串

# 定义生成多个完成的函数
def generate_multiple_completions(problems, num_samples_per_task, start=True):
    results = read_json_file(load_path)
    right = 0
    wrong = 0
    error = 0
    start = 0
    do_run = True
    for item in results:
        if item['correct'] == 1:
            right += 1
        elif item['correct'] == 0:
            wrong += 1
        else:
            error += 1
        start = item['id']
        do_run = False
    template = []
    bench = []
    for problem in problems:
        if not do_run:
            if problem['id'] == start:
                do_run = 1
            continue
        print(datetime.now().strftime("%H:%M:%S"), problem['id'])
        func_name = get_func_name(problem['python'])
        func_define = get_func_define(func_name)
        print(f"函数调用：{func_name}，函数定义：{func_define}")
        test_bot = BoT(
            # user_input为输入的问题
            user_input=problem['content']['problem'] + '\n```' + func_name + '\n```' + problem['content']['constraints'] if problem['content']['constraints'] else "" + '\n' + problem['content']['follow_up'] if problem['content']['follow_up'] else "",
            model_name=model_name,
            problem=problem,
            problems=problems,
            device=device,
            model=model,
            bgeModel=bgeModel,
            dic=dic,
            output_templates=output_templates,
            tokenizer=tokenizer,
            CoT = True,
            ATP = True,
            GPT_CoT = False,
            one_shot = False,
            SCoT = False
        )
        for _ in range(num_samples_per_task):
            correct = -1
            run_temp = {
                'id': problem['id'],
                'func': f"函数调用：{func_name}，函数定义：{func_define}",
                'result': "NULL",
                'correct': 0,
                'run': "NULL",
            }
            results.append(run_temp)
            result = test_bot.bot_run()
            # print(result)
            # result = problem['python']
            try:
                print('run func ')
                correct = run_func(problem=problem, func_define=func_define, result=result)
            except:
                correct = -1
            run_temp['correct'] = correct
            run_temp['result'] = "AC" if correct == 1 else ('WA' if correct == 0 else 'RE')
            if correct == 1:
                right += 1
                template.append(problem)
            elif correct == 0:
                wrong += 1
                bench.append(problem)
            else:
                error += 1
                bench.append(problem)
            print(f"累计正确：{right}，错误：{wrong}，运行时出错：{error}")
    results.append({
        '--model': args.model,
        '--model_path': args.model_path,
        '--problem': args.problem,
        '--ATP': args.ATP,
        '--tag': args.tag,
        '--info': args.info,
    })
    with open(load_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    # with open('problems-for-template-total.json', 'w', encoding='utf-8') as f:
    #     json.dump(template, f, ensure_ascii=False, indent=4)
    # with open('problems-for-bench.json', 'w', encoding='utf-8') as f:
    #     json.dump(bench, f, ensure_ascii=False, indent=4)
    return right/(right+wrong+error)


if __name__ == "__main__":
    # 生成多条记录
    num_samples_per_task = 1  # 每个任务生成的样本数量
    # print(problems)
    # all_task_ids = list(problems.keys())
    # all_prompts = [problems[task_id]["prompt"] for task_id in all_task_ids]

    # 直接生成所有记录
    samples = generate_multiple_completions(problems, num_samples_per_task)
    print(samples)
    # write_jsonl("chatCodeQwenPython.jsonl", samples, True)
