# 导入所需的库
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
import time
from datetime import datetime
from human_eval.data import write_jsonl, read_problems
import torch.nn as nn


class Pipeline:
    def __init__(self,model_id,api_key=None):
        self.model_name = "/run/wbw/Qwen-main/model/qwen/CodeQwen1___5-7B-Chat"
        self.tokenizer_name = "/run/wbw/Qwen-main/model/qwen/CodeQwen1___5-7B-Chat"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="torch.bfloat16",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.problems = read_problems('/run/wbw/humaneval-x/codegeex/benchmark/humaneval-x/python/data/humaneval_python.jsonl.gz')
    def get_respond(self,meta_prompt,user_prompt):
        messages = [
            {"role": "system", "content": meta_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt", return_token_type_ids=False).to(device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.4,
            top_p=0.9
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response


class BoT:
    def __init__(self, user_input,problems,api_key=None,model_id=None,need_check=False):
        self.api_key = api_key
        self.model_id = model_id
        self.pipeline = Pipeline(self.model_id,self.api_key)
        self.user_input = user_input
        # Only for test use, stay tuned for our update
        self.problems = problems
        self.need_check = need_check
    def update_input(self,new_input):
        self.user_input = new_input
    def problem_distillation(self):
        print(f'User prompt:{self.user_input}')
        self.distilled_information = self.pipeline.get_respond(meta_distiller_prompt, self.user_input)
        print(f'Distilled information:{self.distilled_information}')

    def buffer_retrieve(self):
        # For initial test use, we will later update the embedding retrieval version to support more
        if self.problem_id == 0:
            self.thought_template = game24
        elif self.problem_id == 1:
            self.thought_template = checkmate
        elif self.problem_id == 2:
            self.thought_template = word_sorting

    def reasoner_instantiation(self):
        # Temporay using selection method to select answer extract method
        problem_id_list = [0,1,2]
        self.instantiation_instruct = """
You are a Python expert in problem analysis and can apply previous problem-solving approaches to new issues.  The user will provide a specific task description and a thought template.  Your goal is to analyze the user's task and generate a Python code based on the thought template.  Only provide the code and let the compiler handle it.  
It should be noted that all the python code should be within one code block, the answer should not include more than one code block!  And strictly follow the thought-template to instantiate the python code but you should also adjust the input parameter according to the user input!
        """

        self.formated_input = f"""
Distilled information:
{self.distilled_information}
User Input:
{self.user_input}
Thought template:
{self.thought_template}

Instantiated Solution:
Please analyze the above user task description and thought template, and generate a specific, detailed solution.  Only provide the Python code.  
        """
        self.inspector_prompt = """
You are an excellent python programming master who are proficient in analyzing and editing python code, and you are also good at understanding the real-world problem. Your task is:
1. Analyze the given python code
2. Edit the input code to make sure the edited code is correct and could run and solve the problem correctly.  
Your respond should follow the format below:
```python
## Edited code here
```
        """
        self.result = self.pipeline.get_respond(self.instantiation_instruct,self.formated_input)
        print(f'Instantiated reasoning result: {self.result}')
        if self.problem_id in problem_id_list:
            self.final_result, code_str = extract_and_execute_code(self.result)
            if self.need_check:
                self.count = 0
                self.inter_input = f"""
                User_input:{self.user_input}
                {code_str}
                {self.final_result}
                """
                self.inter_result = self.final_result
                while(('An error occurred' in self.inter_result) or (self.inter_result == '') or (self.inter_result == 'None')):
                    print('The code cannot be executed correctly, here we continue the edit phase:',self.inter_result)
                    print('The problem code is:',code_str)
                    self.inter_input = self.pipeline.get_respond(self.inspector_prompt,self.inter_input)
                    print(self.inter_input)
                    self.inter_result, inter_code_str = extract_and_execute_code(self.inter_input)
                    self.inter_input = f"""
                User_input:{self.user_input}
                {inter_code_str}
                The result of code execution: {self.inter_result}
                """
                    self.count = self.count + 1
                    if self.count > 3:
                        break
                self.final_result = self.inter_result
            print(f'The result of code execution: {self.final_result}')
        else:
            self.final_result = self.result


    def bot_run(self):
        self.problem_distillation()
        # self.buffer_retrieve()
        self.reasoner_instantiation()
        return self.final_result
