from typing import Any
import time
from openai import OpenAI

default_template = """
Determine if the following two answers are consistent: 
Ground Truth: {ground_truth}
Model Answer: {model_answer}

please only return "yes" if they convey the same essential information, or "no" if they do not.
"""

# Set your own api key\
client = OpenAI(api_key='YOUR_API_KEY')

def GPT4_eval(user_prompt):
    # 设置重试次数和初始重试间隔（秒）
    max_retries = 10  # 例如，最多重试10次
    retry_interval = 1  # 初始重试间隔为1秒

    for attempt in range(max_retries):
        try:
            result = openai_request(user_prompt)
            break

        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_interval)
                retry_interval *= 2  # 按指数增长重试间隔
            else:
                result = "#### ERROR ####"
                break

    return result


def openai_request(prompt: str):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    result = str(completion.choices[0].message.content)
    
    return result


class GPT4Eval:
    def __init__(
        self,
    ):
        pass

    def __call__(self, doc, ground_truth, results) -> Any:
        """Take a single document and the LM input/output/ground_truth.
        Returns the  values of the metric for that one document
        """
        try:
            template = default_template
            prompt = template.format(ground_truth=ground_truth, model_answer=results[0])
            res = GPT4_eval(prompt)
            print(res)
            
            if res == "#### ERROR ####":
                print(f"ERROR on {ground_truth} and {results[0]}")
            
        except Exception as e:
            print(e,doc,results)
            print('-'*100)
            res = ""
            
        
        res = 1.0 if res.strip(" \n.,").lower() == "yes" else 0.0
        return res

if __name__ == "__main__":
    print(openai_request("hello, this is a test"))