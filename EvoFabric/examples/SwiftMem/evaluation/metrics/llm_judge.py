# evaluation/metrics/llm_judge.py
"""LLM-based evaluation judge"""

import json
import os
from openai import OpenAI
import dotenv

dotenv.load_dotenv()

# Read configuration from environment variables
api_key = os.getenv("LLM_API_KEY")
base_url = os.getenv("LLM_BASE_URL")
model = os.getenv("LLM_MODEL", "gpt-4o-mini")

client = OpenAI(api_key=api_key, base_url=base_url)

ACCURACY_PROMPT = """
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user), 
    (2) a 'gold' (ground truth) answer, 
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""


def evaluate_llm_judge(question, gold_answer, generated_answer):
    """Evaluate the generated answer against the gold answer using an LLM judge."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert grader that determines if answers to questions match a gold standard answer.",
                },
                {
                    "role": "user",
                    "content": ACCURACY_PROMPT.format(
                        question=question, gold_answer=gold_answer, generated_answer=generated_answer
                    ),
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        label = json.loads(response.choices[0].message.content)["label"]
        return 1 if label == "CORRECT" else 0
    except Exception as e:
        print(f"Error in LLM judge: {e}")
        return 0