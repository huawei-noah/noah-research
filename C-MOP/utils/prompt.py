
PROMPT_IMPROVEMENT_TEMPLATE = """I'm trying to write a prompt about some different tasks.

My current prompt is:
"{prompt}"

But it gets the some examples wrong. Based on these wrong examples, the problem with this prompt is that {new_constraint}

Note that new prompts should not focus excessively on or be limited to a single example of an error. Instead, they should be generated from a more generalized perspective, based on the analyzed causes of the errors.

Please do not provide any examples(i.e. few-shot examples) in the new prompt; no other redundant information is needed.

Based on the above information, you should write Four different improved prompts.

Each new prompt is wrapped with <START> and <END>.

The Four new prompts are: """

ERROR_ANALYSIS_TEMPLATE = """I'm trying to write a prompt about some different tasks.

My current prompt is:
"{current_prompt}"

However, this system prompt causes errors in samples from various different tasks:
{failure}

Please give 3 reasons why the prompt could have gotten these examples wrong.
Please wrap each reason with <START> and <END> tags !!"""


DEFAULT_SYSTEM_PROMPT = "You are an expert at accomplishing tasks. You are a highly capable problem-solver."
