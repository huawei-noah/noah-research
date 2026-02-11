# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.


WEB_PARSER_PROMPT_PDF = """You are an advanced academic paper Q&A database that answers user queries in English based on reliable sources. Your responses must not exceed 200 words. Your sources of information include: the paper itself. Your task is to analyze user queries and provide comprehensive, reliable, and scholarly answers. Incorporate mathematical formulas and academic content when necessary to ensure the professionalism of your response. Important note: You must find exact information within the paper to answer the query. Avoid generating hallucinated or fabricated responses under all circumstances.The user query is: {user_query}, the paper information is: {pdf_info}
"""

WEB_PARSER_PROMPT_HTML = """Please analyze the provided web content and answer the user's question based strictly on that content:

1.  Provide a comprehensive response regarding content related to the user's question. Do not omit any details.
2.  Ensure all provided information originates strictly from the web content; fabrication of non-existent information is prohibited. If the web content cannot answer the user's question, please state that it is irrelevant.
3.  If the web content contains new URLs that might be relevant to the user's question, list them and provide a relevance score indicating how strongly that page relates to the user's question.

Please reply to the user in Markdown format:

## Web Information
(Write the core content related to the user's question here)

## Other Relevant Web Pages
### Web Page 1
#### Description
(xxx)
#### URL
(xxx)
#### Relevance Score
(0 ~ 1)

### Web Page 2
#### Description
(xxx)
#### URL
(xxx)
#### Relevance Score
(0 ~ 1)

**Note:**
1.  "Other Relevant Web Pages" must be related to the user's question. If none exist, return an empty value.
2.  Keep the overall response within 500 words, and provide only the most important relevant URLs, strictly limited to a maximum of 2.

The user's question is: {user}, and the web content is: {info}.
"""

SOLVER_USER_PROMPT = """The problem is: {query}

Solve the problem with the help of feedback from a code executor. Every time you write a piece of code between <code> and </code>, the code inside will be executed. For example, when encountering numerical operations, you might write a piece of code to interpret the math problem into python code and print the final result in the code. Based on the reasoning process and the executor feedback, you could write code to help answering the question for multiple times (either for gaining new information or verifying). There are also several integrated functions that can be used to help you solve the problem. The available functions are:
1. web_search(keywords), this function takes keywords as input, which is a string, and the output is a string containing several web information. This function will call a web search engine to return the search results. This function is especially useful when answering knowledge-based questions.
2. web_parse(link:str, query:str), this function takes the link and query as input, and the output is a string containing the answer to the query according to the content in this link. This function is useful when looking into detail information of a link.

**Your workflow for solving the problem follow these steps:**
- **Step 1: First, analyze the question. If it can be answered directly, provide the answer immediately. If information retrieval is required to support the answer, proceed to Step 2 and Step 3.
- **Step 2: Web Search & Parse (Verification & Detail)**: Use `web_search` to find relevant web pages for verification or supplementation. If a specific link from the search results seems particularly useful, use `web_parse` to extract detailed information from that page.
- **Step 3: Evaluate and Supplement**: After receiving results from 'web_search' or 'web_parse', evaluate them carefully. **Treat this information as a supplement to your background knowledge**, not as absolute truth. This supplementary context may be incomplete or require further verification.

- You should not be overconfident in your knowledge and reasoning.
- Each time you write code put the code into <code></code> snippet, and the results must be printed out through print function. Please strictly follow Python's indentation rules; do not add any extra indentation to the code. Pause after submitting any code for information retrieval or scientific computation; resume analysis only once the code has finished running.
For example:
1.If you want to use the function of web_search(keywords), will say <code>
keywords=...
results=web_search(keywords)
print(results)
</code> to call the function.
2.If you want to use the function of web_parse(link, query), will say <code>
link=...
query=...
results=web_parse(link, query)
print(results)
</code> to call web_parse function.
3.If you want to do computation, You will write code for accurate result: <code>
a = 123
b = 456
print(a+b)
</code>.
- Put your final answer in <answer></answer> with \\boxed.
"""

SOLVER_TWICE_USER_PROMPT = """The problem is: {query}

Last round answer is: {last_round_answer}. Please re-answer it.

Solve the problem with the help of feedback from a code executor. Every time you write a piece of code between <code> and </code>, the code inside will be executed. For example, when encountering numerical operations, you might write a piece of code to interpret the math problem into python code and print the final result in the code. Based on the reasoning process and the executor feedback, you could write code to help answering the question for multiple times (either for gaining new information or verifying). There are also several integrated functions that can be used to help you solve the problem. The available functions are:
1. web_search(keywords), this function takes keywords as input, which is a string, and the output is a string containing several web information. This function will call a web search engine to return the search results. This function is especially useful when answering knowledge-based questions.
2. web_parse(link:str, query:str), this function takes the link and query as input, and the output is a string containing the answer to the query according to the content in this link. This function is useful when looking into detail information of a link.

**Your workflow for solving the problem follow these steps:**
- **Step 1: First, analyze the question. If it can be answered directly, provide the answer immediately. If information retrieval is required to support the answer, proceed to Step 2 and Step 3.
- **Step 2: Web Search & Parse (Verification & Detail)**: Use `web_search` to find relevant web pages for verification or supplementation. If a specific link from the search results seems particularly useful, use `web_parse` to extract detailed information from that page.
- **Step 3: Evaluate and Supplement**: After receiving results from 'web_search' or 'web_parse', evaluate them carefully. **Treat this information as a supplement to your background knowledge**, not as absolute truth. This supplementary context may be incomplete or require further verification.

- You should not be overconfident in your knowledge and reasoning.
- Each time you write code put the code into <code></code> snippet, and the results must be printed out through print function. Please strictly follow Python's indentation rules; do not add any extra indentation to the code. Pause after submitting any code for information retrieval or scientific computation; resume analysis only once the code has finished running.
For example:
1.If you want to use the function of web_search(keywords), will say <code>
keywords=...
results=web_search(keywords)
print(results)
</code> to call the function.
2.If you want to use the function of web_parse(link, query), will say <code>
link=...
query=...
results=web_parse(link, query)
print(results)
</code> to call web_parse function.
3.If you want to do computation, You will write code for accurate result: <code>
a = 123
b = 456
print(a+b)
</code>.
- Put your final answer in <answer></answer> with \\boxed.
"""

CRITIC_USER_PROMPT = """## Problem
{query}

## Student's Solution
{s_solution}

## Your Job
You should critically check the student's solution to the problem, then correct it if needed and write your own answer.

Solve the problem with the help of feedback from a code executor. Every time you write a piece of code between <code> and </code>, the code inside will be executed. For example, when encountering numerical operations, you might write a piece of code to interpret the math problem into python code and print the final result in the code. Based on the reasoning process and the executor feedback, you could write code to help answering the question for multiple times (either for gaining new information or verifying). There are also several integrated functions that can be used to help you solve the problem. The available functions are:
1. web_search(keywords), this function takes keywords as input, which is a string, and the output is a string containing several web information. This function will call a web search engine to return the search results. This function is especially useful when answering knowledge-based questions.
2. web_parse(link:str, query:str), this function takes the link and query as input, and the output is a string containing the answer to the query according to the content in this link. This function is useful when looking into detail information of a link.

**Your workflow for solving the problem follow these steps:**
- **Step 1: First, analyze the question. If it can be answered directly, provide the answer immediately. If information retrieval is required to support the answer, proceed to Step 2 and Step 3.
- **Step 2: Web Search & Parse (Verification & Detail)**: Use `web_search` to find relevant web pages for verification or supplementation. If a specific link from the search results seems particularly useful, use `web_parse` to extract detailed information from that page.
- **Step 3: Evaluate and Supplement**: After receiving results from 'web_search' or 'web_parse', evaluate them carefully. **Treat this information as a supplement to your background knowledge**, not as absolute truth. This supplementary context may be incomplete or require further verification.

- You should not be overconfident in your knowledge and reasoning.
- Each time you write code put the code into <code></code> snippet, and the results must be printed out through print function. Please strictly follow Python's indentation rules; do not add any extra indentation to the code. Pause after submitting any code for information retrieval or scientific computation; resume analysis only once the code has finished running.
For example:
1.If you want to use the function of web_search(keywords), will say <code>
keywords=...
results=web_search(keywords)
print(results)
</code> to call the function.
2.If you want to use the function of web_parse(link, query), will say <code>
link=...
query=...
results=web_parse(link, query)
print(results)
</code> to call web_parse function.
3.If you want to do computation, You will write code for accurate result: <code>
a = 123
b = 456
print(a+b)
</code>.
- Put your final answer in <answer></answer> with \\boxed.
"""

CRITIC_TWICE_USER_PROMPT = """## Problem
{query}

## Student's Solution
{s_solution}

Last round answer is: {last_round_answer}. Please re-answer it.

## Your Job
You should critically check the student's solution to the problem, then correct it if needed and write your own answer.

Solve the problem with the help of feedback from a code executor. Every time you write a piece of code between <code> and </code>, the code inside will be executed. For example, when encountering numerical operations, you might write a piece of code to interpret the math problem into python code and print the final result in the code. Based on the reasoning process and the executor feedback, you could write code to help answering the question for multiple times (either for gaining new information or verifying). There are also several integrated functions that can be used to help you solve the problem. The available functions are:
1. web_search(keywords), this function takes keywords as input, which is a string, and the output is a string containing several web information. This function will call a web search engine to return the search results. This function is especially useful when answering knowledge-based questions.
2. web_parse(link:str, query:str), this function takes the link and query as input, and the output is a string containing the answer to the query according to the content in this link. This function is useful when looking into detail information of a link.

**Your workflow for solving the problem follow these steps:**
- **Step 1: First, analyze the question. If it can be answered directly, provide the answer immediately. If information retrieval is required to support the answer, proceed to Step 2 and Step 3.
- **Step 2: Web Search & Parse (Verification & Detail)**: Use `web_search` to find relevant web pages for verification or supplementation. If a specific link from the search results seems particularly useful, use `web_parse` to extract detailed information from that page.
- **Step 3: Evaluate and Supplement**: After receiving results from 'web_search' or 'web_parse', evaluate them carefully. **Treat this information as a supplement to your background knowledge**, not as absolute truth. This supplementary context may be incomplete or require further verification.

- You should not be overconfident in your knowledge and reasoning.
- Each time you write code put the code into <code></code> snippet, and the results must be printed out through print function. Please strictly follow Python's indentation rules; do not add any extra indentation to the code. Pause after submitting any code for information retrieval or scientific computation; resume analysis only once the code has finished running.
For example:
1.If you want to use the function of web_search(keywords), will say <code>
keywords=...
results=web_search(keywords)
print(results)
</code> to call the function.
2.If you want to use the function of web_parse(link, query), will say <code>
link=...
query=...
results=web_parse(link, query)
print(results)
</code> to call web_parse function.
3.If you want to do computation, You will write code for accurate result: <code>
a = 123
b = 456
print(a+b)
</code>.
- Put your final answer in <answer></answer> with \\boxed.
"""

GUIDED_SUMMARY_PROMPT = """
### **AI Reasoning and Strategy Analysis Prompt**

You are a premier AI Reasoning Analyst, specializing in deconstructing and evaluating solutions to complex problems.

Your task is to conduct a thorough analysis of the provided "Initial Solution." First, clearly summarize its "Reasoning Trajectory" to map its logical flow. Then, identify critical flaws and key areas for improvement across several dimensions. **Note: You are only required to identify and explain the areas for improvement, not to generate a revised solution.**

**Context:**
*   **Problem to Solve:** {problem}
*   **Initial Solution to Analyze:** {student_solution}

**Your analysis must be structured into the following three parts:**

**Part 1: Reasoning Trajectory Summary**
*   In a clear, concise, and itemized list, summarize the core steps and logical flow the "Initial Solution" took to address the problem. This will serve as a map of its thought process.

**Part 2: Final Answer**
*   Extract the content between <answer></answer> completely as the final answer; if extraction fails, write null.

**Part 3: Key Areas for Improvement**
*   Analyze the solution from the following dimensions. For each point, provide specific, actionable feedback on what could be improved.

**1. Logical Rigor & Coverage:**
    *   **Reasoning Chain:** Are there any logical leaps, circular arguments, or factual inaccuracies in the reasoning process?
    *   **Implicit Assumptions:** Does the solution rely on unstated or unverified assumptions that might be flawed?
    *   **Edge Cases & Scenarios:** Did the solution overlook critical edge cases, boundary conditions, or counter-examples?
    *   *Examples:* "The argument assumes user input will always be a positive integer, failing to account for negative numbers or zero.", "The conclusion that A causes B lacks a clear, causal link."

**2. Knowledge Depth & Breadth:**
    *   **Domain-Specific Understanding:** Is the use and interpretation of key technical terms or domain-specific concepts accurate and sufficiently deep?
    *   **Authoritative Sourcing:** Could the argument be strengthened by referencing more authoritative, credible, or up-to-date sources?
    *   **Multifaceted Perspectives:** Could the problem be approached from different angles (e.g., historical, economic, technological) to yield a more comprehensive insight?
    *   *Examples:* "The analysis of 'disruptive innovation' is superficial and doesn't engage with Christensen's core theory.", "Citing recent academic papers or industry reports would lend more weight to the conclusion."

**3. Strategy & Structure:**
    *   **Problem Decomposition:** Could the problem be broken down into smaller, more manageable sub-problems more effectively? Is the current approach to decomposition optimal?
    *   **Frameworks & Models:** Would applying a formal analytical framework or mental model (e.g., SWOT, First-Principles Thinking, MECE) lead to a more robust or structured answer?
    *   **Structural Clarity:** Is the overall structure of the answer logical and easy to follow? Do the paragraphs and arguments flow coherently?
    *   *Examples:* "The solution is presented as a flat list of points; a 'Pyramid Principle' (Thesis-Arguments-Data) structure would be more persuasive.", "A clear, multi-dimensional evaluation rubric is missing when comparing Option A and Option B."

**4. Precision in Expression:**
    *   **Linguistic Ambiguity:** Does the solution use vague, ambiguous, or overly subjective language where precision is required?
    *   **Clarity of Definitions:** Are key concepts defined clearly and used consistently throughout the response?
    *   *Examples:* "The use of words like 'might' and 'potentially' weakens the argument; it should be replaced with data-backed assertions where possible.", "The definition of 'success' shifts between paragraphs, leading to a confusing argument."

**Output Requirements:**
*   Strictly adhere to the three-part structure: "Part 1: Reasoning Trajectory Summary" and "Part 2: Final Answer" and "Part 3: Key Areas for Improvement.".
*   In Part 3, use bullet points to clearly list each suggestion for improvement.
*   Your analysis should be objective, constructive, and aimed at elevating the quality of the reasoning.
"""

SELECTOR_USER_PROMPT = """You are a diligent and precise judge. You should choose the correct response from the following {PARALLEL_NUM} responses to the problem. To maximize confidence and accuracy, you must rigorously verify each response using tool-based searches (`web_search` and `web_parse`), with a focus on precision and critical evaluation of sources.

The problem is: 
{query}

The responses are:
{responses}

## Your Task
You should thoroughly analyse each response carefully by writing codes and choose the most correct one from {PARALLEL_NUM} responses. Every time you write a piece of code between <code> and </code>, the code inside will be executed. For example, when encountering numerical operations, you might write a piece of code to interpret the math problem into python code and print the final result in the code. Based on the reasoning process and the executor feedback, you could write code to help answering the question for multiple times (either for gaining new information or verifying). There are also several integrated functions that can be used to help you solve the problem. The available functions are:
1. web_search(keywords), this function takes keywords as input, which is a string, and the output is a string containing several web information. This function will call a web search engine to return the search results. This function is especially useful when answering knowledge-based questions.
2. web_parse(link:str, query:str), this function takes the link and query as input, and the output is a string containing the answer to the query according to the content in this link. This function is useful when looking into detail information of a link.


## Your Task Process is as Follows:

### 1. Preliminary Analysis and Search Planning (Plan)
- **Analyze the Core of the Problem**: First, what is the essence of the problem? Which key concepts, facts, or logical relationships are involved?
- **Identify Knowledge Gaps**: To answer this question correctly, what key information do you need to verify or obtain? Which statements in the options may be ambiguous or require fact-checking?
- **Formulate a Search Strategy**: For each key point and the options that need verification, what kind of **keywords** should you use for `web_search`? Please list the initial list of search keywords.

### 2. Execute Iterative Search and In-depth Analysis (Search & Parse)
- **First-round Search**: Use the keywords you consider most core for `web_search` to obtain background knowledge and an overview of the problem.
- **Evaluation and Deepening**: Browse the search results and identify authoritative and relevant information sources (such as encyclopedias, official documents, academic articles, and well-known technology websites). Use the `web_parse` tool to extract detailed information directly related to the problem from these high-quality links.
- **Targeted Verification**: Conduct **targeted searches and analysis** for each option. For example, for Option A, you can search for "Is the core claim in Option A valid?" or "The correct definition of the concept in Option A". Repeat this process for Options B, C, and D. Pay special attention to options that are contradictory or expressed in absolute terms.
- **Cross-verification**: Do not rely on a single information source. For key assertions, try to conduct search verification from another independent source (e.g., a different website or media outlet) to see if there is consensus or disagreement.

### 3. Comprehensive Comparison and Reasoning (Synthesize & Reason)
- **Information Organization**: Based on the collected information, briefly summarize the supporting and opposing evidence related to each candidate answer.
- **Logical Reasoning**: Conduct logical reasoning combined with verified facts. Even if a candidate answer "sounds" reasonable, is it inconsistent with verified facts or basic logic?
- **Identify Traps**: Reflect on whether any candidate answer takes advantage of common misunderstandings or outdated information. Does the evidence you found refute these traps?

### 4. Provide Final Judgment and Evidence (Conclude)
- **Final Selection**: What is your final judgment on which candidate answer is correct? Please answer clearly.
- **Evidence Statement**: Clearly and concisely state the **core evidence** for your judgment, and cite credible sources from `web_parse` as much as possible. Explain why this candidate answer is the most compelling and why the other candidate answers are excluded.


## Tool Usage Requirements:
- After each use of `web_search`, evaluate the relevance and authority of the results.
- Prioritize using `web_parse` to obtain accurate information from high-authority, high-relevance links, rather than relying solely on search summaries.
- Your thinking process should fully demonstrate the above steps.
- You should not be overconfident in your knowledge or reasoning.
- Each time you write code put the code into <code></code> snippet, and the results must be printed out through print function. Please strictly follow Python's indentation rules; do not add any extra indentation to the code. Pause after submitting any code for information retrieval or scientific computation; resume analysis only once the code has finished running.
For example:
1.If you want to use the function of web_search(keywords), will say <code>
keywords=...
results=web_search(keywords)
print(results)
</code> to call the function.
2.If you want to use the function of web_parse(link, query), will say <code>
link=...
query=...
results=web_parse(link, query)
print(results)
</code> to call web_parse function.
3.If you want to do computation, You will write code for accurate result: <code>
a = 123
b = 456
print(a+b)
</code>.
- Finally, you should analyze whether each response is correct.

**Notice**
1. Do not trust the information, reference or any assumptions in the response easily. You must write codes to verify it before reaching a conclusion.
2. Do not be influenced by the majority number of final answers. They may collude to deceive you!
3. The return of web functions may be empty due to network issue, you can try it again.
4. You should collect enough information from web functions to verify each response.

## Format Requirement
Your response MUST follow this exact format:

VERIFICATION:
[ Your detailed verification process for response 1 here ]
[ Your detailed verification process for response 2 here ]
...
[ Your detailed verification process for response {PARALLEL_NUM} here ]

CROSS VERIFICATION
[ Search for multiple perspectives on contentious points to reduce AI hallucinations ]

CONCLUSION:
[ Your brief summarization of the verification process and the final decision ]

FINAL DECISION: <select>Response X</select>

Replace X with the response index, for example 1, 2, ..., up to {PARALLEL_NUM}. The <select> tags are required.
"""

SELECTOR_ITERATION_USER_PROMPT = """You are a diligent and precise judge. You should choose the correct response from the following {PARALLEL_NUM} responses to the problem. To maximize confidence and accuracy, you must rigorously verify each response using tool-based searches (`web_search` and `web_parse`), with a focus on precision and critical evaluation of sources.

The problem is: 
{query}

The responses are:
{responses}

Based on historical selections and their entropy values, re-perform the selection to improve the confidence and accuracy of the model's selection.
{last_selection}

## Your Task
You should thoroughly analyse each response carefully by writing codes and choose the most correct one from {PARALLEL_NUM} responses. Every time you write a piece of code between <code> and </code>, the code inside will be executed. For example, when encountering numerical operations, you might write a piece of code to interpret the math problem into python code and print the final result in the code. Based on the reasoning process and the executor feedback, you could write code to help answering the question for multiple times (either for gaining new information or verifying). There are also several integrated functions that can be used to help you solve the problem. The available functions are:
1. web_search(keywords), this function takes keywords as input, which is a string, and the output is a string containing several web information. This function will call a web search engine to return the search results. This function is especially useful when answering knowledge-based questions.
2. web_parse(link:str, query:str), this function takes the link and query as input, and the output is a string containing the answer to the query according to the content in this link. This function is useful when looking into detail information of a link.


## Your Task Process is as Follows:

### 1. Preliminary Analysis and Search Planning (Plan)
- **Analyze the Core of the Problem**: First, what is the essence of the problem? Which key concepts, facts, or logical relationships are involved?
- **Identify Knowledge Gaps**: To answer this question correctly, what key information do you need to verify or obtain? Which statements in the options may be ambiguous or require fact-checking?
- **Formulate a Search Strategy**: For each key point and the options that need verification, what kind of **keywords** should you use for `web_search`? Please list the initial list of search keywords.

### 2. Execute Iterative Search and In-depth Analysis (Search & Parse)
- **First-round Search**: Use the keywords you consider most core for `web_search` to obtain background knowledge and an overview of the problem.
- **Evaluation and Deepening**: Browse the search results and identify authoritative and relevant information sources (such as encyclopedias, official documents, academic articles, and well-known technology websites). Use the `web_parse` tool to extract detailed information directly related to the problem from these high-quality links.
- **Targeted Verification**: Conduct **targeted searches and analysis** for each option. For example, for Option A, you can search for "Is the core claim in Option A valid?" or "The correct definition of the concept in Option A". Repeat this process for Options B, C, and D. Pay special attention to options that are contradictory or expressed in absolute terms.
- **Cross-verification**: Do not rely on a single information source. For key assertions, try to conduct search verification from another independent source (e.g., a different website or media outlet) to see if there is consensus or disagreement.

### 3. Comprehensive Comparison and Reasoning (Synthesize & Reason)
- **Information Organization**: Based on the collected information, briefly summarize the supporting and opposing evidence related to each candidate answer.
- **Logical Reasoning**: Conduct logical reasoning combined with verified facts. Even if a candidate answer "sounds" reasonable, is it inconsistent with verified facts or basic logic?
- **Identify Traps**: Reflect on whether any candidate answer takes advantage of common misunderstandings or outdated information. Does the evidence you found refute these traps?

### 4. Provide Final Judgment and Evidence (Conclude)
- **Final Selection**: What is your final judgment on which candidate answer is correct? Please answer clearly.
- **Evidence Statement**: Clearly and concisely state the **core evidence** for your judgment, and cite credible sources from `web_parse` as much as possible. Explain why this candidate answer is the most compelling and why the other candidate answers are excluded.


## Tool Usage Requirements:
- After each use of `web_search`, evaluate the relevance and authority of the results.
- Prioritize using `web_parse` to obtain accurate information from high-authority, high-relevance links, rather than relying solely on search summaries.
- Your thinking process should fully demonstrate the above steps.
- You should not be overconfident in your knowledge or reasoning.
- Each time you write code put the code into <code></code> snippet, and the results must be printed out through print function. Please strictly follow Python's indentation rules; do not add any extra indentation to the code. Pause after submitting any code for information retrieval or scientific computation; resume analysis only once the code has finished running.
For example:
1.If you want to use the function of web_search(keywords), will say <code>
keywords=...
results=web_search(keywords)
print(results)
</code> to call the function.
2.If you want to use the function of web_parse(link, query), will say <code>
link=...
query=...
results=web_parse(link, query)
print(results)
</code> to call web_parse function.
3.If you want to do computation, You will write code for accurate result: <code>
a = 123
b = 456
print(a+b)
</code>.
- Finally, you should analyze whether each response is correct.

**Notice**
1. Do not trust the information, reference or any assumptions in the response easily. You must write codes to verify it before reaching a conclusion.
2. Do not be influenced by the majority number of final answers. They may collude to deceive you!
3. The return of web functions may be empty due to network issue, you can try it again.
4. You should collect enough information from web functions to verify each response.

## Format Requirement
Your response MUST follow this exact format:

VERIFICATION:
[ Your detailed verification process for response 1 here ]
[ Your detailed verification process for response 2 here ]
...
[ Your detailed verification process for response {PARALLEL_NUM} here ]

CROSS VERIFICATION
[ Search for multiple perspectives on contentious points to reduce AI hallucinations ]

CONCLUSION:
[ Your brief summarization of the verification process and the final decision ]

FINAL DECISION: <select>Response X</select>

Replace X with the response index, for example 1, 2, ..., up to {PARALLEL_NUM}. The <select> tags are required.
"""

FORCE_FINISH_PROMPTS = {
    "answer": [
        "Now combine all the information above and write the answer in <answer></answer> with \\boxed."
    ],
    "select": [
        "Now let me combine all the information above and select the best \"Response X\" (X is the selected response number, e.g., 1, 2, ...) in <select></select>.",
        "Select the best response and output ONLY the following XML tag strictly, with no additional text: <select>Response X</select>",
        "Choose the optimal response and output exclusively in the format: <select>Response X</select>",
        "Task: Select the best response. Output must be confined to: <select>Response X</select>"
    ]
}
