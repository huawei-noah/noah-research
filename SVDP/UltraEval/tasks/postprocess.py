import re


class GeneralTorch:
    def __init__(self):
        pass

    def __call__(self, result, request):
        raw_outputs = []
        process_outputs = []

        if isinstance(result, list):  # 返回是列表，加了batch。已做截断
            raw_outputs = result
        elif isinstance(result, str):  # 返回是字符串，没有加batch。已做截断
            raw_outputs = [result]

        process_outputs = raw_outputs

        return raw_outputs, process_outputs


class GeneralTorchPPL:
    def __init__(self):
        pass

    def __call__(self, result, request):
        # result是三层的list
        process_outputs = [[-sum(inner) for inner in outer] for outer in result]
        return result, process_outputs


class GeneralTorchPPLNorm:
    def __init__(self):
        pass

    def __call__(self, result, request):
        process_outputs = [self.process_inner_lists(sublist) for sublist in result]
        return result, process_outputs

    def process_inner_lists(self, inner_lists):
        transposed = list(zip(*inner_lists))
        prefix = []
        for items in transposed:
            if all(item == items[0] for item in items):
                prefix.append(items[0])
            else:
                break
        processed_lists = [
            -(sum(lst[len(prefix):]) / len(lst[len(prefix):])) for lst in inner_lists
        ]
        return processed_lists


class BBHPost:
    def __init__(self):
        pass

    def __call__(self, raw_outputs, processed_outputs):
        if isinstance(processed_outputs, str):
            processed_outputs = [processed_outputs]
        processed_outputs_ = [self.postprocess(text) for text in processed_outputs]
        return raw_outputs, processed_outputs_

    def postprocess(self, text: str) -> str:
        text = text.strip().split("\n\n")[0].strip()
        if "So the answer is" in text:
            text = text.split("So the answer is")[-1].strip().rstrip('.').strip()
        if "answer is" in text:
            text = text.split("answer is")[-1].strip()
        if "答案是" in text:
            text = text.split("答案是")[-1].strip()

        return text

class ExactMatchPost:
    def __init__(self):
        pass

    def __call__(self, raw_outputs, processed_outputs):
        if isinstance(processed_outputs, str):
            processed_outputs = [processed_outputs]
        processed_outputs_ = [self.postprocess(text) for text in processed_outputs]
        return raw_outputs, processed_outputs_

    def postprocess(self, text: str) -> str:
        text = re.split(r"[\n]", text, 1)[0].lower()
        if "answer is" in text:
            text = text.split("answer is")[-1].strip()
        if "答案是" in text:
            text = text.split("答案是")[-1].strip()
        text = general_postprocess(text)
        return text


class ArithmeticPost:
    def __init__(self):
        pass

    def __call__(self, raw_outputs, processed_outputs):
        if isinstance(processed_outputs, str):
            processed_outputs = [processed_outputs]
        processed_outputs_ = [self.postprocess(text) for text in processed_outputs]
        return raw_outputs, processed_outputs_

    def postprocess(self, text):
        text = text.strip().split("=")[-1].strip()
        return text


class UntilReturnPost:
    def __init__(self):
        pass

    def __call__(self, raw_outputs, processed_outputs):
        processed_outputs_ = []

        if isinstance(processed_outputs, str):
            processed_outputs = [processed_outputs]

        for output in processed_outputs:
            output = output.strip().split("\n")[0].strip()
            processed_outputs_.append(output)

        return raw_outputs, processed_outputs_


class MbppPost:
    def __init__(self):
        pass

    def __call__(self, raw_outputs, processed_outputs):
        processed_outputs_ = []

        if isinstance(processed_outputs, str):
            processed_outputs = [processed_outputs]

        for output in processed_outputs:
            output = output.split("\n\n")[0]
            processed_outputs_.append(output)

        return raw_outputs, processed_outputs_


class HumanEvalPost:
    def __init__(self):
        pass

    def __call__(self, raw_outputs, processed_outputs):
        processed_outputs_ = []

        if isinstance(processed_outputs, str):
            processed_outputs = [processed_outputs]

        split_keywords = [
            "\nclass",
            "\ndef",
            "\n#",
            "\n@",
            "\nprint",
            "\nif",
            "<|endoftext|>",
        ]

        for output in processed_outputs:
            for keyword in split_keywords:
                output = output.split(keyword)[0]
            processed_outputs_.append(output)

        return raw_outputs, processed_outputs_


class CommonMathPost:
    def __init__(self):
        pass

    def __call__(self, raw_outputs, processed_outputs):
        if isinstance(processed_outputs, str):
            processed_outputs = [processed_outputs]
        processed_outputs_ = []

        for text in processed_outputs:
            try:
                stripped_text = _strip_string(text)
            except:
                stripped_text = text
            processed_outputs_.append(stripped_text)

        return raw_outputs, processed_outputs_


class MathPost:
    SUBSTITUTIONS = [
        ("an ", ""),
        ("a ", ""),
        (".$", "$"),
        ("\\$", ""),
        (r"\ ", ""),
        (" ", ""),
        ("mbox", "text"),
        (",\\text{and}", ","),
        ("\\text{and}", ","),
        ("\\text{m}", "\\text{}"),
        ("\\le", "<"),
    ]
    REMOVED_EXPRESSIONS = [
        "square",
        "ways",
        "integers",
        "dollars",
        "mph",
        "inches",
        "ft",
        "hours",
        "km",
        "units",
        "\\ldots",
        "sue",
        "points",
        "feet",
        "minutes",
        "digits",
        "cents",
        "degrees",
        "cm",
        "gm",
        "pounds",
        "meters",
        "meals",
        "edges",
        "students",
        "childrentickets",
        "multiples",
        "\\text{s}",
        "\\text{.}",
        "\\text{\ns}",
        "\\text{}^2",
        "\\text{}^3",
        "\\text{\n}",
        "\\text{}",
        r"\mathrm{th}",
        r"^\circ",
        r"^{\circ}",
        r"\;",
        r",\!",
        "{,}",
        '"',
        "\\dots",
        "\n",
        "\r",
        "\f",
    ]

    def __init__(self):
        pass

    def __call__(self, raw_outputs, processed_outputs):
        if isinstance(processed_outputs, str):
            processed_outputs = [processed_outputs]

        processed_outputs_ = []

        for text in processed_outputs:
            try:
                processed_text = _strip_string(self.math_postprocess(text))
            except Exception as e:
                try:
                    processed_text = self.math_postprocess(text)
                except Exception as e:
                    processed_text = text
            processed_outputs_.append(processed_text)

        return raw_outputs, processed_outputs_

    def normalize_final_answer(self, final_answer: str) -> str:
        """Normalize a final answer to a quantitative reasoning question."""
        # final_answer = final_answer.split('=')[-1]
        for before, after in self.SUBSTITUTIONS:
            final_answer = final_answer.replace(before, after)
        for expr in self.REMOVED_EXPRESSIONS:
            final_answer = final_answer.replace(expr, "")

        # Extract answer that is in LaTeX math, is bold,
        # is surrounded by a box, etc.
        final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
        final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
        final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
        final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
        assert "\n" not in final_answer
        assert "\r" not in final_answer
        assert "\f" not in final_answer
        if len(re.findall(r"finalansweris(.*)", final_answer)) > 0:
            final_answer = re.findall(r"finalansweris(.*)", final_answer)[-1]

        if len(re.findall(r"oxed\{(.*?)\}", final_answer)) > 0:
            final_answer = re.findall(r"oxed\{(.*?)\}", final_answer)[-1]

        if len(re.findall(r"\$(.*?)\$", final_answer)) > 0:
            final_answer = re.findall(r"\$(.*?)\$", final_answer)[-1]
        final_answer = final_answer.strip()
        if "rac" in final_answer and "\\frac" not in final_answer:
            final_answer = final_answer.replace("rac", "\\frac")

        # Normalize shorthand TeX:
        # \fracab -> \frac{a}{b}
        # \frac{abc}{bef} -> \frac{abc}{bef}
        # \fracabc -> \frac{a}{b}c
        # \sqrta -> \sqrt{a}
        # \sqrtab -> sqrt{a}b
        final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
        final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
        final_answer = final_answer.replace("$", "")

        # Normalize 100,000 -> 100000
        if final_answer.replace(",", "").isdigit():
            final_answer = final_answer.replace(",", "")

        return final_answer

    def math_postprocess(self, text: str) -> str:
        for line in reversed(text.split("\n")):
            for maybe_ans in line.split('.'):
                if ('final answer' in maybe_ans.lower() or 'therefore' in maybe_ans.lower() or
                        'thus' in maybe_ans.lower() or 'hence' in maybe_ans.lower()):
                    return self.normalize_final_answer(maybe_ans)
        return self.normalize_final_answer(text.split('.')[0])


class TheoremQAPost:
    def __init__(self):
        pass

    def __call__(self, raw_outputs, processed_outputs):
        if isinstance(processed_outputs, str):
            processed_outputs = [processed_outputs]
        processed_outputs_ = [self.postprocess(text) for text in processed_outputs]
        return raw_outputs, processed_outputs_

    def postprocess(self, text):
        text = text.strip()
        matches = re.findall(r"answer is ([^\s]+)", text)
        if len(matches) == 0:
            return text
        else:
            text = matches[0].strip().strip(".,?!\"';:")
            return text


class GSM8KPost:
    def __init__(self):
        pass

    def __call__(self, raw_outputs, processed_outputs):
        if isinstance(processed_outputs, str):
            processed_outputs = [processed_outputs]
        processed_outputs_ = [self.postprocess(text) for text in processed_outputs]
        return raw_outputs, processed_outputs_

    def postprocess(self, text):
        ans_re = re.compile(r"#### (\-?[0-9\.\,]+)")
        m = ans_re.search(text)
        if m is not None:
            text = m.group(1).strip().replace(",", "")
        else:
            if ">>" in text and text[text.rfind(">>") + 2:].strip().split():
                text = text[text.rfind(">>") + 2:].strip().split()[0]
                text = text.strip("$")
                text = text.replace(",", '')
                text = text.rstrip(".")
            elif "=" in text and text[text.rfind("=") + 1:].strip().split():
                text = text[text.rfind("=") + 1:].strip().split()[0]
                text = text.strip("$")
                text = text.replace(",", '')
                text = text.rstrip(".")
            else:
                for text in reversed(text.split()):
                    if text.isdigit():
                        return text

        return text.strip()


class HumanEvalGPT:
    def __init__(self):
        pass

    def __call__(self, raw_outputs, processed_outputs):
        if isinstance(processed_outputs, str):
            processed_outputs = [processed_outputs]

        processed_outputs_ = [self.humaneval_gpt_postprocess(text) for text in processed_outputs]
        return raw_outputs, processed_outputs_

    @staticmethod
    def humaneval_gpt_postprocess(text: str) -> str:
        """Better answer postprocessor for better instruction-aligned models like
        GPT."""
        if '```' in text:
            blocks = re.findall(r'```(.*?)```', text, re.DOTALL)
            if len(blocks) == 0:
                text = text.split('```')[1]  # fall back to default strategy
            else:
                text = blocks[0]  # fetch the first code block
                if not text.startswith('\n'):  # in case starting with ```python
                    text = text[max(text.find('\n') + 1, 0):]
        if text.strip().startswith('from') or text.strip().startswith('import'):
            def_idx = text.find('def')
            if def_idx != -1:
                text = text[max(text.find('\n', def_idx) + 1, 0):]
        text = text.split('\n\n\n')[0]
        if text.strip().startswith('def'):
            text = '\n'.join(text.split('\n')[1:])
        if not text.startswith('    '):
            if text.startswith(' '):
                text = '    ' + text.lstrip()
            else:
                text = '\n'.join(['    ' + line for line in text.split('\n')])
        return text


class HellaSwagPost:
    def __init__(self):
        pass

    def __call__(self, raw_outputs, processed_outputs):
        if isinstance(processed_outputs, str):
            processed_outputs = [processed_outputs]

        processed_outputs_ = [self.process(text) for text in processed_outputs]

        # print("processed_outputs:", processed_outputs)
        # print("processed_outputs_:", processed_outputs_)
        return raw_outputs, processed_outputs_

    @staticmethod
    def process(text):
        # if len(text) >= 2 and text[1] in string.ascii_letters:
        #     m = re.compile("[Oo]ption [ABCD]").search(text)
        #     if m:
        #         text = text[m.span()[1] - 1:]
        #         return text
        for idx, c in enumerate(text):
            if c in ['A', 'B', 'C', 'D']:
                return text[idx:]

        return text


class GaoKaoSingleChoicePost:
    def __init__(self):
        pass

    def __call__(self, raw_outputs, processed_outputs):
        if isinstance(processed_outputs, str):
            processed_outputs = [processed_outputs]
        processed_outputs_ = [self.postprocess(text) for text in processed_outputs]
        return raw_outputs, processed_outputs_

    def postprocess(self, text):
        choice = re.findall(r"[A-D]", text[::-1])
        if choice:
            return choice[0]
        else:
            return ""


class GaoKaoMultiQuestionChoicePost:
    def __init__(self):
        pass

    def __call__(self, raw_outputs, processed_outputs):
        if isinstance(processed_outputs, str):
            processed_outputs = [processed_outputs]
        processed_outputs_ = [self.postprocess(text) for text in processed_outputs]
        return raw_outputs, processed_outputs_

    def postprocess(self, text):
        model_answer1 = []
        model_answer2 = []
        match1 = re.findall(r"【答案】\s*[:：]*\s*[A-Z]", text)
        for t in match1:
            model_answer1.append(re.findall(r"[A-Z]", t)[0])
        text1 = "".join(model_answer1)

        match2 = re.findall(r"[A-Z]", text)
        if len(match2) > 0:
            for k in range(len(match2)):
                model_answer2.append(match2[k])
        text2 = "".join(model_answer2)
        return text1 + "[SEP]" + text2


class GaoKaoMultiChoicePost:
    def __init__(self):
        pass

    def __call__(self, raw_outputs, processed_outputs):
        if isinstance(processed_outputs, str):
            processed_outputs = [processed_outputs]
        processed_outputs_ = [self.postprocess(text) for text in processed_outputs]
        return raw_outputs, processed_outputs_

    def postprocess(self, text):
        model_answer = []
        answer = ""
        content = re.sub(r"\s+", "", text)
        answer_index = content.find("【答案】")
        if answer_index > 0:
            temp = content[answer_index:]
            if len(re.findall(r"[A-D]", temp)) > 0:
                for t in re.findall(r"[A-D]", temp):
                    answer += t
        else:
            temp = content[-10:]
            if len(re.findall(r"[A-D]", temp)) > 0:
                for t in re.findall(r"[A-D]", temp):
                    answer += t
        if len(answer) != 0:
            model_answer.append(answer)
        text = "".join(model_answer)
        return text


class GaoKaoFiveOutOfSevenPost:
    def __init__(self):
        pass

    def __call__(self, raw_outputs, processed_outputs):
        if isinstance(processed_outputs, str):
            processed_outputs = [processed_outputs]
        processed_outputs_ = [self.postprocess(text) for text in processed_outputs]
        return raw_outputs, processed_outputs_

    def postprocess(self, text):
        model_answer = []
        temp = re.findall(r"[A-G]", text)
        if len(temp) > 0:
            for k in range(min(5, len(temp))):
                model_answer.append(temp[k])
        text = "".join(model_answer)
        return text


class AGIEvalClozePost:
    def __init__(self):
        pass

    def __call__(self, raw_outputs, processed_outputs):
        processed_outputs_ = []
        for output in processed_outputs:
            raw_string = self.remove_few_shot_prefix(output)
            if "\\boxed" in raw_string:
                answer = self.remove_boxed(self.last_boxed_only_string(raw_string))
            else:
                answer = self.get_answer_with_dollar_sign(raw_string)
                if not answer:
                    answer = self.get_answer_without_dollar_sign(raw_string)
            processed_outputs_.append(answer)

        cmp = CommonMathPost()
        _, processed_outputs_ = cmp([], processed_outputs_)

        return raw_outputs, processed_outputs_

    def remove_few_shot_prefix(self, string: str):
        prefix_list = ["The answer is therefore", "答案是"]
        for prefix in prefix_list:
            if string.startswith(prefix):
                string = string[len(prefix):].strip()
            elif prefix in string:
                index = string.rfind(prefix)
                if index >= 0:
                    string = string[index + len(prefix):].strip()
        return string

    def remove_boxed(self, s):
        left = "\\boxed{"
        try:
            assert s[: len(left)] == left
            assert s[-1] == "}"
            answer = s[len(left): -1]
            if "=" in answer:
                answer = answer.split("=")[-1].lstrip(" ")
            return answer
        except:
            return None

    def last_boxed_only_string(self, string):
        idx = string.rfind("\\boxed")
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return None
        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
            if string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx == None:
            retval = None
        else:
            retval = string[idx: right_brace_idx + 1]

        return retval

    def get_answer_with_dollar_sign(self, s):
        first_pattern = "\$(.*)\$"
        last_match = None
        matches = re.findall(first_pattern, s)
        if matches:
            last_match = matches[-1]
            if "=" in last_match:
                last_match = last_match.split("=")[-1].lstrip(" ")
        return last_match

    def get_answer_without_dollar_sign(self, s):
        last_match = None
        if "=" in s:
            last_match = s.split("=")[-1].lstrip(" ").rstrip(".")
            if "\\n" in last_match:
                last_match = last_match.split("\\n")[0]
        else:
            pattern = "(?:\\$)?\d+(?:\.\d+)?(?![\w\d])"
            matches = re.findall(pattern, s)
            if matches:
                last_match = matches[-1]
        return last_match


class AGIEvalMultipleAnswerPost:
    def __init__(self):
        pass

    def __call__(self, raw_outputs, processed_outputs):
        if isinstance(processed_outputs, str):
            processed_outputs = [processed_outputs]
        processed_outputs_ = [self.postprocess(text) for text in processed_outputs]
        return raw_outputs, processed_outputs_

    def postprocess(self, text):
        pattern = "\(*([A-F])\)*"
        match = re.findall(pattern, text)
        if match:
            return "".join(match)
        return ""


class AGIEvalSingleAnswerPost:
    def __init__(self):
        pass

    def __call__(self, raw_outputs, processed_outputs):
        if isinstance(processed_outputs, str):
            processed_outputs = [processed_outputs]
        processed_outputs_ = [self.postprocess(text) for text in processed_outputs]
        return raw_outputs, processed_outputs_

    def postprocess(self, text):
        pattern1 = "answer is .*?([A-G])"
        match = re.search(pattern1, text)
        if match:
            return match.group(1)
        pattern2 = "答案是.*?([A-G])"
        match = re.search(pattern2, text)
        if match:
            return match.group(1)
        return self.find_first_capital_letter(text)

    def find_first_capital_letter(self, answer):
        letter_set = {"A", "B", "C", "D", "E", "F"}
        for c in answer:
            if c in letter_set:
                return c
        return ""


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively,
    # add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works
    # with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix
    # in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def extract_code_from_string(s):
    code_start = s.find("```python") + len("```python")
    code_end = s.find("```", code_start)

    return s[code_start:code_end]


def cut(input, output):
    if output.strip().startswith(input.strip()):
        return output.strip()[len(input.strip()):]
    else:
        return output


def general_postprocess(text: str) -> str:
    # Cut off the first newline, period, or comma
    truncated_text = re.split(r"[\n.,]", text, 1)[0]
    # Remove punctuation
    no_punctuation = re.sub(r"[^\w\s]", "", truncated_text)
    # Remove article
    no_articles = re.sub(r"\b(a|an|the)\b", "", no_punctuation, flags=re.IGNORECASE)
    # Remove duplicated blank spaces
    cleaned_text = re.sub(r"\s+", " ", no_articles).strip()
    return cleaned_text

class NewMbppPost:
    def __call__(self, raw_outputs, processed_outputs):
        processed_outputs_ = []

        if isinstance(processed_outputs, str):
            processed_outputs = [processed_outputs]

        processed_outputs_ = [self.process_text(text) for text in processed_outputs]
        return raw_outputs, processed_outputs_
    
    @staticmethod
    def process_text(text):
        # 找到所有的三重双引号位置
        triple_quotes_indices = [i for i, _ in enumerate(text) if text.startswith('\"\"\"', i)]

        # 如果三重双引号的数量是偶数并且大于0
        if len(triple_quotes_indices) % 2 == 0 and len(triple_quotes_indices) > 0:
            # 逐对检查三重双引号之间的内容
            for i in range(0, len(triple_quotes_indices), 2):
                start_index = triple_quotes_indices[i]
                end_index = triple_quotes_indices[i + 1]
                # 检查是否包含'def'
                if 'def' in text[end_index:]:
                    # 删除三重双引号后的所有内容
                    return text[:start_index].strip()
            # 如果没有任何一对三重双引号之间的内容包含'def'，返回原字符串
            return text.strip()
        # 如果三重双引号的数量不是偶数
        elif len(triple_quotes_indices) > 0:
            # 删除第一个三重双引号之后的所有内容
            return text[:triple_quotes_indices[0]].strip()
        # 如果没有三重双引号，返回原字符串
        else:
            return text.strip()


POSTPROCESS_REGISTRY = {
    "general_torch": GeneralTorch,
    "general_torch_ppl": GeneralTorchPPL,
    "general_torch_ppl_norm": GeneralTorchPPLNorm,
    "exact_match_post": ExactMatchPost,
    "math_post": MathPost,
    "until_return_post": UntilReturnPost,
    "humaneval_post": HumanEvalPost,
    "arithmetic_post": ArithmeticPost,
    "theoremqa_post": TheoremQAPost,
    "gsm8k_post": GSM8KPost,
    "mbpp_post": MbppPost,
    "humaneval_chatgpt": HumanEvalGPT,
    "hellaswag_post": HellaSwagPost,
    "gaokao_single_choice_post": GaoKaoSingleChoicePost,
    "gaokao_multi_question_choice_post": GaoKaoMultiQuestionChoicePost,
    "gaokao_multi_choice_post": GaoKaoMultiChoicePost,
    "gaokao_five_out_of_seven_post": GaoKaoFiveOutOfSevenPost,
    "agieval_cloze_post": AGIEvalClozePost,
    "agieval_single_answer_post": AGIEvalSingleAnswerPost,
    "agieval_multiple_answer_post": AGIEvalMultipleAnswerPost,
    "common_math_post": CommonMathPost,
    "new_mbpp_post": NewMbppPost,
    "bbh_post": BBHPost
}


def get_postprocess(postprocess_name):
    return POSTPROCESS_REGISTRY[postprocess_name]


if __name__ == "__main__":
    text = HellaSwagPost.process(
        "Based on the given information, the most logical ending would be Option C: they sit in a canoe "
        "while the man paddles.")
    print(text)