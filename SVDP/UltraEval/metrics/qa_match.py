from typing import Any


class QaMatch:
    def __init__(
        self,
    ):
        pass

    def __call__(self, doc, ground_truth, results) -> Any:
        """Take a single document and the LM input/output/ground_truth.
        Returns the  values of the metric for that one document
        """
        # return 1.0 if results[0].strip().startswith(ground_truth.strip()) else 0.0

        question = doc["question"] # 问题本身
        choices = doc["target_scores"] # 选项字典
        answer = [key for key, value in choices.items() if value ==1][0] # 答案字符串
        # ground_truth = "(C)" # 答案序号

        n = len(choices) # 选项数量
        input_string = results[0].strip() # 模型的输出
        # print(answer, input_string, ground_truth, n)

        # letters = ['({})'.format(chr(65 + i)) for i in range(n)]
        # print(letters)

        # print(find_first_letter(input_string, n))
        if ground_truth == self.find_first_letter(input_string, n):
            return 1.0

        if input_string.startswith(question.strip()):
            input_string = input_string[len(question.strip()):].strip()

        if input_string.startswith(answer.strip()):
            return 1.0

        return 0.0
    
    def find_first_letter(self, input_string, n): # 返回匹配的序号字符串，如果没有，则返回空字符串
        # 切分字符串
        splitted_strings = input_string.split('\n')[0].replace('\n', ' ').split(' ')
        
        # 如果字符串以.结尾，则去掉.
        splitted_strings = [s[:-1] if s.endswith('.') else s for s in splitted_strings]
        
        # 生成包含前n个字母的列表，每个字母用()包围
        letters = ['({})'.format(chr(65 + i)) for i in range(n)]
        
        # 检查每个切分后的字符串
        for s in splitted_strings:
            # 检查是否有包围字母
            if s in letters:
                return s
            if f"({s})" in letters:
                return f"({s})"
    
        return ""