import requests
import json
import re
import concurrent.futures
from abc import ABC, abstractmethod
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score


class DataProcessor(ABC):
    def __init__(self, data_dir, max_threads=1):
        self.data_dir = data_dir
        self.max_threads = max_threads

    @abstractmethod
    def get_train_examples(self):
        pass

    @abstractmethod
    def get_test_examples(self):
        pass

    @abstractmethod
    def evaluate(self, predictor, test_exs):
        pass

    @abstractmethod
    def stringify_prediction(self, pred):
        pass


def process_example(ex, predictor, prompt):
    pred = predictor.inference(ex, prompt)
    return ex, pred


class ClassificationTask(DataProcessor):

    def run_evaluate(self, predictor, prompt, test_exs, n=100):
        labels = []
        preds = []
        texts = []
        exs = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [executor.submit(process_example, ex, predictor, prompt) for ex in test_exs[:n]]
            for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), total=len(futures),
                                  desc='running evaluate'):
                ex, pred = future.result()
                texts.append(ex['text'])
                labels.append(ex['label'])
                preds.append(pred)
                exs.append(ex)

        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='micro')
        return f1, texts, labels, preds, exs

    def evaluate(self, predictor, prompt, test_exs, n=100):
        while True:
            try:
                f1, texts, labels, preds, exs = self.run_evaluate(predictor, prompt, test_exs, n=n)
                break
            except (concurrent.futures.process.BrokenProcessPool, requests.exceptions.SSLError):
                pass
        return f1, texts, labels, preds, exs


class BinaryClassificationTask(ClassificationTask):
    categories = ['No', 'Yes']

    def stringify_prediction(self, pred):
        return BinaryClassificationTask.categories[pred]


class JailbreakBinaryTask(BinaryClassificationTask):
    categories = ['No', 'Yes']

    def get_train_examples(self):
        exs = []
        for i, l in enumerate(open(self.data_dir + '/train.tsv')):
            convo, label = l.strip().split('\t')
            label = int(label)
            text = ' '.join([x['text'].strip() for x in json.loads(convo) if x['role'] == 'user'])
            exs.append({'id': i, 'text': text, 'label': label})
        return exs

    def get_test_examples(self):
        exs = []
        for i, l in enumerate(open(self.data_dir + '/test.tsv')):
            convo, label = l.strip().split('\t')
            label = int(label)
            text = ' '.join([x['text'].strip() for x in json.loads(convo) if x['role'] == 'user'])
            exs.append({'id': i, 'text': text, 'label': label})
        return exs


class DefaultHFBinaryTask(BinaryClassificationTask):
    categories = ['No', 'Yes']

    def get_train_examples(self):
        exs = []
        for i, row in enumerate(open(self.data_dir + '/train.jsonl')):
            row = json.loads(row.strip())
            exs.append({'id': f'train-{i}', 'label': row['label'], 'text': row['text']})
        return exs

    def get_test_examples(self):
        exs = []
        for i, row in enumerate(open(self.data_dir + '/test.jsonl')):
            row = json.loads(row.strip())
            exs.append({'id': f'test-{i}', 'label': row['label'], 'text': row['text']})
        return exs


class BBHTask:
    def __init__(self, data_dir, max_threads=1):
        self.data_dir = data_dir
        self.max_threads = max_threads

    def get_train_examples(self):
        exs = []
        for i, row in enumerate(open(self.data_dir + '/train.jsonl')):
            row = json.loads(row.strip())
            # print(row)
            exs.append({'id': f'train-{i}', 'label': row['target'], 'text': row['input'],
                        'question_type': row['question_type']})
        return exs

    def get_test_examples(self):
        exs = []
        for i, row in enumerate(open(self.data_dir + '/test.jsonl')):
            row = json.loads(row.strip())

            exs.append({'id': f'test-{i}', 'label': row['target'], 'text': row['input'],
                        'question_type': row['question_type']})
        return exs

    def bbh_mcq_postprocess(self, text: str) -> str:
        ans = text
        ans_line = ans.split('answer is ')
        if len(ans_line) != 1:
            ans = ans_line[1].strip()
        match = re.search(r'\(([A-Z])\)*', ans)
        if match:
            return match.group(1)
        match = re.search(r'([A-Z])', ans)
        if match:
            return match.group(1)
        return ans

    def bbh_freeform_postprocess(self, text: str) -> str:
        ans = text
        ans_line = ans.split('answer is ')
        if len(ans_line) != 1:
            ans = ans_line[1].strip()
        ans = ans.split('\n')[0].strip()

        if ans.endswith('.'):
            ans = ans[:-1].strip()

        match = re.search(r'\*\*(.*?)\*\*', ans)
        if match:
            return match.group(1)

        return ans


class CfinBenchTask:
    def __init__(self, data_dir, max_threads=1):
        self.data_dir = data_dir
        self.max_threads = max_threads

    def get_train_examples(self):
        exs = []
        for i, row in enumerate(open(self.data_dir + '/train.jsonl')):
            row = json.loads(row.strip())
            exs.append({'id': f'train-{i}', 'label': row['Answer'], 'text': row['question'],
                        'question_type': row['question_type']})
        return exs

    def get_test_examples(self):
        exs = []
        for i, row in enumerate(open(self.data_dir + '/test.jsonl')):
            row = json.loads(row.strip())

            exs.append({'id': f'test-{i}', 'label': row['Answer'], 'text': row['question'],
                        'question_type': row['question_type']})
        return exs

    def cfinbench_postprocess(self, string: str, category: str):
        result = ""
        string = re.sub(r'[^\w\s]', '', string)
        if category == "multi_choice":
            answer = ''
            content = re.sub(r'\s+', '', string)
            match = re.search(r'([A-E]+)', content)
            if match:
                result = match.group(1)

        elif category == "single_choice":
            content = re.sub(r'\s+', '', string)
            for t in content:
                if t.isupper():
                    result = t
                    break

        elif category == "judgment":
            content = re.sub(r'\s+', '', string)
            match = re.search(r'(正确|错误)', content)
            if match:
                result = match.group(1)
        return result


class Gsm8kTask:
    def __init__(self, data_dir, max_threads=1):
        self.data_dir = data_dir
        self.max_threads = max_threads

    def get_train_examples(self):
        exs = []
        for i, row in enumerate(open(self.data_dir + '/train.jsonl')):
            row = json.loads(row.strip())
            exs.append({'id': f'train-{i}', 'label': row['answer'], 'text': row['question']})
        return exs

    def get_test_examples(self):
        exs = []
        for i, row in enumerate(open(self.data_dir + '/test.jsonl')):
            row = json.loads(row.strip())

            exs.append({'id': f'test-{i}', 'label': row['answer'], 'text': row['question']})
        return exs

    def gsm8k_postprocess(self, text: str) -> str:
        text = text.split('Question:')[0]
        numbers = re.findall(r'\-?\d+\.\d+|\-?\d+', text)
        if not numbers:
            return 'NULL'
        return numbers[-1]
