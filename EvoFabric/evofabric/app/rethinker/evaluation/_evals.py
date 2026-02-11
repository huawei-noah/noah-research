# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import copy
import json
import os
import re
import traceback
from abc import abstractmethod
from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

import httpx
from openai import BaseModel, OpenAI
from pydantic import ConfigDict, Field, PrivateAttr
from tqdm import tqdm

from evofabric.logger import get_logger
from ._prompts import JUDGE_PROMPT_HLE, JUDGE_PROMPT_GAIA, JUDGE_PROMPT_XBENCH
from .._config import LLMConfig

logger = get_logger()

RESULT_FILE_NAME = "result.json"


class BaseBenchmarkEvaluator(BaseModel):
    """
    Base class for benchmark evaluation.

    This class provides a common evaluation pipeline for benchmark datasets,
    including result matching, parallel evaluation, error handling, and
    result aggregation. Subclasses must implement the `evaluate_item` method.

    Attributes:
        data_file (Path): Path to the source dataset file in JSON format.
        result_root (Path): Root directory containing model generation results.
        eval_llm (LLMConfig): Configuration for the LLM used as the evaluator.
        max_workers (int): Maximum number of threads for parallel evaluation.
        max_char (int): Maximum number of characters to keep from model responses.
        max_completion_tokens (int): Maximum tokens allowed for judge model output.
        save_path (Path): File path to save evaluation results.
        result_extractor (Optional[Callable]): Optional custom function to extract
            model predictions from result files.
    """
    data_file: Path = Field(..., description="Path to the source data file (json).")
    result_root: Path = Field(..., description="Root directory containing generation results.")
    eval_llm: LLMConfig = Field(..., description="Configuration for the LLM used as a judge.")

    max_workers: int = Field(default=32, ge=1, description="Max threads for evaluation.")
    max_char: int = Field(default=200, description="Max characters to clip from response for the prompt.")
    max_completion_tokens: int = Field(default=4096, description="Max tokens for judge response.")
    save_path: Path = Field(default=Path("eval_result.json"), description="Path to save the evaluation results.")

    # Optional custom extractor. If None, default logic is used.
    # Excluded from serialization because functions cannot be easily serialized.
    result_extractor: Optional[Callable[[str], Optional[str]]] = Field(default=None, exclude=True)

    _client: OpenAI = PrivateAttr()

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self._client = self._create_openai_client()

        # Ensure directories exist (optional validation)
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        if not self.result_root.exists():
            logger.warning(f"Result root does not exist: {self.result_root}")

        if not self.result_extractor:
            self.result_extractor = self._default_result_extractor

    def _create_openai_client(self) -> OpenAI:
        """Initializes the OpenAI client based on LLMConfig."""
        client_kwargs = self.eval_llm.create_client_kwargs()
        http_client_kwargs = getattr(self.eval_llm, "http_client_kwargs", {})

        if self.eval_llm.csb_token:
            client_kwargs.update({"default_headers": {"csb-token": self.eval_llm.csb_token}})

        return OpenAI(
            http_client=httpx.Client(**http_client_kwargs),
            **client_kwargs
        )

    def _load_json(self, file: Path) -> Any:
        """Load json file."""
        with open(file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _clip_str(self, string: str) -> str:
        """Truncates string if it exceeds the threshold."""
        if len(string) > self.max_char:
            return string[self.max_char:]
        return string

    def _default_result_extractor(self, q_root: str) -> Optional[str]:
        """Default logic to extract the selected response from result.json."""
        q_root_path = Path(q_root)
        result_file = q_root_path / RESULT_FILE_NAME
        if not result_file.exists():
            return None

        try:
            result = self._load_json(result_file)
            # Assuming 'selector' is the key, similar to original code
            selection_key = 'selector'
            if selection_key not in result:
                return None

            final_answer_list = result[selection_key].get("response_list", [])
            select_index = result[selection_key].get("selected_response")

            if not final_answer_list:
                return None

            if not select_index:
                idx = 0
            else:
                idx = int(select_index) - 1

            # Boundary check
            if 0 <= idx < len(final_answer_list):
                return final_answer_list[idx]
            return final_answer_list[0]

        except Exception as e:
            logger.warning(f"Error extracting result from {q_root}: {e}")
            return None

    def _match_data_and_result(self) -> List[Dict]:
        """Matches source data with generated results based on ID."""
        data_list = self._load_json(self.data_file)
        data_id_map = {data["id"]: data for data in data_list}
        mapped_data = []

        if not self.result_root.exists():
            logger.error(f"Result root directory does not exist: {self.result_root}")
            return []

        qid_lists = os.listdir(self.result_root)

        for qid in qid_lists:
            _q_root = self.result_root / qid
            if not _q_root.exists() or not _q_root.is_dir():
                continue

            answers = self.result_extractor(str(_q_root))

            # Normalize to list
            if answers is None:
                answers = []
            elif isinstance(answers, str):
                answers = [answers]

            if qid in data_id_map:
                for answer in answers:
                    data = copy.deepcopy(data_id_map[qid])
                    data["prediction"] = answer
                    mapped_data.append(data)

        return mapped_data

    @abstractmethod
    def evaluate_item(self, data: Dict) -> Dict:
        """
        Evaluate a single data item.

        Subclasses must implement this method to define benchmark-specific
        evaluation logic.

        Args:
            data (Dict): A single data item containing question, answer,
                and model prediction.

        Returns:
            Dict: Evaluation result including score and metadata.
        """
        pass

    def _safe_evaluate_item(self, data: Dict) -> Dict:
        """Wrapper to handle exceptions during item evaluation."""
        try:
            return self.evaluate_item(data)
        except Exception as e:
            traceback.print_exc()
            return {
                "score": 0,
                "error": str(e),
                "details": None
            }

    def _summarize_and_save(self, results: List[Dict]):
        """Calculates statistics and saves results to file."""
        error_cnt = 0
        num = 0
        score_sum = 0

        for data in results:
            eval_res = data.get("eval_result", {})
            score = eval_res.get("score")
            error = eval_res.get("error")

            if error or score is None:
                error_cnt += 1
            else:
                score_sum += score
                num += 1

        avg_score = (score_sum / num) if num > 0 else 0.0
        logger.info(
            f"Evaluation Complete. Total: {len(results)}, "
            f"Valid: {num}, Avg Score: {avg_score:.4f}, Errors: {error_cnt},"
            f"Results will be saving to {self.save_path}"
        )

        with open(self.save_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(results, indent=4, ensure_ascii=False))

    def run(self):
        """
        Run the full benchmark evaluation pipeline.

        This method matches data with predictions, evaluates all items in
        parallel, and saves aggregated results.
        """
        mapped_data = self._match_data_and_result()
        unique_ids = set(x["id"] for x in mapped_data)

        logger.info(
            f"Starting evaluation for {self.__class__.__name__}. "
            f"Matched items: {len(mapped_data)}, Unique IDs: {len(unique_ids)}"
        )

        final_results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Map futures to original data items
            future_to_item = {
                executor.submit(self._safe_evaluate_item, data): data
                for data in mapped_data
            }

            pbar = tqdm(total=len(mapped_data), desc=f"Evaluating {self.__class__.__name__}")

            for future in as_completed(future_to_item):
                original_item = future_to_item[future]
                eval_result = future.result()
                merged_item = {**original_item, "eval_result": eval_result}
                final_results.append(merged_item)
                pbar.update(1)

            pbar.close()

        self._summarize_and_save(final_results)


class HLEEvaluator(BaseBenchmarkEvaluator):
    """Evaluator for the HLE benchmark."""

    def evaluate_item(self, data: Dict) -> Dict:
        class ExtractedAnswer(BaseModel):
            extracted_final_answer: str
            reasoning: str
            correct: Literal["yes", "no"]
            confidence: int
            strict: Literal[True]

        prompt = str(JUDGE_PROMPT_HLE).format(
            question=data["question"],
            correct_answer=data["answer"],
            response=self._clip_str(data["prediction"]),
        )

        response = self._client.beta.chat.completions.parse(
            model=self.eval_llm.model_name,
            max_completion_tokens=self.max_completion_tokens,
            messages=[{"role": "user", "content": prompt}],
            response_format=ExtractedAnswer,
        )

        content = response.choices[0].message.parsed
        # Handle case where parsed content might be None
        if not content:
            raise ValueError("Empty parsed response from model")

        return {
            "score": 1 if content.correct.lower() == "yes" else 0,
            "error": None,
            "details": {
                **content.model_dump(),
                "prompt": prompt
            }
        }


class XBenchEvaluator(BaseBenchmarkEvaluator):
    """Evaluator for the XBench benchmark."""

    def evaluate_item(self, data: Dict) -> Dict:
        prompt = str(JUDGE_PROMPT_XBENCH).format(
            question=data["question"],
            correct_answer=data["answer"],
            response=self._clip_str(data["prediction"]),
        )

        response = self._client.chat.completions.create(
            model=self.eval_llm.model_name,
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.choices[0].message.content or ""

        # Regex to find conclusion
        correct_match = re.search(r"结论:*.(正确|错误)", content)

        if not correct_match:
            raise ValueError("Failed to parse judge conclusion.")

        conclusion = correct_match.group(0)
        score = 1 if "正确" in conclusion else 0

        return {
            "score": score,
            "error": None,
            "details": {
                **response.model_dump(),
                "prompt": prompt,
                "conclusion_match": conclusion,
            }
        }


class GaiaEvaluator(BaseBenchmarkEvaluator):
    """Evaluator for the GAIA benchmark."""

    def evaluate_item(self, data: Dict) -> Dict:
        data["prediction"] = data["answer"]
        prompt = str(JUDGE_PROMPT_GAIA).format(
            question=data["question"],
            correct_answer=data["answer"],
            response=self._clip_str(data["prediction"]),
        )

        response = self._client.chat.completions.create(
            model=self.eval_llm.model_name,
            max_completion_tokens=self.max_completion_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.choices[0].message.content.strip().lower()

        content = content.split("</think>")[-1].strip().lower()
        score = 1 if content == "correct" else 0

        return {
            "score": score,
            "error": None,
            "details": {
                **response.model_dump(),
                "prompt": prompt,
            }
        }
