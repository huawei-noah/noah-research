# evaluation/search.py
"""Search and generate answers for LoCoMo evaluation questions"""

from __future__ import annotations

import argparse
import json
import time
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import threading
from jinja2 import Template
import re
import sys
from pathlib import Path
import numpy as np

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI

from src.memory.memory_manager import MemoryManager
from src.models.message import Message
from src.retrieval.multi_stage_retriever import MultiStageRetriever
from src.retrieval.query_tag_router import QueryTagRouter
from src.retrieval.types import RetrievalConfig

load_dotenv()

logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)



def create_conversation_id(speaker_a: str, speaker_b: str, idx: int) -> str:
    """创建 conversation ID"""
    speakers = sorted([speaker_a, speaker_b])
    return f"conv_{idx}_{speakers[0]}_{speakers[1]}"


def get_index_path(conversation_id: str, base_path: str = "data/indices") -> str:
    """获取 conversation 的索引路径"""
    parts = conversation_id.split("_", 2)
    speaker_pair = parts[2]
    return f"{base_path}/locomo_{speaker_pair}"

def get_database_paths(conversation_id: str, base_path: str = "data/conversations") -> Tuple[str, str]:
    """
    获取 conversation 的数据库路径
    
    Args:
        conversation_id: Conversation ID (如 "conv_0_Caroline_Melanie")
        base_path: 数据库基础路径
        
    Returns:
        (episode_db_path, semantic_db_path)
    """
    conv_db_dir = Path(f"{base_path}/{conversation_id}")
    return (
        str(conv_db_dir / "episodes.db"),
        str(conv_db_dir / "semantic.db")
    )


class IndexPool:
    """索引池：预加载所有索引，避免重复 I/O"""
    
    def __init__(self, dataset: List[Dict[str, Any]], config: Dict[str, Any]):
        """
        初始化索引池
        
        Args:
            dataset: 数据集
            config: 配置
        """
        self.managers: Dict[str, MemoryManager] = {}
        self.retrievers: Dict[str, MultiStageRetriever] = {}
        self.tag_routers: Dict[str, QueryTagRouter] = {}
        self.config = config
        
        print("\n🔄 Initializing Index Pool...")
        print("📂 Loading MemoryManagers and TagRouters...")
        self._load_all_indices(dataset)
        print(f"✅ Index Pool ready: {len(self.managers)} conversations loaded")
        print(f"✅ TagRouters ready: {len(self.tag_routers)} routers initialized\n")
    
    def _load_all_indices(self, dataset: List[Dict[str, Any]]) -> None:
        """预加载所有 conversation 的索引"""
        
        for idx, item in enumerate(tqdm(dataset, desc="Loading indices", ncols=80)):
            conversation = item.get("conversation", {})
            speaker_a = conversation.get("speaker_a", "")
            speaker_b = conversation.get("speaker_b", "")
            
            if not speaker_a or not speaker_b:
                continue
            
            # 创建 conversation_id
            conversation_id = create_conversation_id(speaker_a, speaker_b, idx)
            index_path = get_index_path(conversation_id)
            
            # 检查索引是否存在
            if not Path(index_path).exists():
                logger.warning(f"⚠️  Index not found: {index_path}")
                continue
            
            try:
                #获取该 conversation 的独立数据库路径
                episode_db_path, semantic_db_path = get_database_paths(conversation_id)
                
                # 创建 MemoryManager
                manager = MemoryManager(
                    episode_db_path=episode_db_path,
                    semantic_db_path=semantic_db_path,
                    openai_api_key=os.getenv("EMBEDDING_API_KEY"),
                    embedding_model=os.getenv("EMBEDDING_MODEL"),
                    embedding_cache_path=self.config["embedding_cache_path"],
                    embedding_dim=int(os.getenv("EMBEDDING_DIM", "1536")),
                    openai_base_url=os.getenv("EMBEDDING_BASE_URL")
                )
                
                # 加载索引
                manager.load_index(index_path)
                
                # 保存到池
                self.managers[conversation_id] = manager
                
                #立即创建 TagRouter
                from src.retrieval.query_tag_router import QueryTagRouter
                
                tag_router = QueryTagRouter(
                    tag_dag_index=manager.index.tag_dag_index,
                    embedding_manager=manager.embedding_manager
                )
                
                #缓存 TagRouter
                self.tag_routers[conversation_id] = tag_router
                
            except Exception as e:
                #TagRouter 创建失败时终止加载
                logger.error(f"Failed to load index or create TagRouter for {conversation_id}: {e}")
                raise RuntimeError(f"Critical error: Cannot initialize {conversation_id}") from e

        #预生成所有 tag embeddings
        print("📌 Pre-generating tag embeddings...")
        self._pregenerate_tag_embeddings()

    def _pregenerate_tag_embeddings(self) -> None:
        """预生成所有 tag embeddings（批量优化）"""
        from tqdm import tqdm
        
        total_tags = 0
        generated_tags = 0
        
        for conversation_id, manager in self.managers.items():
            tag_dag = manager.index.tag_dag_index
            print(f"{conversation_id}: {len(tag_dag.tag_nodes)} tags")
            
            # 收集需要生成 embedding 的 tags
            tags_to_generate = []
            tag_nodes_to_update = []
            
            for tag, tag_node in tag_dag.tag_nodes.items():
                if tag_node.embedding is None:
                    tags_to_generate.append(tag)
                    tag_nodes_to_update.append(tag_node)
            
            total_tags += len(tag_dag.tag_nodes)
            
            if not tags_to_generate:
                continue
            
            #批量生成 embeddings
            try:
                print(f"  🔄 Generating embeddings for {len(tags_to_generate)} tags...")
                
                embeddings = manager.embedding_manager.get_embeddings_batch(
                    texts=tags_to_generate,
                    use_cache=True
                )
                
                #验证返回类型和 shape
                print(f"  📊 Embeddings type: {type(embeddings)}")
                print(f"  📊 Embeddings shape: {embeddings.shape}")
                print(f"  📊 Expected: ({len(tags_to_generate)}, {manager.embedding_manager.dimension})")
                
                if not isinstance(embeddings, np.ndarray):
                    raise TypeError(f"Expected np.ndarray, got {type(embeddings)}")
                
                if embeddings.shape[0] != len(tags_to_generate):
                    raise ValueError(
                        f"Shape mismatch: {embeddings.shape[0]} embeddings "
                        f"for {len(tags_to_generate)} tags"
                    )
                
                # 更新 tag_node.embedding
                print(f"  💾 Saving embeddings to tag_nodes...")
                for i, (tag_node, embedding) in enumerate(zip(tag_nodes_to_update, embeddings)):
                    #embedding 是 np.ndarray 的一行
                    tag_node.embedding = embedding.tolist()
                    
                    #验证前 3 个 tag
                    if i < 3:
                        print(f"    Tag '{tag_node.tag}': embedding type = {type(tag_node.embedding)}, "
                            f"len = {len(tag_node.embedding) if tag_node.embedding else 'None'}")
                
                #验证保存是否成功
                print(f"  🔍 Verifying saved embeddings...")
                failed = []
                for tag_node in tag_nodes_to_update:
                    if tag_node.embedding is None:
                        failed.append(tag_node.tag)
                
                if failed:
                    raise RuntimeError(
                        f"Failed to save embeddings for {len(failed)} tags: {failed[:5]}"
                    )
                
                print(f" All {len(tags_to_generate)} embeddings saved successfully")
                
                generated_tags += len(tags_to_generate)
                
            except Exception as e:
                logger.error(f"Failed to generate tag embeddings for {conversation_id}: {e}")
                #不允许失败，直接抛出
                raise RuntimeError(f"Critical error: Cannot generate tag embeddings for {conversation_id}") from e
        
        print(f"✅ Tag embeddings ready: {total_tags} total tags, {generated_tags} newly generated")

        #为所有 TagRouter 构建 embeddings cache
        print("🔨 Building tag embeddings cache for all routers...")
        
        #在构建 cache 前再次验证
        print("🔬 Pre-cache verification:")
        for conversation_id, manager in self.managers.items():
            tag_dag = manager.index.tag_dag_index
            total = len(tag_dag.tag_nodes)
            missing = [tag for tag, node in tag_dag.tag_nodes.items() if node.embedding is None]
            
            if missing:
                print(f"  ❌ {conversation_id}: {len(missing)}/{total} tags still have None embeddings!")
                print(f"     Missing: {missing[:5]}")
            else:
                print(f" {conversation_id}: All {total} tags have embeddings")
        
        # 构建 cache
        for conversation_id, tag_router in self.tag_routers.items():
            tag_router._rebuild_tag_embeddings_cache()
        
        print("✅ Tag embeddings cache ready")

    def get_manager(self, conversation_id: str) -> Optional[MemoryManager]:
        """获取已加载的 MemoryManager"""
        return self.managers.get(conversation_id)
    
    def get_retriever(self, conversation_id: str) -> Optional['MultiStageRetriever']:
        """获取 Retriever（懒加载 + 缓存）"""
        # 检查缓存
        if conversation_id in self.retrievers:
            return self.retrievers[conversation_id]
        
        # 获取 manager
        manager = self.managers.get(conversation_id)
        if not manager:
            return None
        
        # 创建 retriever
        from src.retrieval.multi_stage_retriever import MultiStageRetriever
        from src.retrieval.types import RetrievalConfig
        
        retriever = MultiStageRetriever(
            temporal_index=manager.index.temporal_index,
            tag_dag_index=manager.index.tag_dag_index,
            vector_index=manager.index.vector_index,
            episode_storage=manager.episode_storage,
            semantic_storage=manager.semantic_storage,
            embedding_manager=manager.embedding_manager,
            unified_index=manager.index
        )
        
        # 配置 retriever
        retrieval_config = self.config.get("retrieval", {})
        retriever.config = RetrievalConfig(
            use_temporal_filter=retrieval_config.get("use_temporal_filter", True),
            max_temporal_candidates=retrieval_config.get("max_temporal_candidates", 1000),
            use_semantic_routing=retrieval_config.get("use_semantic_routing", True),
            max_tags_per_query=retrieval_config.get("max_tags_per_query", 10),
            tag_expansion_depth=retrieval_config.get("tag_expansion_depth", 1),
            max_expanded_tags=retrieval_config.get("max_expanded_tags", 10),
            stage2_max_candidates=retrieval_config.get("stage2_max_candidates", 100),
            vector_top_k=retrieval_config.get("vector_top_k", 50),
            vector_similarity_threshold=retrieval_config.get("vector_similarity_threshold", 0.1),
            vector_dynamic_topk=retrieval_config.get("vector_dynamic_topk", True),
            vector_top_k_multiplier=retrieval_config.get("vector_top_k_multiplier", 1.5),
            use_reranking=retrieval_config.get("use_reranking", True),
            rerank_top_k=retrieval_config.get("rerank_top_k", 50),
            bm25_weight=retrieval_config.get("bm25_weight", 0.35),
            vector_weight=retrieval_config.get("vector_weight", 0.50),
            tag_weight=retrieval_config.get("tag_weight", 0.15),
            stage3_min_candidates=retrieval_config.get("stage3_min_candidates", 20),
            stage3_tag_coverage_threshold=retrieval_config.get("stage3_tag_coverage_threshold", 0.3),
            vector_supplement_candidates=retrieval_config.get("vector_supplement_candidates", True),
            vector_min_candidates=retrieval_config.get("vector_min_candidates", 30)
        )
        retriever.reranker.config = retriever.config
        
        # 缓存
        self.retrievers[conversation_id] = retriever
        return retriever
    
    def get_tag_router(self, conversation_id: str) -> Optional['QueryTagRouter']:
        """获取 TagRouter（预加载，无懒加载）"""
        return self.tag_routers.get(conversation_id)
    
    def close_all(self) -> None:
        """清理所有资源"""
        print("\n🔄 Closing Index Pool...")
        for conversation_id, manager in self.managers.items():
            try:
                manager.close()
            except Exception as e:
                logger.error(f"Failed to close {conversation_id}: {e}")
        print("✅ Index Pool closed\n")


# ==========人名提取函数 ==========

def extract_speaker_names(question: str, dataset: List[Dict[str, Any]]) -> Set[str]:
    """
    从问题中提取提到的人名
    
    策略:
    1. 收集数据集中所有可能的人名
    2. 在问题中查找这些人名（完整单词匹配 + 大小写不敏感）
    
    Args:
        question: 问题文本
        dataset: 数据集
        
    Returns:
        问题中提到的人名集合
        
    Examples:
        question = "What did John say about India?"
        返回: {"John"}
        
        question = "Did Caroline and Melanie discuss travel?"
        返回: {"Caroline", "Melanie"}
    """
    # 1. 收集所有人名
    all_names = set()
    for item in dataset:
        conversation = item.get("conversation", {})
        speaker_a = conversation.get("speaker_a", "")
        speaker_b = conversation.get("speaker_b", "")
        if speaker_a:
            all_names.add(speaker_a)
        if speaker_b:
            all_names.add(speaker_b)
    
    # 2. 在问题中查找（完整单词匹配）
    question_lower = question.lower()
    found_names = set()
    
    for name in all_names:
        if name:
            #使用正则表达式确保完整单词匹配
            pattern = r'\b' + re.escape(name.lower()) + r'\b'
            if re.search(pattern, question_lower):
                found_names.add(name)
    
    return found_names


# ==========索引匹配函数 ==========

def match_indices_by_speakers(
    speaker_names: Set[str],
    dataset: List[Dict[str, Any]],
    base_path: str = "data/indices"
) -> List[Tuple[str, str]]:
    """
    根据人名匹配对应的索引路径
    
    Args:
        speaker_names: 提取的人名集合
        dataset: 数据集
        base_path: 索引基础路径
        
    Returns:
        [(conversation_id, index_path), ...]
        
    Examples:
        speaker_names = {"John"}
        返回:
        [
            ("conv_2_John_Maria", "data/indices/locomo_John_Maria"),
            ("conv_4_John_Tim", "data/indices/locomo_John_Tim"),
            ("conv_6_James_John", "data/indices/locomo_James_John")
        ]
    """
    matched = []
    
    for idx, item in enumerate(dataset):
        conversation = item.get("conversation", {})
        speaker_a = conversation.get("speaker_a", "")
        speaker_b = conversation.get("speaker_b", "")
        
        # 检查是否有任何一个人名在这个 conversation 中
        if speaker_a in speaker_names or speaker_b in speaker_names:
            conversation_id = create_conversation_id(speaker_a, speaker_b, idx)
            index_path = get_index_path(conversation_id, base_path)
            matched.append((conversation_id, index_path))
    
    return matched


# ==========从 QA 归属推断 conversation ==========

def infer_conversation_from_qa(
    qa_item: Dict[str, Any],
    dataset: List[Dict[str, Any]]
) -> Optional[Tuple[str, str]]:
    """
    从 QA 项推断其所属的 conversation
    
    Args:
        qa_item: QA 数据项
        dataset: 数据集
        
    Returns:
        (conversation_id, index_path) 或 None
    """
    question = qa_item.get("question", "")
    
    for idx, item in enumerate(dataset):
        conversation = item.get("conversation", {})
        qa_list = item.get("qa", [])
        
        # 检查这个 qa_item 是否在这个 conversation 中
        for qa in qa_list:
            if qa.get("question") == question:
                speaker_a = conversation.get("speaker_a", "")
                speaker_b = conversation.get("speaker_b", "")
                
                conversation_id = create_conversation_id(speaker_a, speaker_b, idx)
                index_path = get_index_path(conversation_id)
                
                return (conversation_id, index_path)
    
    return None



def load_config(path: Path) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))

def parse_session_datetime(datetime_str: str) -> datetime:
    """解析 session 时间格式: "1:56 pm on 8 May, 2023" """
    try:
        from dateutil import parser
        return parser.parse(datetime_str)
    except Exception as e:
        logger.warning(f"Failed to parse datetime '{datetime_str}': {e}")
        return datetime(2023, 1, 1)

def extract_time_ranges_from_question(question: str) -> List:
    """从问题中提取时间范围"""
    from src.retrieval.types import TimeRange
    import re
    
    if question.lower().strip().startswith("when "):
        return []
    
    time_ranges = []
    
    # 模式1: 具体年份
    year_pattern = r'\bin\s+(\d{4})\b'
    year_matches = re.findall(year_pattern, question.lower())
    for year_str in year_matches:
        year = int(year_str)
        start = datetime(year, 1, 1, 0, 0, 0)
        end = datetime(year, 12, 31, 23, 59, 59, 999999)
        time_ranges.append(TimeRange(start=start, end=end))
    
    # 模式2: 具体月份
    month_pattern = r'\bin\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})\b'
    month_matches = re.findall(month_pattern, question.lower())
    for month_name, year_str in month_matches:
        month_map = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        year = int(year_str)
        month = month_map[month_name]
        
        if month == 12:
            next_month = datetime(year + 1, 1, 1)
        else:
            next_month = datetime(year, month + 1, 1)
        
        start = datetime(year, month, 1, 0, 0, 0)
        end = next_month - datetime.resolution
        time_ranges.append(TimeRange(start=start, end=end))
    
    return time_ranges[:3]


def generate_answer_with_llm(
    question: str,
    retrieved_episodes: List[Any],
    api_key: str = None,
    base_url: str = None,
) -> tuple[str, float]:
    """Generate answer using LLM with retrieved context."""
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    context_parts = []

    for i, item in enumerate(retrieved_episodes, 1):
        timestamp = getattr(item, 'timestamp', None)
        if timestamp:
            timestamp_str = timestamp.strftime('%d %B %Y, %I:%M %p')
        else:
            timestamp_str = "Unknown date"
        
        if hasattr(item, 'messages'):
            messages_text = []
            for msg in item.messages:
                role = "User" if msg.role == "user" else "Assistant"
                messages_text.append(f"{role}: {msg.content}")
            episode_text = "\n".join(messages_text)
        else:
            episode_text = getattr(item, 'content', str(item))
        
        context_parts.append(f"--- Memory {i} (Timestamp: {timestamp_str}) ---\n{episode_text}")
    
    context = "\n\n".join(context_parts) if context_parts else "No relevant conversations found."
    
    prompt = f"""
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from two speakers in a conversation. These memories contain
    timestamped information that may be relevant to answering the question.

    # IMPORTANT RULES:
    - Carefully analyze all provided memories from both speakers
    - If the question asks about a specific event or fact, look for direct evidence in the memories
    - If the memories contain contradictory information, prioritize the most recent memory
    - Each memory above shows its timestamp (when it was recorded)
    - If a memory mentions relative time like "last year", "two months ago", "last week":
      * Use that memory's timestamp as the reference point
      * Calculate the absolute date/year from the timestamp
      * Example: If a memory timestamped "8 May 2023" says "went to India last year", 
        the trip was in 2022, so answer "2022" NOT "last year"
    - Focus only on the content of the memories from both speakers. Do not confuse character
       names mentioned in memories with the actual users who created those memories.
    - The answer should be less than 5-6 words.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories from both speakers
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", etc.),
       calculate the actual date based on the memory timestamp. For example, if a memory from
       4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example,
       convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory
       timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories from both speakers. Do not confuse character
       names mentioned in memories with the actual users who created those memories.
    8. The answer should be less than 5-6 words.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    Conversation History:
    { context }

    Question: { question }

    Answer:
    """
    
    try:
        llm_start = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on conversation history."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=500,
        )
        llm_latency_ms = (time.time() - llm_start) * 1000
        
        return response.choices[0].message.content.strip(), llm_latency_ms
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return f"Error: {str(e)}", 0.0
    
# ==========单索引检索 ==========

def retrieve_from_single_index(
    question: str,
    conversation_id: str,
    index_pool: IndexPool,
    config: Dict[str, Any],
    query_tags: List[str],
    time_ranges: List,
    top_k: int,
    verbose: bool = False
):
    """
    从单个索引检索（使用 IndexPool 中已加载的索引）
    
    Args:
        question: 问题
        conversation_id: Conversation ID
        index_pool: 索引池
        config: 配置
        query_tags: Query tags
        time_ranges: 时间范围
        top_k: 返回数量
        verbose: 是否详细输出
        
    Returns:
        (search_results, metrics, detailed)
    """
    # 从池中获取 retriever
    retriever = index_pool.get_retriever(conversation_id)
    
    if not retriever:
        raise ValueError(f"Retriever not found for {conversation_id}")
    
    # 检索
    search_results, metrics, detailed = retriever.retrieve(
        query=question,
        user_id=conversation_id,
        query_tags=query_tags,
        time_ranges=time_ranges,
        top_k=top_k,
        return_metrics=True,
        return_detailed=True
    )
    
    return search_results, metrics, detailed


# ==========多索引检索 ==========

def retrieve_from_multiple_indices(
    question: str,
    matched_indices: List[Tuple[str, str]],
    index_pool: IndexPool,
    config: Dict[str, Any],
    query_tags: List[str],
    time_ranges: List,
    top_k: int,
    verbose: bool = False
):
    """
    从多个索引检索并合并结果（使用 IndexPool）
    
    Args:
        question: 问题
        matched_indices: [(conversation_id, index_path), ...]
        index_pool: 索引池
        config: 配置
        query_tags: Query tags
        time_ranges: 时间范围
        top_k: 最终返回数量
        verbose: 是否详细输出
        
    Returns:
        (merged_results, combined_metrics, combined_detailed)
    """
    all_results = []
    all_metrics = []
    
    if verbose:
        logger.info(f"Retrieving from {len(matched_indices)} indices: {[cid for cid, _ in matched_indices]}")
    
    # 对每个索引检索
    for conversation_id, index_path in matched_indices:
        try:
            results, metrics, detailed = retrieve_from_single_index(
                question=question,
                conversation_id=conversation_id,
                index_pool=index_pool,
                config=config,
                query_tags=query_tags,
                time_ranges=time_ranges,
                top_k=top_k,
                verbose=verbose
            )
            
            all_results.extend(results)
            all_metrics.append(metrics)
            
            if verbose:
                logger.info(f"  {conversation_id}: {len(results)} results")
        
        except Exception as e:
            logger.error(f"Failed to retrieve from {conversation_id}: {e}")
            continue
    
    # 合并结果（去重 + 排序）
    seen = set()
    merged = []
    for result in sorted(all_results, key=lambda x: x.score, reverse=True):
        if result.memory_id not in seen:
            seen.add(result.memory_id)
            merged.append(result)
    
    merged = merged[:top_k]
    
    # 合并 metrics（简单求平均）
    if all_metrics:
        from src.retrieval.types import RetrievalMetrics
        
        combined_metrics = RetrievalMetrics(
            total_memories=sum(m.total_memories for m in all_metrics),
            temporal_filtered=sum(m.temporal_filtered for m in all_metrics) // len(all_metrics),
            semantic_filtered=sum(m.semantic_filtered for m in all_metrics) // len(all_metrics),
            vector_candidates=sum(m.vector_candidates for m in all_metrics) // len(all_metrics),
            final_results=len(merged),
            temporal_time_ms=sum(m.temporal_time_ms for m in all_metrics) / len(all_metrics),
            semantic_time_ms=sum(m.semantic_time_ms for m in all_metrics) / len(all_metrics),
            vector_time_ms=sum(m.vector_time_ms for m in all_metrics) / len(all_metrics),
            get_mem_time_ms=sum(m.get_mem_time_ms for m in all_metrics) / len(all_metrics),
            rerank_time_ms=sum(m.rerank_time_ms for m in all_metrics) / len(all_metrics),
            total_time_ms=sum(m.total_time_ms for m in all_metrics)
        )
    else:
        combined_metrics = None
    
    return merged, combined_metrics, None


def process_single_question_with_retriever(
    config: Dict[str, Any],
    user_id: str,
    qa_item: Dict[str, Any],
    dataset: List[Dict[str, Any]],
    index_pool: IndexPool,
    query_embedding_cache: Dict[str, np.ndarray],
    tracker: Dict[str, Any],
    tracker_lock: threading.Lock,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
   强制使用单索引检索（通过 QA 归属推断）
    """
    result = {
        "user_id": user_id,
        "question": qa_item.get("question", ""),
        "answer": qa_item.get("answer", ""),
        "category": qa_item.get("category", 0),
        "response": "",
        "retrieved_count": 0,
        "success": False,
        "error": None,
        "search_latency_ms": 0.0,
        "llm_latency_ms": 0.0,
        "e2e_latency_ms": 0.0,
        "inferred_tags": [],
        "retrieval_metrics": {},
        "timing_breakdown": {},
        "matched_indices": [] 
    }
    
    e2e_start = time.time()
    
    try:
        question = result["question"]
        timing = {}
        
        #1. 直接推断 conversation（移除人名提取和索引匹配）
        t_start = time.time()
        inferred = infer_conversation_from_qa(qa_item, dataset)
        timing["infer_conversation_ms"] = (time.time() - t_start) * 1000
        
        if not inferred:
            result["error"] = "Cannot infer conversation from QA item"
            result["success"] = False
            result["timing_breakdown"] = timing
            return result
        
        conversation_id, index_path = inferred
        result["matched_indices"] = [conversation_id]
        
        if verbose:
            logger.info(f"Inferred conversation: {conversation_id}")
        
        #2. 推断 query tags
        t_start = time.time()
        tag_router = index_pool.get_tag_router(conversation_id)
        
        if not tag_router:
            raise RuntimeError(f"TagRouter not found for {conversation_id}")
        
        # 从缓存获取 query embedding
        query_embedding = query_embedding_cache.get(question)
        
        if query_embedding is None:
            manager = index_pool.get_manager(conversation_id)
            if manager:
                query_embedding = manager.embedding_manager.get_embedding(question)
        
        query_tags = tag_router.infer_tags(
            query=question,
            method="embedding",
            top_k=5,
            query_embedding=query_embedding
        )
        
        result["inferred_tags"] = query_tags
        timing["infer_tags_ms"] = (time.time() - t_start) * 1000
        
        if verbose:
            logger.info(f"Inferred tags: {query_tags}")
        
        #3. 提取时间范围
        t_start = time.time()
        time_ranges = extract_time_ranges_from_question(question)
        timing["extract_time_ranges_ms"] = (time.time() - t_start) * 1000
        
        #4. 单索引检索
        t_start = time.time()
        search_results, metrics, detailed = retrieve_from_single_index(
            question=question,
            conversation_id=conversation_id,
            index_pool=index_pool,
            config=config,
            query_tags=query_tags,
            time_ranges=time_ranges,
            top_k=config.get("final_top_k", 50),
            verbose=verbose
        )
        
        timing["index_search_ms"] = (time.time() - t_start) * 1000
        result["search_latency_ms"] = (time.time() - e2e_start) * 1000
        result["retrieved_count"] = len(search_results)
        
        # 保存 metrics
        if metrics:
            result["retrieval_metrics"] = {
                "stage1_candidates": metrics.temporal_filtered,
                "stage2_candidates": metrics.semantic_filtered,
                "stage3_candidates": metrics.vector_candidates,
                "final_results": metrics.final_results,
                "temporal_time_ms": metrics.temporal_time_ms,
                "semantic_time_ms": metrics.semantic_time_ms,
                "vector_time_ms": metrics.vector_time_ms,
                "get_mem_time_ms": metrics.get_mem_time_ms,
                "rerank_time_ms": metrics.rerank_time_ms,
                "total_time_ms": metrics.total_time_ms
            }
        
        # 详细日志
        if verbose:
            if len(search_results) == 0:
                logger.warning(f"⚠️  No results for question: {question[:50]}...")
            else:
                logger.info(f"✅ Retrieved {len(search_results)} results")
        
        #5. 生成答案
        if config.get("enable_llm_answer", True):
            episodes = [r for r in search_results]
            
            answer, llm_latency = generate_answer_with_llm(
                question=question,
                retrieved_episodes=episodes,
                api_key=os.getenv("LLM_API_KEY"),
                base_url=os.getenv("LLM_BASE_URL"),
            )
            result["response"] = answer
            result["llm_latency_ms"] = llm_latency
        else:
            content_parts = [r.content[:100] for r in search_results[:3]]
            result["response"] = " ".join(content_parts)
            result["llm_latency_ms"] = 0.0
        
        result["success"] = True
        
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}"
        if verbose:
            logger.error(f"Error processing question: {e}")
            import traceback
            traceback.print_exc()
    
    finally:
        result["e2e_latency_ms"] = (time.time() - e2e_start) * 1000
        result["timing_breakdown"] = timing
        
        with tracker_lock:
            if user_id not in tracker:
                tracker[user_id] = []
            tracker[user_id].append(result)
    
    return result


# ==========处理所有问题 ==========

def process_all_questions_with_retriever(
    config: Dict[str, Any],
    dataset: List[Dict[str, Any]],
    index_pool: IndexPool,
    max_workers: int,
    verbose: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """
   批量预生成 query embeddings
    """
    tracker: Dict[str, List[Dict[str, Any]]] = {}
    tracker_lock = threading.Lock()
    
    # Collect all questions
    all_tasks = []

    for idx, item in enumerate(dataset):
        conversation = item.get("conversation", {})
        speaker_a = conversation.get("speaker_a", "speaker_a")
        user_id = f"{speaker_a}_{idx}"
        
        qa_list = item.get("qa", [])
        
        if not qa_list:
            if verbose:
                logger.warning(f"⚠️  No QA found for {user_id}")
            continue
        
        for qa_item in qa_list:
            all_tasks.append((user_id, qa_item))
    
    if not all_tasks:
        print("\n⚠️  Warning: No questions found in dataset!")
        return tracker
    
    #批量预生成所有 query embeddings
    print(f"\n📌 Pre-generating query embeddings for {len(all_tasks)} questions...")
    
    all_questions = [qa_item["question"] for _, qa_item in all_tasks]
    unique_questions = list(set(all_questions))
    
    print(f"   Unique questions: {len(unique_questions)}")
    
    # 使用第一个 manager 的 embedding_manager
    first_manager = list(index_pool.managers.values())[0]
    
    batch_start = time.time()
    query_embeddings_batch = first_manager.embedding_manager.get_embeddings_batch(
        texts=unique_questions,
        use_cache=True
    )
    batch_time = (time.time() - batch_start) * 1000
    
    print(f"  Batch embedding generation took: {batch_time:.2f}ms")
    print(f"   📊 Average per question: {batch_time / len(unique_questions):.2f}ms")
    
    # 构建 question -> embedding 映射
    query_embedding_cache = {
        question: embedding 
        for question, embedding in zip(unique_questions, query_embeddings_batch)
    }
    
    print(f"\n🔍 Processing {len(all_tasks)} questions with {max_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for user_id, qa_item in all_tasks:
            future = executor.submit(
                process_single_question_with_retriever,
                config,
                user_id,
                qa_item,
                dataset,
                index_pool,
                query_embedding_cache,  #传入缓存
                tracker,
                tracker_lock,
                verbose,
            )
            futures.append(future)
        
        with tqdm(total=len(futures), desc="Questions", ncols=80) as pbar:
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    if verbose:
                        print(f"\n⚠️  Question processing failed: {e}")
                pbar.update(1)
    
    # Print summary
    print_search_summary(tracker)
    
    return tracker



def print_search_summary(tracker: Dict[str, List[Dict[str, Any]]]):
    """打印检索摘要"""
    print("\n" + "=" * 80)
    print("📊 Search Summary")
    print("=" * 80)
    
    all_results = [r for results in tracker.values() for r in results]
    total_questions = len(all_results)
    successful = sum(1 for r in all_results if r.get("success", False))
    failed = total_questions - successful
    
    if successful > 0:
        successful_results = [r for r in all_results if r.get("success", False)]
        
        avg_search = sum(r.get("search_latency_ms", 0) for r in successful_results) / successful
        avg_llm = sum(r.get("llm_latency_ms", 0) for r in successful_results) / successful
        avg_e2e = sum(r.get("e2e_latency_ms", 0) for r in successful_results) / successful
        avg_retrieved = sum(r.get("retrieved_count", 0) for r in successful_results) / successful
        
        results_with_metrics = [
            r for r in successful_results 
            if r.get("retrieval_metrics")
        ]
        
        if results_with_metrics:
            avg_temporal = sum(
                r["retrieval_metrics"].get("temporal_time_ms", 0) 
                for r in results_with_metrics
            ) / len(results_with_metrics)
            
            avg_semantic = sum(
                r["retrieval_metrics"].get("semantic_time_ms", 0) 
                for r in results_with_metrics
            ) / len(results_with_metrics)
            
            avg_vector = sum(
                r["retrieval_metrics"].get("vector_time_ms", 0) 
                for r in results_with_metrics
            ) / len(results_with_metrics)
            
            avg_get_mem = sum(
                r["retrieval_metrics"].get("get_mem_time_ms", 0) 
                for r in results_with_metrics
            ) / len(results_with_metrics)
            
            avg_rerank = sum(
                r["retrieval_metrics"].get("rerank_time_ms", 0) 
                for r in results_with_metrics
            ) / len(results_with_metrics)
            
            avg_total_retrieval = sum(
                r["retrieval_metrics"].get("total_time_ms", 0) 
                for r in results_with_metrics
            ) / len(results_with_metrics)
        else:
            avg_temporal = avg_semantic = avg_vector = 0.0
            avg_get_mem = avg_rerank = avg_total_retrieval = 0.0
        
        print(f"✅ Successful: {successful}/{total_questions}")
        print(f"❌ Failed: {failed}/{total_questions}")
        print(f"📝 Total Episodes: {sum(r.get('retrieved_count', 0) for r in all_results)}")
        print(f"📊 Avg Retrieved Results: {avg_retrieved:.2f}")
        
        print(f"\n⏱️  Retrieval Pipeline Breakdown (avg):")
        print(f"   ├─ Stage 1 (Temporal):   {avg_temporal:>8.2f}ms")
        print(f"   ├─ Stage 2 (Tag DAG):    {avg_semantic:>8.2f}ms")
        print(f"   ├─ Stage 3 (Vector):     {avg_vector:>8.2f}ms")
        print(f"   ├─ Stage 4a (Get Mem):   {avg_get_mem:>8.2f}ms")
        print(f"   ├─ Stage 4b (Rerank):    {avg_rerank:>8.2f}ms")
        print(f"   └─ Total Retrieval:      {avg_total_retrieval:>8.2f}ms")
        
        print(f"\n⏱️  End-to-End Latency (avg):")
        print(f"   ├─ 🔍 Retrieval:         {avg_search:>8.2f}ms")
        print(f"   ├─ 🤖 LLM Generation:    {avg_llm:>8.2f}ms")
        print(f"   └─ 📊 Total E2E:         {avg_e2e:>8.2f}ms")

        #search.py 流程细分统计
        results_with_timing = [r for r in successful_results if r.get("timing_breakdown")]
        
        if results_with_timing:
            print(f"\n🔬 search.py Pipeline Breakdown (avg):")
            
            # 计算各阶段平均耗时
            avg_extract_speakers = sum(
                r["timing_breakdown"].get("extract_speakers_ms", 0)
                for r in results_with_timing
            ) / len(results_with_timing)
            
            avg_match_indices = sum(
                r["timing_breakdown"].get("match_indices_ms", 0)
                for r in results_with_timing
            ) / len(results_with_timing)
            
            avg_infer_conversation = sum(
                r["timing_breakdown"].get("infer_conversation_ms", 0)
                for r in results_with_timing
            ) / len(results_with_timing)
            
            avg_infer_tags = sum(
                r["timing_breakdown"].get("infer_tags_ms", 0)
                for r in results_with_timing
            ) / len(results_with_timing)
            
            avg_extract_time = sum(
                r["timing_breakdown"].get("extract_time_ranges_ms", 0)
                for r in results_with_timing
            ) / len(results_with_timing)
            
            avg_index_search = sum(
                r["timing_breakdown"].get("index_search_ms", 0)
                for r in results_with_timing
            ) / len(results_with_timing)
            
            print(f"   ├─ Extract Speakers:     {avg_extract_speakers:>8.2f}ms")
            print(f"   ├─ Match Indices:        {avg_match_indices:>8.2f}ms")
            print(f"   ├─ Infer Conversation:   {avg_infer_conversation:>8.2f}ms")
            print(f"   ├─ Infer Tags:           {avg_infer_tags:>8.2f}ms")
            print(f"   ├─ Extract Time Ranges:  {avg_extract_time:>8.2f}ms")
            print(f"   ├─ Index Search:         {avg_index_search:>8.2f}ms")
            print(f"   └─ LLM Generation:       {avg_llm:>8.2f}ms")
        
        if results_with_metrics:
            avg_s1 = sum(
                r["retrieval_metrics"]["stage1_candidates"] 
                for r in results_with_metrics
            ) / len(results_with_metrics)
            avg_s2 = sum(
                r["retrieval_metrics"]["stage2_candidates"] 
                for r in results_with_metrics
            ) / len(results_with_metrics)
            avg_s3 = sum(
                r["retrieval_metrics"]["stage3_candidates"] 
                for r in results_with_metrics
            ) / len(results_with_metrics)
            
            print(f"\n🎯 Candidate Filtering (avg):")
            print(f"   Stage 1 (Temporal): {avg_s1:>8.1f} candidates")
            print(f"   Stage 2 (Tag DAG):  {avg_s2:>8.1f} candidates")
            print(f"   Stage 3 (Vector):   {avg_s3:>8.1f} candidates")
        
        #索引使用统计
        multi_index_count = sum(1 for r in successful_results if len(r.get("matched_indices", [])) > 1)
        single_index_count = sum(1 for r in successful_results if len(r.get("matched_indices", [])) == 1)
        
        print(f"\n🗂️  Index Usage:")
        print(f"   Single index queries: {single_index_count}")
        print(f"   Multi-index queries:  {multi_index_count}")
        
        print(f"\n📝 Sample Results (first 3):")
        print("─" * 80)
        for i, (user_id, results) in enumerate(list(tracker.items())[:3]):
            if results:
                r = results[0]
                print(f"\n{i+1}. User: {user_id}")
                print(f"   Question: {r['question'][:60]}...")
                print(f"   Matched indices: {r.get('matched_indices', [])}")
                print(f"   Inferred Tags: {r['inferred_tags']}")
                print(f"   Retrieved: {r['retrieved_count']} results")
                
                if r.get("retrieval_metrics"):
                    m = r["retrieval_metrics"]
                    print(f"   Pipeline: S1={m['stage1_candidates']}, "
                          f"S2={m['stage2_candidates']}, "
                          f"S3={m['stage3_candidates']}")
    else:
        print(f"❌ All {total_questions} questions failed")
    
    print("=" * 80 + "\n")


# ==========main() ==========

def main() -> None:
    parser = argparse.ArgumentParser(description="Search and generate answers for LoCoMo questions")
    parser.add_argument("--data", default="dataset/locomo10.json")
    parser.add_argument("--config", default="evaluation/config.json")
    parser.add_argument("--output", default="results/search_results.json")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    config = load_config(config_path)
    
    # Override with command line args
    if args.max_workers is not None:
        config["max_workers_search"] = args.max_workers
    
    # Load dataset
    dataset_path = Path(args.data)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    
    # 数据验证
    print(f"\n📊 Dataset loaded: {len(dataset)} conversations")
    if dataset:
        print(f"📋 First conversation keys: {list(dataset[0].keys())}")
        if "qa" in dataset[0]:
            print(f"💬 First conversation has {len(dataset[0]['qa'])} QA pairs")
            if dataset[0]['qa']:
                print(f"🔍 First question: {dataset[0]['qa'][0].get('question', 'N/A')[:50]}...")
    
    # ==========初始化 IndexPool ==========
    print("\n" + "=" * 80)
    print("🔍 Starting retrieval with IndexPool...")
    print("=" * 80)
    
    index_pool = IndexPool(dataset=dataset, config=config)
    
    try:
        # ========== 执行检索 ==========
        results = process_all_questions_with_retriever(
            config=config,
            dataset=dataset,
            index_pool=index_pool,
            max_workers=config.get("max_workers_search", 10),
            verbose=args.verbose or config.get("verbose", False)
        )
        
        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results_summary = {}
        for user_id, user_results in results.items():
            results_summary[user_id] = []
            for r in user_results:
                summary_item = {k: v for k, v in r.items() if k != "detailed_stages"}
                results_summary[user_id].append(summary_item)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {output_path}")
    
    finally:
        # ==========清理 IndexPool ==========
        index_pool.close_all()


if __name__ == "__main__":
    main()