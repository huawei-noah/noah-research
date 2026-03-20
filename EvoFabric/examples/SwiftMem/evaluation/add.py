# evaluation/add.py
"""Add LoCoMo dataset conversations into TridentMem"""

from __future__ import annotations

import argparse
import json
import time
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import threading

import sys
from pathlib import Path

# 添加这几行 - 确保能找到 src 模块
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from tqdm import tqdm

from src.memory.memory_manager import MemoryManager
from src.models.message import Message
from src.models.episode import Episode


from src.llm.client import LLMClient
from src.memory.tag_generator import TagGenerator
from src.segmentation import EpisodeSegmentor


load_dotenv()

logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)



def create_conversation_id(speaker_a: str, speaker_b: str, idx: int) -> str:
    """
    创建 conversation ID
    
    规则：speaker 按字母序排列（保证唯一性）
    
    Examples:
        create_conversation_id("Caroline", "Melanie", 0) -> "conv_0_Caroline_Melanie"
        create_conversation_id("Melanie", "Caroline", 0) -> "conv_0_Caroline_Melanie"  # 相同
    
    Args:
        speaker_a: Speaker A
        speaker_b: Speaker B
        idx: Conversation index
        
    Returns:
        Conversation ID
    """
    speakers = sorted([speaker_a, speaker_b])
    return f"conv_{idx}_{speakers[0]}_{speakers[1]}"


def get_index_path(conversation_id: str, base_path: str = "data/indices") -> str:
    """
    获取 conversation 的索引路径
    
    Examples:
        get_index_path("conv_0_Caroline_Melanie") 
        -> "data/indices/locomo_Caroline_Melanie"
    
    Args:
        conversation_id: Conversation ID
        base_path: 索引基础路径
        
    Returns:
        索引完整路径
    """
    
    parts = conversation_id.split("_", 2)  # ["conv", "0", "Caroline_Melanie"]
    speaker_pair = parts[2]  # "Caroline_Melanie"
    
    return f"{base_path}/locomo_{speaker_pair}"


def load_config(path: Path) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def parse_timestamp(value: str) -> datetime:
    """
    Parse dataset timestamps such as "1:56 pm on 8 May, 2023".
    
    Args:
        value: 时间戳字符串
        
    Returns:
        解析后的 datetime
        
    Raises:
        ValueError: 如果时间戳格式无效或缺少必要信息
    """
    if not value or not value.strip():
        raise ValueError("Empty timestamp value")
    
    value = " ".join(value.split())
    
    # 尝试 ISO 格式
    if " on " not in value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as e:
            raise ValueError(f"Invalid ISO timestamp format: {value}") from e
    
    # 解析 "1:56 pm on 8 May, 2023" 格式
    time_part, date_part = value.split(" on ")
    time_part = time_part.lower().strip()
    
    hour = 0
    minute = 0
    is_pm = "pm" in time_part
    time_part = time_part.replace("pm", "").replace("am", "").strip()
    
    if ":" in time_part:
        hour_str, minute_str = time_part.split(":", 1)
        hour = int(hour_str)
        minute = int(minute_str)
    else:
        hour = int(time_part)
    
    if is_pm and hour != 12:
        hour += 12
    if not is_pm and hour == 12:
        hour = 0
    
    months = {
        "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
        "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6,
        "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9,
        "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
    }
    
    parts = date_part.replace(",", "").split()
    day = None
    month = None
    year = None
    
    for part in parts:
        lower = part.lower()
        if lower in months:
            month = months[lower]
        elif part.isdigit():
            num = int(part)
            if num > 31:
                year = num
            else:
                day = num
    
    # 必须有完整的日期信息
    if year is None:
        raise ValueError(f"Missing year in timestamp: {value}")
    if month is None:
        raise ValueError(f"Missing month in timestamp: {value}")
    if day is None:
        raise ValueError(f"Missing day in timestamp: {value}")
    
    return datetime(year=year, month=month, day=day, hour=hour, minute=minute)

def extract_sessions(conversation: Dict[str, Any]) -> List[str]:
    """
    提取对话中的所有 session keys
    
    Returns:
        ["session_1", "session_2", ...] 按顺序排列
    """
    sessions = []
    for key in conversation.keys():
        if key.startswith("session_") and not key.endswith("_date_time"):
            sessions.append(key)
    
    # 按数字排序
    sessions.sort(key=lambda x: int(x.split("_")[1]))
    return sessions


def build_messages_for_session(
    conversation: Dict[str, Any],
    session_key: str,
    perspective_speaker: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    为特定 session 构建消息
    
    Args:
        conversation: 对话数据
        session_key: session key (如 "session_1")
        perspective_speaker: 从谁的视角构建消息
        
    Returns:
        消息列表
    """
    speaker_a = conversation.get("speaker_a", "speaker_a")
    speaker_b = conversation.get("speaker_b", "speaker_b")
    
    if perspective_speaker is None:
        perspective_speaker = speaker_a
    
    # 获取该 session 的消息
    chats = conversation.get(session_key, [])
    timestamp_raw = conversation.get(f"{session_key}_date_time")
    if not timestamp_raw:
        raise ValueError(f"Missing timestamp for {session_key} in conversation")
    timestamp = parse_timestamp(timestamp_raw)
    
    messages: List[Dict[str, Any]] = []
    
    for chat in chats or []:
        speaker = chat.get("speaker", speaker_a)
        text = chat.get("text", "")
        parts = [text]
        
        if chat.get("blip_caption"):
            parts.append(f"[Image: {chat['blip_caption']}]")
        if chat.get("query"):
            parts.append(f"[Search: {chat['query']}]")
        
        role = "user" if speaker == perspective_speaker else "assistant"
        
        messages.append({
            "role": role,
            "content": " ".join(parts),
            "timestamp": timestamp,
            "metadata": {
                "original_speaker": speaker,
                "dataset_timestamp": timestamp_raw,
                "session_key": session_key,
                "blip_caption": chat.get("blip_caption"),
                "search_query": chat.get("query"),
            },
        })
    
    return messages


def convert_session_to_tag_input(
    messages_data: List[Dict[str, Any]],
    session_key: str,
    speaker: str,
    other_speaker: str
) -> Dict[str, Any]:
    """
    将 session 的消息数据转换为 tag_generator 能理解的格式
    
    Args:
        messages_data: Session 的消息列表
        session_key: Session key (如 "session_1")
        speaker: 当前说话者
        other_speaker: 对方说话者
        
    Returns:
        适合 tag_generator 的字典格式
    """
    # 构建消息列表
    formatted_messages = []
    for msg in messages_data:
        formatted_messages.append({
            "speaker": msg["metadata"]["original_speaker"],
            "text": msg["content"]
        })
    
    # 返回 tag_generator 期望的格式
    return {
        "speaker_a": speaker,
        "speaker_b": other_speaker,
        session_key: formatted_messages,
        f"{session_key}_date_time": messages_data[0]["metadata"]["dataset_timestamp"] if messages_data else None
    }




def process_single_conversation(
    manager: MemoryManager,
    item: Dict[str, Any],
    idx: int,
    tracker: Dict[str, Any],
    tracker_lock: threading.Lock,
    tag_generator: TagGenerator,
    verbose: bool = False,
    max_retries: int = 3,
) -> None:
    """
     修改：为 conversation 的两个 speaker 使用共享的 conversation_id
    """
    conversation = item.get("conversation", {})
    speaker_a = conversation.get("speaker_a", "speaker_a")
    speaker_b = conversation.get("speaker_b", "speaker_b")
    
    
    conversation_id = create_conversation_id(speaker_a, speaker_b, idx)
    
    results = {}
    
    # ========== 处理 Speaker A 的 episodes ==========
    result_a = process_speaker_perspective(
        manager=manager,
        conversation=conversation,
        speaker=speaker_a,
        other_speaker=speaker_b,
        conversation_id=conversation_id,  
        idx=idx,
        tracker_lock=tracker_lock,
        tag_generator=tag_generator,
        verbose=verbose,
        max_retries=max_retries
    )
    #  使用 conversation_id 作为 tracker key
    results[conversation_id] = result_a
    
    # ========== 处理 Speaker B 的 episodes ==========
    result_b = process_speaker_perspective(
        manager=manager,
        conversation=conversation,
        speaker=speaker_b,
        other_speaker=speaker_a,
        conversation_id=conversation_id, 
        idx=idx,
        tracker_lock=tracker_lock,
        tag_generator=tag_generator,
        verbose=verbose,
        max_retries=max_retries
    )
    
    #  合并结果（累加统计）
    with tracker_lock:
        if conversation_id in tracker:
            # 已存在，合并统计
            existing = tracker[conversation_id]
            existing["message_count"] += result_b.get("message_count", 0)
            existing["episode_count"] += result_b.get("episode_count", 0)
            existing["all_tags_generated"].extend(result_b.get("all_tags_generated", []))
            existing["all_relations_generated"].extend(result_b.get("all_relations_generated", []))
        else:
            tracker[conversation_id] = result_a


def process_speaker_perspective(
    manager: MemoryManager,
    conversation: Dict[str, Any],
    speaker: str,
    other_speaker: str,
    conversation_id: str,  
    idx: int,
    tracker_lock: threading.Lock,
    tag_generator: TagGenerator, 
    verbose: bool = False,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
     使用 conversation_id 作为 user_id
    """
    result = {
        "conversation_id": conversation_id,  
        "success": False,
        "message_count": 0,
        "episode_count": 0,
        "error": None,
        "stage": "initialized",
        "retries": 0,
        "processing_time_ms": 0.0,
        "all_tags_generated": [],
        "all_relations_generated": []
    }
    
    start_time = time.time()
    
    try:
        result["stage"] = "extracting_sessions"
        
        # 提取所有 sessions
        sessions = extract_sessions(conversation)
        
        if not sessions:
            result["error"] = "No sessions found"
            result["stage"] = "validation_failed"
            return result
        
        if verbose:
            logger.info(f"Found {len(sessions)} sessions for {conversation_id}")

        # ============ 初始化 EpisodeSegmentor ============
        config_path = Path(__file__).parent / "config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        from src.llm.client import LLMClient
        llm_client = LLMClient()
        
        segmentor = EpisodeSegmentor(
            llm_client=llm_client,
            split_threshold=config.get("conversation_split_threshold", 15),
            temperature=config.get("splitting_temperature", 0.2)
        )
        
        enable_splitting = config.get("enable_episode_splitting", True)
        
        if verbose:
            logger.info(f"Episode splitting: {'enabled' if enable_splitting else 'disabled'}")
        
        result["stage"] = "processing_sessions"
        
        # ============ 为每个 session 创建 episodes ============
        for session_key in sessions:
            messages_data = build_messages_for_session(
                conversation, 
                session_key, 
                perspective_speaker=speaker
            )
            
            if not messages_data:
                if verbose:
                    logger.warning(f"No messages in {session_key} for {conversation_id}")
                continue
            
            result["message_count"] += len(messages_data)
            
            # ============ 构建 Message 对象 ============
            messages = []
            for msg_data in messages_data:
                msg = Message(
                    role=msg_data["role"],
                    content=msg_data["content"],
                    timestamp=msg_data["timestamp"],
                    metadata=msg_data.get("metadata", {})
                )
                messages.append(msg)
            
            # ============ 核心分段逻辑 ============
            session_num = session_key.split("_")[1]
            
            # 准备 session metadata
            session_metadata = {
                "dataset_idx": idx,
                "speaker": speaker,
                "other_speaker": other_speaker,
                "speaker_a": conversation.get("speaker_a"),
                "speaker_b": conversation.get("speaker_b"),
                "session_key": session_key,
                "session_num": int(session_num)
            }
            
            if enable_splitting and len(messages) >= segmentor.split_threshold:
                # 拆分为多个 episodes
                if verbose:
                    logger.info(
                        f"Splitting {len(messages)} messages in {session_key} "
                        f"(threshold: {segmentor.split_threshold})"
                    )
                
                episodes = segmentor.split_session_to_episodes(
                    user_id=conversation_id,  
                    messages=messages,
                    session_metadata=session_metadata
                )
                
                # 为每个 episode 生成 tags
                for ep_idx, episode in enumerate(episodes):
                    episode_messages_data = []
                    for msg in episode.messages:
                        episode_messages_data.append({
                            "role": msg.role,
                            "content": msg.content,
                            "timestamp": msg.timestamp,
                            "metadata": msg.metadata if hasattr(msg, 'metadata') else {}
                        })
                    
                    episode_input = convert_session_to_tag_input(
                        messages_data=episode_messages_data,
                        session_key=f"{session_key}_ep{ep_idx+1}",
                        speaker=speaker,
                        other_speaker=other_speaker
                    )
                    
                    try:
                        episode_tags, episode_relations = tag_generator.generate_tags_with_relations(episode_input)
                        
                        if verbose:
                            logger.info(
                                f"Generated tags for episode {ep_idx+1}/{len(episodes)} "
                                f"from {session_key}: {episode_tags}"
                            )
                        
                        episode.tags = episode_tags
                        episode.tag_relations = episode_relations
                        
                        result["all_tags_generated"].extend(episode_tags)
                        result["all_relations_generated"].extend(episode_relations)
                        
                    except Exception as e:
                        logger.error(
                            f"Failed to generate tags for episode {ep_idx+1} "
                            f"from {session_key}: {e}"
                        )
                        continue
                    
                    # 添加 episode (带重试)
                    for retry in range(max_retries):
                        try:
                            result["stage"] = f"adding_{session_key}_episode_{ep_idx+1}_attempt_{retry+1}"
                            with tracker_lock:
                                manager.add_episode(episode)
                            result["episode_count"] += 1
                            if verbose:
                                logger.info(
                                    f"Added episode {ep_idx+1}/{len(episodes)} from {session_key} "
                                    f"for {conversation_id}: {episode.title}"
                                )
                            break
                        except Exception as e:
                            if retry == max_retries - 1:
                                raise
                            time.sleep(0.5 * (retry + 1))
                            result["retries"] = retry + 1
                
            else:
                # 单个 episode
                if verbose and enable_splitting:
                    logger.info(
                        f"Session {session_key} has {len(messages)} messages "
                        f"(below threshold {segmentor.split_threshold}), using single episode"
                    )
                
                episode_messages_data = []
                for msg in messages:
                    episode_messages_data.append({
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp,
                        "metadata": msg.metadata if hasattr(msg, 'metadata') else {}
                    })
                
                session_input = convert_session_to_tag_input(
                    messages_data=episode_messages_data,
                    session_key=session_key,
                    speaker=speaker,
                    other_speaker=other_speaker
                )
                
                try:
                    session_tags, session_relations = tag_generator.generate_tags_with_relations(session_input)
                    
                    if verbose:
                        logger.info(f"Generated tags for {session_key} of {conversation_id}: {session_tags}")
                    
                    result["all_tags_generated"].extend(session_tags)
                    result["all_relations_generated"].extend(session_relations)
                    
                except Exception as e:
                    logger.error(f"Failed to generate tags for {session_key} of {conversation_id}: {e}")
                    continue
                
                # 创建 Episode
                episode = Episode(
                    user_id=conversation_id,  
                    messages=messages,
                    title=f"{speaker} - {other_speaker} ({session_key})",
                    content=" ".join([m.content for m in messages[:5]]),
                    timestamp=messages[0].timestamp,
                    tags=session_tags,
                    tag_relations=session_relations,
                    metadata=session_metadata
                )
                
                # 添加 episode (带重试)
                for retry in range(max_retries):
                    try:
                        result["stage"] = f"adding_{session_key}_attempt_{retry+1}"
                        with tracker_lock:
                            manager.add_episode(episode)
                        result["episode_count"] += 1
                        if verbose:
                            logger.info(f"Added {session_key} for {conversation_id}")
                        break
                    except Exception as e:
                        if retry == max_retries - 1:
                            raise
                        time.sleep(0.5 * (retry + 1))
                        result["retries"] = retry + 1
        
        result["success"] = True
        result["stage"] = "completed"
        
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}"
        result["stage"] = f"exception_at_{result.get('stage', 'unknown')}"
        if verbose:
            logger.error(f"Error processing {conversation_id}: {e}")
    
    finally:
        result["processing_time_ms"] = (time.time() - start_time) * 1000
    
    return result



def process_dataset(
    config: Dict[str, Any],
    dataset: List[Dict[str, Any]],
    max_workers: int,
    verbose: bool = False,
    max_retries: int = 3,
) -> Dict[str, MemoryManager]: 
    """
    为每个 conversation 创建独立的 MemoryManager 和索引
    
    Returns:
        {conversation_id: MemoryManager} 字典
    """
    tracker: Dict[str, Any] = {}
    tracker_lock = threading.Lock()
    
    # 初始化 TagGenerator
    print("\n🤖 Initializing LLM Tag Generator...")
    try:
        llm_client = LLMClient(
            api_key=os.getenv("LLM_API_KEY"),
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            base_url=os.getenv("LLM_BASE_URL")
        )
        tag_generator = TagGenerator(llm_client)
        print(" Tag Generator initialized successfully")
    except Exception as e:
        print(f"⚠️  Warning: Failed to initialize LLM Tag Generator: {e}")
        print("   Will use keyword-based fallback for tag extraction")
        tag_generator = None
    
    #  为每个 conversation 创建独立的 MemoryManager
    managers: Dict[str, MemoryManager] = {}
    
    print(f"\n🚀 Processing {len(dataset)} conversations with {max_workers} workers...")
    
    #  预先为每个 conversation 创建 manager
    for idx, item in enumerate(dataset):
        conversation = item.get("conversation", {})
        speaker_a = conversation.get("speaker_a", "speaker_a")
        speaker_b = conversation.get("speaker_b", "speaker_b")
        
        conversation_id = create_conversation_id(speaker_a, speaker_b, idx)
        
        # 创建独立的数据库和索引路径
        conv_db_dir = Path(f"data/conversations/{conversation_id}")
        conv_db_dir.mkdir(parents=True, exist_ok=True)
        
        manager = MemoryManager(
            episode_db_path=str(conv_db_dir / "episodes.db"),
            semantic_db_path=str(conv_db_dir / "semantic.db"),
            openai_api_key=os.getenv("EMBEDDING_API_KEY"),
            embedding_model=os.getenv("EMBEDDING_MODEL"),
            embedding_cache_path=config["embedding_cache_path"],
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "1536")),
            openai_base_url=os.getenv("EMBEDDING_BASE_URL")
        )
        
        managers[conversation_id] = manager
    
    print(f" Created {len(managers)} independent MemoryManagers")
    
    # 处理所有 conversations
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, item in enumerate(dataset):
            conversation = item.get("conversation", {})
            speaker_a = conversation.get("speaker_a", "speaker_a")
            speaker_b = conversation.get("speaker_b", "speaker_b")
            conversation_id = create_conversation_id(speaker_a, speaker_b, idx)
            
            manager = managers[conversation_id]
            
            future = executor.submit(
                process_single_conversation,
                manager,
                item,
                idx,
                tracker,
                tracker_lock,
                tag_generator,
                verbose,
                max_retries,
            )
            futures.append((future, idx, item, conversation_id))
        
        with tqdm(total=len(futures), desc="Overall", ncols=80) as pbar:
            for future, idx, item, conversation_id in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"\n⚠️  [{idx+1}] {conversation_id} failed: {e}")
                pbar.update(1)
    
    # 打印统计
    print("\n" + "=" * 80)
    print("📊 Processing Summary")
    print("=" * 80)
    
    successful = sum(1 for r in tracker.values() if r.get("success", False))
    failed = len(tracker) - successful
    total_episodes = sum(r.get("episode_count", 0) for r in tracker.values())
    total_messages = sum(r.get("message_count", 0) for r in tracker.values())
    avg_time = sum(r.get("processing_time_ms", 0) for r in tracker.values()) / len(tracker) if tracker else 0
    
    all_tags = []
    all_relations = []
    for result in tracker.values():
        all_tags.extend(result.get("all_tags_generated", []))
        all_relations.extend(result.get("all_relations_generated", []))

    unique_tags = set(all_tags)
    unique_relations = set(all_relations)

    print(f" Successful: {successful}/{len(dataset)}")
    print(f"❌ Failed: {failed}/{len(dataset)}")
    print(f"📝 Total Episodes: {total_episodes}")
    print(f"💬 Total Messages: {total_messages}")
    print(f"🏷️  Unique Tags: {len(unique_tags)}")
    print(f"🏷️  Total Tags: {len(all_tags)}")
    print(f"🔗 Unique Relations: {len(unique_relations)}")
    print(f"🔗 Total Relations: {len(all_relations)}")
    print(f"⏱️  Avg Time: {avg_time:.2f}ms per conversation")

    if all_tags:
        from collections import Counter
        tag_counts = Counter(all_tags)
        print(f"\n🔝 Top 10 Most Common Tags:")
        for tag, count in tag_counts.most_common(10):
            print(f"   {tag}: {count}")

    if all_relations:
        from collections import Counter
        relation_counts = Counter(all_relations)
        print(f"\n🔗 Top 10 Most Common Relations:")
        for relation, count in relation_counts.most_common(10):
            print(f"   {relation[0]} → {relation[1]}: {count}")
    
    if failed > 0:
        print("\n🔍 Failure Details:")
        for conversation_id, result in tracker.items():
            if not result.get("success", False):
                error = result.get("error", "Unknown")
                stage = result.get("stage", "unknown")
                print(f"  - {conversation_id}: {error} (stage: {stage})")
    
    print("\n" + "=" * 80)
    
    return managers  # ← 返回所有 managers


def main() -> None:
    parser = argparse.ArgumentParser(description="Add LoCoMo dataset into TridentMem")
    parser.add_argument("--data", default="dataset/locomo10.json")
    parser.add_argument("--config", default="evaluation/config.json")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    config_path = Path(args.config)
    config = load_config(config_path)
    
    if args.max_workers is not None:
        config["max_workers_add"] = args.max_workers
    if args.max_retries is not None:
        config["max_retries"] = args.max_retries
    
    dataset_path = Path(args.data)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    
    #  处理数据集，获取所有 managers
    managers = process_dataset(
        config,
        dataset,
        config.get("max_workers_add", 10),
        args.verbose or config.get("verbose", False),
        config.get("max_retries", 3)
    )

    # ==========  为每个 conversation 运行 Consolidation + 保存索引 ==========
    print("\n" + "=" * 80)
    print("🔧 Running Consolidation and Saving Indices...")
    print("=" * 80)
    
    for conversation_id, manager in managers.items():
        print(f"\n📦 Processing {conversation_id}...")
        
        # Consolidation
        try:
            consolidation_metrics = manager.trigger_consolidation(force=True)
            
            if consolidation_metrics:
                print(f"   Consolidation completed")
            else:
                print(f"  ⚠️  Consolidation skipped")
        
        except Exception as e:
            print(f"  ❌ Consolidation failed: {e}")
        
        # 保存索引
        try:
            index_path = get_index_path(conversation_id)
            manager.save_index(index_path)
            print(f"   Index saved to: {index_path}")
        
        except Exception as e:
            print(f"  ❌ Failed to save index: {e}")
        
        # 关闭 manager
        manager.close()
    
    print("\n" + "=" * 80)
    print(" All conversations processed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()