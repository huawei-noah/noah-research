# SwiftMem

**SwiftMem** is a high-performance agentic memory management framework designed for API-level serving with intelligent three-layer retrieval architecture.

## Key Innovations

SwiftMem addresses critical limitations in existing memory systems through three core innovations:

### 1. **Temporal Episodic Indexing (L1)**
- **Solution**: Multi-time-range index supporting efficient queries across arbitrary time intervals

### 2. **Semantic Tag DAG Indexing (L2)**
- **Solution**: Hierarchical tag-based indexing with DAG structure, mimicking human memory organization

### 3. **Embedding Co-Consolidation (L3)**
- **Solution**: Offline consolidation that physically reorganizes embeddings by semantic similarity

---

## Project Structure

    SwiftMem/
        ├── dataset
        │   └── locomo-example.json
        ├── env_example
        ├── evaluation
        │   ├── add.py
        │   ├── config.json
        │   ├── evals.py
        │   ├── generate_scores.py
        │   ├── metrics
        │   │   ├── __init__.py
        │   │   ├── llm_judge.py
        │   │   └── utils.py
        │   ├── Readme.md
        │   ├── run.sh
        │   └── search.py
        ├── Readme.md
        ├── requirements.txt
        └── src
            ├── consolidation
            │   ├── __init__.py
            │   ├── clustering_strategy.py
            │   ├── embedding_consolidator.py
            │   └── types.py
            ├── core
            │   ├── base.py
            │   ├── config.py
            │   └── types.py
            ├── embeddings
            │   ├── __init__.py
            │   ├── embedding_client.py
            │   └── embedding_manager.py
            ├── llm
            │   ├── __init__.py
            │   └── client.py
            ├── memory
            │   ├── __init__.py
            │   ├── memory_manager.py
            │   ├── tag_generator.py
            │   └── tag_hierarchy.py
            ├── models
            │   ├── __init__.py
            │   ├── episode.py
            │   ├── message.py
            │   └── semantic.py
            ├── retrieval
            │   ├── __init__.py
            │   ├── multi_stage_retriever.py
            │   ├── query_tag_router.py
            │   ├── reranker.py
            │   └── types.py
            ├── segmentation
            │   ├── __init__.py
            │   └── episode_segmentor.py
            └── storage
                ├── __init__.py
                ├── backends
                │   ├── __init__.py
                │   ├── episode_storage.py
                │   └── semantic_storage.py
                ├── indexing
                │   ├── __init__.py
                │   ├── tag_dag_index.py
                │   ├── temporal_index.py
                │   └── vector_index.py
                └── unified_index.py

---

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

### Evaluation

See readme file in evaluation