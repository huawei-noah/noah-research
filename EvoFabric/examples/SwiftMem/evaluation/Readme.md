# TridentMem LoCoMo Evaluation

This directory contains the evaluation pipeline for SwiftMem on the LoCoMo benchmark.

## Setup

1. Install required dependencies:
```bash
pip install nltk openai python-dotenv tqdm numpy
```

2. Set your OpenAI API key in `.env`

3. Download the LoCoMo dataset and place it in `dataset/locomo10.json`

## Quick Start

```
chmod +x evaluation/run.sh
./evaluation/run.sh
```

## Manual Execution

Step 1: Add Conversations

```
python evaluation/add.py \
    --data dataset/locomo10.json \
    --config evaluation/config.json \
    --max-workers 10
```

Step 2: Search and Generate Answers

```
python evaluation/search.py \
    --data dataset/locomo10.json \
    --config evaluation/config.json \
    --output results/search_results.json \
    --max-workers 10
```


Step 3: Evaluate Results

```
python evaluation/evals.py \
    --input results/search_results.json \
    --config evaluation/config.json \
    --output results/evaluation_results.json
```


Step 4: Generate Final Scores

```
python evaluation/generate_scores.py \
    --input results/evaluation_results.json \
    --config evaluation/config.json \
    --output-dir results
```


## Output

The evaluation pipeline generates:

results/search_results.json: Raw search results with generated answers
results/evaluation_results.json: Detailed evaluation metrics
results/final_scores.json: Summary scores by category
results/evaluation_report.md: Human-readable markdown repor