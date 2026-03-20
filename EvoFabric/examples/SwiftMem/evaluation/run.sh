#!/bin/bash
# evaluation/run.sh

# Make sure you have set OPENAI_API_KEY in your environment
# export OPENAI_API_KEY=your_api_key_here

echo "🚀 Starting TridentMem LoCoMo Evaluation Pipeline"
echo "================================================"
echo ""

# Step 1: Add conversations to TridentMem
echo "📝 Step 1/4: Adding conversations to TridentMem..."
# python3 add.py --data ../dataset/locomo-example.json --config config.json
# python3 add.py --data ../dataset/locomo3.json --config config.json
python3 add.py --data ../dataset/locomo10.json --config config.json
if [ $? -ne 0 ]; then
    echo "❌ Error in add.py"
    exit 1
fi
echo "✅ Step 1 completed"
echo ""

# Step 2: Search and generate answers
echo "🔍 Step 2/4: Searching and generating answers..."
# python3 search.py --data ../dataset/locomo-example.json --config config.json --output results/search_results.json
# python3 search.py --data ../dataset/locomo3.json --config config.json --output results/search_results.json
python3 search.py --data ../dataset/locomo10.json --config config.json --output results/search_results.json
if [ $? -ne 0 ]; then
    echo "❌ Error in search.py"
    exit 1
fi
echo "✅ Step 2 completed"
echo ""

# Step 3: Evaluate results
echo "📊 Step 3/4: Evaluating results..."
python3 evals.py --input results/search_results.json --config config.json --output results/evaluation_results.json
if [ $? -ne 0 ]; then
    echo "❌ Error in evals.py"
    exit 1
fi
echo "✅ Step 3 completed"
echo ""

# Step 4: Generate final scores and report
echo "🎯 Step 4/4: Generating final scores and report..."
python3 generate_scores.py --input results/evaluation_results.json --config config.json --output-dir results
if [ $? -ne 0 ]; then
    echo "❌ Error in generate_scores.py"
    exit 1
fi
echo "✅ Step 4 completed"
echo ""

echo "================================================"
echo "🎉 Evaluation pipeline completed successfully!"
echo "📄 Check results/evaluation_report.md for detailed results"
echo "📊 Check results/final_scores.json for summary scores"