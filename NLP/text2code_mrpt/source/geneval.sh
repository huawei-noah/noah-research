#!/usr/bin/env bash

LOCAL_DIR_GEN="../generation_files"

BS=16
dtype='fp32'
incremental=False
greedy=False

while [ $# -gt 1 ]
do
    case "$1" in
        -mf|--model_family)
            model_family="$2"
            shift
            ;;
        -mp|--model_path)
            model_path="$2"
            shift
            ;;
		-incr|--incremental)
            incremental="$2"
            shift
            ;;
		-cgxp|--codegeex_path)
            codegeex_path="$2"
            shift
            ;;
		-dat|--dataset)
            dataset="$2"
            shift
            ;;
		-greedy|--greedy)
            greedy="$2"
            shift
            ;;
		*)
        echo "Unknown argument $2"
        exit 3
        ;;
    esac
    shift
done


model_name=$(basename ${model_path})

# prefix
if [[ "$model_name" == *"-prefix-"* ]]; then
	prefix=True
else
	prefix=False
fi

# separation
if [[ "$model_name" == *"-full-"* ]]; then
	replicated_tokens_map=True
	echo ">>> FULL Separation <<<"
	data_path="${model_path}"

elif [[ "$model_name" == *"-partial-"* ]]; then
	replicated_tokens_map=True
	echo ">>> PARTIAL Separation <<<"
	data_path="${model_path}"
	
else
	replicated_tokens_map=False
	data_path="none"
fi

# dataset
if [[ "$dataset" == "humaneval" ]]; then
	data_eval_file="${codegeex_path}/codegeex/benchmark/humaneval-x/python/data/humaneval_python.jsonl.gz"
	eval_script="${codegeex_path}/scripts/evaluate_${dataset}_x.sh"
elif [[ "$dataset" == "mbpp" ]]; then
	data_eval_file="${codegeex_path}/mbpp_test.jsonl"
	eval_script="${codegeex_path}/scripts/evaluate_${dataset}.sh"
else
	echo "Invalid dataset name"
	exit
fi


echo "${model_name} | prefix=${prefix} | replicated_tokens_map=${replicated_tokens_map} | incremental=${incremental} | data=${dataset}"

output_dir="${LOCAL_DIR_GEN}/${model_name}_${dataset}_incr${incremental}"


if [[ "$greedy" == True ]]; then
	k=0
	p=1.0
	temp=1.0
	num_return_sequences=1

	python generation.py \
		--torch_dtype="${dtype}" \
		--dataset_file="${data_eval_file}" \
		--model_name_or_path="${model_path}" \
		--max_seq_length=1024 \
		--output_dir="${output_dir}" \
		--greedy \
		--num_return_sequences="${num_return_sequences}" \
		--temperature="${temp}" \
		--k="${k}" \
		--p="${p}" \
		--batch_size="${BS}" \
		--seed=42 \
		--prefix_lm="${prefix}" \
		--model_type="${model_family}" \
		--replicated_tokens_map="${replicated_tokens_map}" \
		--data_path="${data_path}" \
		--incremental="${incremental}"

else
	k=0
	p=0.8
	temp=0.95
	num_return_sequences=200

	python generation.py \
		--torch_dtype="${dtype}" \
		--dataset_file="${data_eval_file}" \
		--model_name_or_path="${model_path}" \
		--max_seq_length=1024 \
		--output_dir="${output_dir}" \
		--num_return_sequences="${num_return_sequences}" \
		--temperature="${temp}" \
		--k="${k}" \
		--p="${p}" \
		--batch_size="${BS}" \
		--seed=42 \
		--prefix_lm="${prefix}" \
		--model_type="${model_family}" \
		--replicated_tokens_map="${replicated_tokens_map}" \
		--data_path="${data_path}" \
		--incremental="${incremental}"
fi


# Evaluate
bash ${eval_script} \
"${output_dir}/samples=${num_return_sequences}_${dtype}_bs=${BS}_t=${temp}_k=${k}_p=${p}.jsonl" \
"python" \
6 > "${output_dir}/samples=${num_return_sequences}_${dtype}_bs=${BS}_t=${temp}_k=${k}_p=${p}.out"

