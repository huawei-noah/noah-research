#!/usr/bin/env bash

LOCAL_DIR_GEN="/nfs/aiml2/nlp_team/fenia/MRPT/HE_generations"
LOCAL_DIR="/nfs/aiml2/nlp_team/fenia/MRPT"


model_family=${1}

MODELS=(
"${model_family}-CodeCLM-100m"
"${model_family}-CodeCLM-partial-100m"
"${model_family}-CodeCLM-full-100m"
"${model_family}-CodeCLM-300m"
"${model_family}-CodeCLM-partial-300m"
"${model_family}-CodeCLM-full-300m"
)

k=0
p=0.8
temp=0.95

num_return_sequences=200
BS=16

GENERATE=true
EVAL=true

dtype='fp32'

for idx in "${MODELS[@]}"; do
	set -- $idx
	model=$1

	model_and_step="${model}"

	if [[ ${GENERATE} == "true" ]]; then

		if [[ "$model" == *"_prefix"* ]]; then
			prefix=True
		else
			prefix=False
		fi

		if [[ "$model" == *"_full_sep"* ]]; then
			replicated_tokens_map=True
			echo "--------------- FULL ---------------"
			data_path="/nfs/aiml2/nlp_team/fenia/MRPT/${model_family}_full_sep/"
		elif [[ "$model" == *"_partial_sep"* ]]; then
			replicated_tokens_map=True
			echo "--------------- PARTIAL ---------------"
			data_path="/nfs/aiml2/nlp_team/fenia/MRPT/${model_family}_partial_sep/"
		else
			replicated_tokens_map=False
			data_path="none"
		fi

		echo "========================================================================================================"
		echo "${model} | prefix=${prefix} | replicate_embeds=${replicated_tokens_map} | ${data_path}"
		echo "========================================================================================================"

		CUDA_VISIBLE_DEVICES=3 python generation.py \
			--torch_dtype=${dtype} \
			--human_eval_file="../data/human_eval/data/human_eval_internal.jsonl" \
			--model_name_or_path="${LOCAL_DIR}/${model_and_step}" \
			--max_seq_length=1024 \
			--output_dir="${LOCAL_DIR_GEN}/${model_and_step}" \
			--greedy=False \
			--num_return_sequences=${num_return_sequences} \
			--temperature=${temp} \
			--k=${k} \
			--p=${p} \
			--batch_size=${BS} \
			--seed=1234 \
			--prefix_lm=${prefix} \
			--model_type="${model_family}" \
			--replicated_tokens_map=${replicated_tokens_map} \
			--data_path=${data_path}
	fi

	# Evaluate
	python ../data/human_eval/evaluate_functional_correctness.py \
	"${LOCAL_DIR_GEN}/${model_and_step}/samples=${num_return_sequences}_${dtype}_bs=${BS}_t=${temp}_k=${k}_p=${p}.jsonl" \
	True

done

