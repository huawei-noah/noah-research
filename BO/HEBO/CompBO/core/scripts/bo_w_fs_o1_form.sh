#!/bin/bash

# ------------------------- Common Prembule ------------------------- #
start_time=$(date +"%T")

#### get script_name ####
script_name=$(basename -- "$0")
l=$(expr ${#script_name} - 3)
script_name=$(expr "$script_name" | cut -c1-$l)

#### define directory where results will be stored ####
root_result_dir="results/$script_name"
root_result_dir="results/hyperparams_tunings"

#### empty existing result directory
# rm -rf $root_result_dir

# ------------------------- Shared Parameters ----------------------- #

num_MC_samples_acq=128
q=16

tuning_seed=0
num_starts=32
num_raw_samples=1024

num_trials=1
num_opt_steps=64
num_acq_steps=32

max_iter=30
initial_design_numdata=3

# acq_func_kwargs=("{}")

test_seed=0
test_num_trials=5

# ------------------------ Specific Parameters ---------------------- #
acq_funcs=(qFiniteSumUpperConfidenceBound qFiniteSumExpectedImprovement qFiniteSumProbabilityOfImprovement qFiniteSumSimpleRegret)
acq_func_kwargs_s=("{'beta':2}" "{}" "{}" "{}")
optimizers=(Adam Adagrad Rprop RMSprop)
Ds=(16 40 60 80 100 120)
test_funcs=(Levy StyblinskiTang Ackley DixonPrice Powell)

do_tuning=1
do_test=1
early_stop=0
do_random_search=0
ub_offset=0
cuda='--cuda 0'
# cuda=''

covar='matern-5/2'
covar_kw="{}"
# -------------------------- Run experiment ------------------------- #

N_optimizers=${#optimizers[@]}
N_test_funcs=${#test_funcs[@]}
N_dims=${#Ds[@]}
N_acq_funcs=${#acq_funcs[@]}

echo "$N_optimizers $N_test_funcs $N_dims  $N_acq_funcs"

for ((j = 0; j < N_dims; j++)); do
  for ((i = 0; i < N_test_funcs; i++)); do
    for ((k = 0; k < N_acq_funcs; k++)); do
      for ((h = 0; h < N_optimizers; h++)); do

        input_dim=${Ds[j]}
        test_func=${test_funcs[i]}
        optimizer=${optimizers[h]}

        acq_func=${acq_funcs[k]}
        acq_func_kwargs=${acq_func_kwargs_s[k]}
        echo "$((i + 1)) / $N_test_funcs"
        echo "$((j + 1)) / $N_dims"
        echo "$((k + 1)) / $N_acq_funcs"
        echo "$((h + 1)) / $N_optimizers"

        # ==== hyperparameters tuning ==== #
        if [ $do_tuning == 1 ]; then
          cmd="python opt_hyper_tuning.py --test_func $test_func --negate 1 --optimizer $optimizer --save 1 --num_trials $num_trials --seed $tuning_seed --root_result_dir $root_result_dir --save_losses 0 --num_acq_steps $num_acq_steps --num_opt_steps $num_opt_steps --input_dim $input_dim --q $q --num_MC_samples_acq $num_MC_samples_acq --num_starts $num_starts --num_raw_samples $num_raw_samples --acq_func $acq_func --acq_func_kwargs $acq_func_kwargs --max_iter $max_iter --initial_design_numdata $initial_design_numdata --verbose 1.5 --early_stop $early_stop $cuda --ub_offset $ub_offset --covar $covar --covar_kw $covar_kw"
          echo $cmd
          echo
          $cmd
        fi

        # ==== testing with best parameters ==== #
        # do_test=${do_tests[i]}
        # do_random_search=${do_random_searchs[i]}

        if [ $do_test == 1 ]; then
          selected_optimizers="['$optimizer']" # taskset --cpu-list 2-11
          cmd="python opt_hyper_test_bo.py --test_func $test_func --input_dim $input_dim --acq_func $acq_func --selected_optimizers $selected_optimizers --seed $test_seed --num_trials $test_num_trials --do_random_search $do_random_search --q $q --verbose 1.5 $cuda --ub_offset $ub_offset --covar $covar"
          echo
          echo $cmd
          $cmd
        fi
      done
    done
  done
done

end_time=$(date +"%T")
echo "Start time : $start_time"
echo "End time : $end_time"
