#!/bin/bash

# ------------------------- Common Prembule ------------------------- #
start_time=$(date +"%T")

#### get script_name ####
script_name=$(basename -- "$0")
l=`expr ${#script_name} - 3`
script_name=`expr "$script_name" | cut -c1-$l`

#### define directory where results will be stored ####
root_result_dir="results/$script_name"
root_result_dir="results/test_best"

#### empty existing result directory
# rm -rf $root_result_dir

# ------------------------- Shared Parameters ----------------------- #

num_MC_samples_acq=128
q=16

tuning_seed=0
num_starts=32
num_raw_samples=1024

num_trials=5
num_opt_steps=64
num_acq_steps=32

noise_free=0

early_stop=0


verbose=3
seed=0
num_initial=3

# ------------------------ Specific Parameters ---------------------- #
acq_funcs=(qUpperConfidenceBound qExpectedImprovement qProbabilityOfImprovement qSimpleRegret)
acq_func_kwargs_s=("{'beta':2}" "{}" "{}" "{}")
optimizers=(LBFSGB)
Ds==(16 40 60 80 100 120)

test_funcs=(Levy Ackley StyblinskiTang DixonPrice Powell)

save=1
save_losses=0
ignore_existing=0

covar='matern-5/2'
covar_kw="{}"

ub_offset=0
cuda='--cuda 0'
# cuda=''
# -------------------------- Run experiment ------------------------- #

N_optimizers=${#optimizers[@]}
N_test_funcs=${#test_funcs[@]}
N_dims=${#Ds[@]}
N_acq_funcs=${#acq_funcs[@]}

echo "$N_optimizers $N_test_funcs $N_dims  $N_acq_funcs"

for ((j=0;j<N_dims;j++));
      do
for ((i=0;i<N_test_funcs;i++));
    do  
    for ((k=0;k<N_acq_funcs;k++));
    do
        for ((h=0;h<N_optimizers;h++));
        do
            
          input_dim=${Ds[j]}
          test_func=${test_funcs[i]}
          optimizer=${optimizers[h]}
          
          acq_func=${acq_funcs[k]}
          acq_func_kwargs=${acq_func_kwargs_s[k]}
          echo "$((i + 1)) / $N_test_funcs"
          echo "$((j + 1)) / $N_dims"
          echo "$((k + 1)) / $N_acq_funcs"
          echo "$((h + 1)) / $N_optimizers"
          
          optimizer_kwargs="{}"
          scheduler=""
          scheduler_kwargs="{}"
          
          cmd="python -m bo_runner --negate 1 --test_func $test_func --acq_func $acq_func --acq_func_kwargs $acq_func_kwargs --optimizer $optimizer --optimizer_kwargs $optimizer_kwargs --num_starts $num_starts --num_raw_samples $num_raw_samples --num_trials $num_trials --num_opt_steps $num_opt_steps --num_acq_steps $num_acq_steps --q $q --input_dim $input_dim --verbose $verbose --early_stop $early_stop --save $save --seed $seed --root_result_dir $root_result_dir --num_MC_samples $num_MC_samples_acq --num_initial $num_initial $cuda $scheduler --scheduler_kwargs $scheduler_kwargs --noise_free $noise_free --covar $covar --covar_kw $covar_kw --save_losses $save_losses --ignore_existing $ignore_existing"
          echo $cmd
          echo
          $cmd
          done
        done
    done
done
          
end_time=$(date +"%T")
echo "Start time : $start_time"
echo "End time : $end_time"