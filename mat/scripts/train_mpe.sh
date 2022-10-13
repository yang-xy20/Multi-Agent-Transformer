#!/bin/sh
env="MPE"
scenario="push_ball"  # simple_speaker_listener # simple_reference simple_spread
num_landmarks=5
num_agents=5
algo="mat"
exp="push_ball_mat_5agents"
seed=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=0 python train/train_mpe.py --env_name ${env} --algorithm_name ${algo} \
--experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} \
--num_landmarks ${num_landmarks} --seed ${seed} --n_block 1 --n_embd 64 --n_training_threads 128 \
--n_rollout_threads 1 --num_mini_batch 5 --episode_length 120 --num_env_steps 20000000 --ppo_epoch 3 \
--clip_param 0.05 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --wandb_name "mapping" \
--user_name "yang-xy20"


# env="MPE"
# scenario="simple_spread"  # simple_speaker_listener # simple_reference simple_spread
# num_landmarks=5
# num_agents=5
# algo="mat"
# exp="simple_spread_mat_5agents"
# seed=1

# echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
# CUDA_VISIBLE_DEVICES=0 python train/train_mpe.py --env_name ${env} --algorithm_name ${algo} \
# --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} \
# --num_landmarks ${num_landmarks} --seed ${seed} --n_block 1 --n_embd 64 --n_training_threads 128 \
# --n_rollout_threads 1 --num_mini_batch 5 --episode_length 60 --num_env_steps 20000000 --ppo_epoch 3 \
# --clip_param 0.05 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --wandb_name "mapping" \
# --user_name "yang-xy20"