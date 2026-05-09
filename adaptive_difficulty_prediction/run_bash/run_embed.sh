#!/bin/bash
source ~/.bashrc
conda activate rl_test


model_name="Qwen/Qwen2.5-Math-1.5B-Instruct"
train_data_mode="questions" 

DATA_TRAIN_PATH="datasets/data_deepscaler_10240_train.pkl"
DATA_REF_PATH="datasets/data_deepscaler_10240_ref.pkl"

python save_embedding.py \
    --model_name $model_name \
    --left_padding \
    --train_data $train_data_mode \
    --data_train_path $DATA_TRAIN_PATH \
    --data_ref_path $DATA_REF_PATH