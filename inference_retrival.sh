#!/bin/bash

if [ $# -lt 4 ]; then
    echo "Ensemble Inference Based on Sample Retrieval"
    echo "Usage $0 <path to model> <save path> <inference config> <dataset dir>"
    exit 0
fi


# example
# torchrun --nproc_per_node=8 inference_classifier.py \
#     --model_path LimiX.ckpt \
#     --save_name result \
#     --inference_config_path config/dynamic_cls.json \
#     --data_dir my_cache/bcco_cls \
#     --debug

torchrun --nproc_per_node=8 inference_classifier.py \
    --model_path $1 \
    --save_name $2 \
    --inference_config_path $3 \
    --data_dir $4 \
    --debug