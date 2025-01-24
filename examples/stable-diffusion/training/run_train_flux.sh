#!/bin/bash




export MODEL_NAME="/host/mnt/ctrl/disk3/HF_models/FLUX.1-dev/"
export DATASET_NAME="/host/mnt/ctrl/disk2/dataset/naruto-blip-captions"


#export PT_HPU_LAZY_MODE=0
# --use_hpu_graphs_for_training \
# --use_hpu_graphs_for_inference \

#python train_flux_gaudi.py \
#accelerate launch train_flux_gaudi.py \
export PT_HPU_MAX_COMPOUND_OP_SYNC=1
export PT_HPU_MAX_COMPOUND_OP_SIZE=1
export PT_HPU_RECIPE_CACHE_CONFIG=/tmp/stdxl_recipe_cache,True,1024
accelerate launch --config_file=accelerate_config.yaml train_text_to_image_flux.py \
 --pretrained_model_name_or_path=$MODEL_NAME \
 --dataset_name=$DATASET_NAME \
 --resolution 512 \
 --crop_resolution 512 \
 --center_crop \
 --random_flip \
 --proportion_empty_prompts=0.2 \
 --train_batch_size 1 \
 --gradient_accumulation_steps=4 --gradient_checkpointing \
 --max_train_steps 10000 \
 --learning_rate 5e-06 \
 --guidance_scale=1 \
 --max_grad_norm 1 \
 --lr_scheduler constant \
 --lr_warmup_steps 0 \
 --output_dir flux_model_output \
 --gaudi_config_name ../Habana/stable-diffusion \
 --throughput_warmup_steps 3 \
 --dataloader_num_workers 8 \
 --sdp_on_bf16 \
 --bf16 \
 --dataset_sz=1221 \
 --validation_prompt="a cute Sundar Pichai creature" \
 --validation_epochs 4 \
 --checkpointing_steps 10000 \
 --logging_step 10 \
 --adjust_throughput \
 --seed=5451 2>&1 | tee recode_train.log
 


