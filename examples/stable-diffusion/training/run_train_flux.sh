#!/bin/bash




export MODEL_NAME="/devops/huijuanz/project/test/fluxpipeline/FLUX.1-schnell/"
export DATASET_NAME="/checkpoint/libo/flux/dataset/lambda_dataset"

#  --enable_xformers_memory_efficient_attention \
#  --report_to="wandb" \
#accelerate launch train_flux_gaudi.py \
#    --pretrained_model_name_or_path=$MODEL_NAME \
#    --dataset_name=$DATASET_NAME \
#    --resolution=1024 --center_crop --random_flip \
#    --proportion_empty_prompts=0.2 \
#    --train_batch_size=1 \
#    --gradient_accumulation_steps=2 --gradient_checkpointing \
#    --max_train_steps=10000 \
#    --use_8bit_adam \
#    --learning_rate=1e-06 --lr_scheduler="constant" --lr_warmup_steps=0 \
#    --bf16 \
#    --gaudi_config_name Habana/stable-diffusion \
#    --dataset_sz=32 \
#    --validation_prompt="a cute Sundar Pichai creature" --validation_epochs 1 \
#    --checkpointing_steps=9900 \
#    --output_dir="bay_out" \
#   --use_hpu_graphs_for_training \
#   --use_hpu_graphs_for_inference \
#   --adjust_throughput \
#   --seed=5451 \


export PT_HPU_LAZY_MODE=0
# --use_hpu_graphs_for_training \
# --use_hpu_graphs_for_inference \

#python train_flux_gaudi.py \
#accelerate launch train_flux_gaudi.py \
python ../../gaudi_spawn.py --world_size 8 --master_port 8897 --use_deepspeed train_flux_gaudi.py \
 --pretrained_model_name_or_path=$MODEL_NAME \
 --dataset_name=$DATASET_NAME \
 --resolution 512 \
 --crop_resolution 512 \
 --center_crop \
 --random_flip \
 --proportion_empty_prompts=0.2 \
 --train_batch_size 1 \
 --max_train_steps 2500 \
 --learning_rate 1e-05 \
 --max_grad_norm 1 \
 --lr_scheduler constant \
 --lr_warmup_steps 0 \
 --output_dir flux_model_output \
 --gaudi_config_name Habana/stable-diffusion \
 --throughput_warmup_steps 3 \
 --dataloader_num_workers 8 \
 --sdp_on_bf16 \
 --bf16 \
 --dataset_sz=32 \
 --validation_prompt="a cute naruto creature" \
 --validation_epochs 2 \
 --checkpointing_steps 2500 \
 --logging_step 10 \
 --adjust_throughput \
 


