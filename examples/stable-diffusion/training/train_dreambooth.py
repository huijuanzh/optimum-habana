#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
"""
Training script for DreamBooth to Text-to-Image Diffusion Models
Adapted from the following source:
https://github.com/huggingface/peft/blob/608a90ded9985ee1c5912d738082bb1fd618902b/examples/stable_diffusion/train_dreambooth.py
"""

import argparse
import gc
import hashlib
import itertools
import logging
import math
import os
import threading
from pathlib import Path
from typing import Union

import datasets
import diffusers
import habana_frameworks.torch.core as htcore
import numpy as np
import psutil
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from habana_frameworks.torch.hpu import memory_stats
from huggingface_hub import HfApi
from peft import BOFTConfig, LoHaConfig, LoKrConfig, LoraConfig, OFTConfig, get_peft_model, tuners
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

from optimum.habana import GaudiConfig
from optimum.habana.accelerate import GaudiAccelerator
from optimum.habana.diffusers import GaudiStableDiffusionPipeline
from optimum.habana.transformers.trainer import _is_peft_model
from optimum.habana.utils import set_seed, warn0


logger = get_logger(__name__)

UNET_TARGET_MODULES = [
    "to_q",
    "to_k",
    "to_v",
    "proj",
    "proj_in",
    "proj_out",
    "conv",
    "conv1",
    "conv2",
    "conv_shortcut",
    "to_out.0",
    "time_emb_proj",
    "ff.net.2",
]

TEXT_ENCODER_TARGET_MODULES = ["fc1", "fc2", "q_proj", "k_proj", "v_proj", "out_proj"]


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def create_unet_adapter_config(
    args: argparse.Namespace,
) -> Union[LoraConfig, LoHaConfig, LoKrConfig, OFTConfig, BOFTConfig]:
    if args.adapter == "full":
        raise ValueError("Cannot create unet adapter config for full parameter")

    if args.adapter == "lora":
        config = LoraConfig(
            r=args.unet_r,
            lora_alpha=args.unet_alpha,
            target_modules=UNET_TARGET_MODULES,
            lora_dropout=args.unet_dropout,
            bias=args.unet_bias,
            init_lora_weights=True,
        )
    elif args.adapter == "loha":
        config = LoHaConfig(
            r=args.unet_r,
            alpha=args.unet_alpha,
            target_modules=UNET_TARGET_MODULES,
            rank_dropout=args.unet_rank_dropout,
            module_dropout=args.unet_module_dropout,
            use_effective_conv2d=args.unet_use_effective_conv2d,
            init_weights=True,
        )
    elif args.adapter == "lokr":
        config = LoKrConfig(
            r=args.unet_r,
            alpha=args.unet_alpha,
            target_modules=UNET_TARGET_MODULES,
            rank_dropout=args.unet_rank_dropout,
            module_dropout=args.unet_module_dropout,
            use_effective_conv2d=args.unet_use_effective_conv2d,
            decompose_both=args.unet_decompose_both,
            decompose_factor=args.unet_decompose_factor,
            init_weights=True,
        )
    elif args.adapter == "oft":
        config = OFTConfig(
            r=args.unet_r,
            target_modules=UNET_TARGET_MODULES,
            module_dropout=args.unet_dropout,
            init_weights=True,
            coft=args.unet_use_coft,
            eps=args.unet_eps,
            oft_block_size=0,
        )
    elif args.adapter == "boft":
        config = BOFTConfig(
            boft_block_size=args.unet_block_size,
            boft_block_num=args.unet_block_num,
            boft_n_butterfly_factor=args.unet_n_butterfly_factor,
            target_modules=UNET_TARGET_MODULES,
            boft_dropout=args.unet_dropout,
            bias=args.unet_bias,
        )
        from optimum.habana.peft.layer import GaudiBoftLinearForward

        tuners.boft.layer.Linear.forward = GaudiBoftLinearForward
        tuners.boft.layer._FBD_CUDA = False

    else:
        raise ValueError(f"Unknown adapter type {args.adapter}")

    return config


def create_text_encoder_adapter_config(
    args: argparse.Namespace,
) -> Union[LoraConfig, LoHaConfig, LoKrConfig, OFTConfig, BOFTConfig]:
    if args.adapter == "full":
        raise ValueError("Cannot create text_encoder adapter config for full parameter")

    if args.adapter == "lora":
        config = LoraConfig(
            r=args.te_r,
            lora_alpha=args.te_alpha,
            target_modules=TEXT_ENCODER_TARGET_MODULES,
            lora_dropout=args.te_dropout,
            bias=args.te_bias,
            init_lora_weights=True,
        )
    elif args.adapter == "loha":
        config = LoHaConfig(
            r=args.te_r,
            alpha=args.te_alpha,
            target_modules=TEXT_ENCODER_TARGET_MODULES,
            rank_dropout=args.te_rank_dropout,
            module_dropout=args.te_module_dropout,
            init_weights=True,
        )
    elif args.adapter == "lokr":
        config = LoKrConfig(
            r=args.te_r,
            alpha=args.te_alpha,
            target_modules=TEXT_ENCODER_TARGET_MODULES,
            rank_dropout=args.te_rank_dropout,
            module_dropout=args.te_module_dropout,
            decompose_both=args.te_decompose_both,
            decompose_factor=args.te_decompose_factor,
            init_weights=True,
        )
    elif args.adapter == "oft":
        config = OFTConfig(
            r=args.te_r,
            target_modules=TEXT_ENCODER_TARGET_MODULES,
            module_dropout=args.te_dropout,
            init_weights=True,
            coft=args.te_use_coft,
            eps=args.te_eps,
            oft_block_size=0,
        )
    elif args.adapter == "boft":
        config = BOFTConfig(
            boft_block_size=args.te_block_size,
            boft_block_num=args.te_block_num,
            boft_n_butterfly_factor=args.te_n_butterfly_factor,
            target_modules=TEXT_ENCODER_TARGET_MODULES,
            boft_dropout=args.te_dropout,
            bias=args.te_bias,
        )
        from optimum.habana.peft.layer import GaudiBoftLinearForward

        tuners.boft.layer.Linear.forward = GaudiBoftLinearForward
        tuners.boft.layer._FBD_CUDA = False
    else:
        raise ValueError(f"Unknown adapter type {args.adapter}")

    return config


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run dreambooth validation every X steps. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")

    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--wandb_key",
        type=str,
        default=None,
        help=("If report to option is set to wandb, api-key for wandb used for login to wandb "),
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default=None,
        help=("If report to option is set to wandb, project name in wandb for log tracking  "),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "bf16"],
        help=(
            "Whether to use mixed precision. Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "bf16"],
        help=("Choose prior generation precision between fp32 and bf16 (bfloat16)."),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--gaudi_config_name",
        type=str,
        default=None,
        help="Local path to the Gaudi configuration file or its name on the Hugging Face Hub.",
    )
    parser.add_argument(
        "--throughput_warmup_steps",
        type=int,
        default=0,
        help=(
            "Number of steps to ignore for throughput calculation. For example, with throughput_warmup_steps=N, the"
            " first N steps will not be considered in the calculation of the throughput. This is especially useful in"
            " lazy mode."
        ),
    )
    parser.add_argument(
        "--use_hpu_graphs_for_training",
        action="store_true",
        help="Use HPU graphs for training on HPU.",
    )
    parser.add_argument(
        "--use_hpu_graphs_for_inference",
        action="store_true",
        help="Use HPU graphs for inference on HPU.",
    )
    parser.add_argument(
        "--sdp_on_bf16",
        action="store_true",
        default=False,
        help="Allow pyTorch to use reduced precision in the SDPA math backend",
    )

    # Adapter arguments
    subparsers = parser.add_subparsers(dest="adapter")

    # Dummy subparser to train whole model
    subparsers.add_parser("full", help="Train full model without adapters")

    # LoRA adapter
    lora = subparsers.add_parser("lora", help="Use LoRA adapter")
    lora.add_argument("--unet_r", type=int, default=8, help="LoRA rank for unet")
    lora.add_argument("--unet_alpha", type=int, default=8, help="LoRA alpha for unet")
    lora.add_argument("--unet_dropout", type=float, default=0.0, help="LoRA dropout probability for unet")
    lora.add_argument(
        "--unet_bias",
        type=str,
        default="none",
        help="Bias type for LoRA. Can be 'none', 'all' or 'lora_only'",
    )
    lora.add_argument(
        "--te_r", type=int, default=8, help="LoRA rank for text_encoder, only used if `train_text_encoder` is True"
    )
    lora.add_argument(
        "--te_alpha",
        type=int,
        default=8,
        help="LoRA alpha for text_encoder, only used if `train_text_encoder` is True",
    )
    lora.add_argument(
        "--te_dropout",
        type=float,
        default=0.0,
        help="LoRA dropout probability for text_encoder, only used if `train_text_encoder` is True",
    )
    lora.add_argument(
        "--te_bias",
        type=str,
        default="none",
        help="Bias type for LoRA. Can be 'none', 'all' or 'lora_only', only used if `train_text_encoder` is True",
    )

    # LoHa adapter
    loha = subparsers.add_parser("loha", help="Use LoHa adapter")
    loha.add_argument("--unet_r", type=int, default=8, help="LoHa rank for unet")
    loha.add_argument("--unet_alpha", type=int, default=8, help="LoHa alpha for unet")
    loha.add_argument("--unet_rank_dropout", type=float, default=0.0, help="LoHa rank_dropout probability for unet")
    loha.add_argument(
        "--unet_module_dropout", type=float, default=0.0, help="LoHa module_dropout probability for unet"
    )
    loha.add_argument(
        "--unet_use_effective_conv2d",
        action="store_true",
        help="Use parameter effective decomposition in unet for Conv2d 3x3 with ksize > 1",
    )
    loha.add_argument(
        "--te_r", type=int, default=8, help="LoHa rank for text_encoder, only used if `train_text_encoder` is True"
    )
    loha.add_argument(
        "--te_alpha",
        type=int,
        default=8,
        help="LoHa alpha for text_encoder, only used if `train_text_encoder` is True",
    )
    loha.add_argument(
        "--te_rank_dropout",
        type=float,
        default=0.0,
        help="LoHa rank_dropout probability for text_encoder, only used if `train_text_encoder` is True",
    )
    loha.add_argument(
        "--te_module_dropout",
        type=float,
        default=0.0,
        help="LoHa module_dropout probability for text_encoder, only used if `train_text_encoder` is True",
    )

    # LoKr adapter
    lokr = subparsers.add_parser("lokr", help="Use LoKr adapter")
    lokr.add_argument("--unet_r", type=int, default=8, help="LoKr rank for unet")
    lokr.add_argument("--unet_alpha", type=int, default=8, help="LoKr alpha for unet")
    lokr.add_argument("--unet_rank_dropout", type=float, default=0.0, help="LoKr rank_dropout probability for unet")
    lokr.add_argument(
        "--unet_module_dropout", type=float, default=0.0, help="LoKr module_dropout probability for unet"
    )
    lokr.add_argument(
        "--unet_use_effective_conv2d",
        action="store_true",
        help="Use parameter effective decomposition in unet for Conv2d 3x3 with ksize > 1",
    )
    lokr.add_argument(
        "--unet_decompose_both", action="store_true", help="Decompose left matrix in kronecker product for unet"
    )
    lokr.add_argument(
        "--unet_decompose_factor", type=int, default=-1, help="Decompose factor in kronecker product for unet"
    )
    lokr.add_argument(
        "--te_r", type=int, default=8, help="LoKr rank for text_encoder, only used if `train_text_encoder` is True"
    )
    lokr.add_argument(
        "--te_alpha",
        type=int,
        default=8,
        help="LoKr alpha for text_encoder, only used if `train_text_encoder` is True",
    )
    lokr.add_argument(
        "--te_rank_dropout",
        type=float,
        default=0.0,
        help="LoKr rank_dropout probability for text_encoder, only used if `train_text_encoder` is True",
    )
    lokr.add_argument(
        "--te_module_dropout",
        type=float,
        default=0.0,
        help="LoKr module_dropout probability for text_encoder, only used if `train_text_encoder` is True",
    )
    lokr.add_argument(
        "--te_decompose_both",
        action="store_true",
        help="Decompose left matrix in kronecker product for text_encoder, only used if `train_text_encoder` is True",
    )
    lokr.add_argument(
        "--te_decompose_factor",
        type=int,
        default=-1,
        help="Decompose factor in kronecker product for text_encoder, only used if `train_text_encoder` is True",
    )
    # oft adapter
    oft = subparsers.add_parser("oft", help="Use Oft adapter")
    oft.add_argument("--unet_r", type=int, default=8, help="Oft rank for unet")
    oft.add_argument("--unet_dropout", type=float, default=0.0, help="Oft dropout probability for unet")
    oft.add_argument("--unet_use_coft", action="store_true", help="Using constrained OFT in unet")
    oft.add_argument("--unet_eps", type=float, default=0.0, help="The control strength of COFT for unet")
    oft.add_argument(
        "--te_r", type=int, default=8, help="Oft rank for text_encoder, only used if `train_text_encoder` is True"
    )
    oft.add_argument(
        "--te_dropout",
        type=float,
        default=0.0,
        help="Oft dropout probability for text_encoder, only used if `train_text_encoder` is True",
    )
    oft.add_argument(
        "--te_use_coft",
        action="store_true",
        help="Using constrained OFT in text_encoder, only used if `train_text_encoder` is True",
    )
    oft.add_argument(
        "--te_eps",
        type=float,
        default=0.0,
        help="The control strength of COFT for text_encoder, only used if `train_text_encoder` is True",
    )

    # boft adapter
    boft = subparsers.add_parser("boft", help="Use Boft adapter")
    boft.add_argument("--unet_block_size", type=int, default=8, help="Boft block_size for unet")
    boft.add_argument("--unet_block_num", type=int, default=0, help="Boft block_num for unet")
    boft.add_argument("--unet_n_butterfly_factor", type=int, default=1, help="Boft n_butterfly_factor for unet")
    boft.add_argument("--unet_dropout", type=float, default=0.1, help="Boft dropout for unet")
    boft.add_argument("--unet_bias", type=str, default="boft_only", help="Boft bias for unet")
    boft.add_argument(
        "--te_block_size",
        type=int,
        default=8,
        help="Boft block_size for text_encoder,only used if `train_text_encoder` is True",
    )
    boft.add_argument(
        "--te_block_num",
        type=int,
        default=0,
        help="Boft block_num for text_encoder,only used if `train_text_encoder` is True",
    )
    boft.add_argument(
        "--te_n_butterfly_factor",
        type=int,
        default=1,
        help="Boft n_butterfly_factor for text_encoder,only used if `train_text_encoder` is True",
    )
    boft.add_argument(
        "--te_dropout",
        type=float,
        default=0.1,
        help="Boft dropout for text_encoder,only used if `train_text_encoder` is True",
    )
    boft.add_argument(
        "--te_bias",
        type=str,
        default="boft_only",
        help="Boft bias for text_encoder, only used if `train_text_encoder` is True",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warn0("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            warn0("You need not use --class_prompt without --with_prior_preservation.")

    return args


# Converting Bytes to Megabytes
def b2mb(x):
    return int(x / 2**20)


# This context manager is used to track the peak memory usage of the process
class TorchTracemalloc:
    def __enter__(self):
        gc.collect()
        mem_stats = memory_stats()

        self.begin = mem_stats["InUse"]
        self.process = psutil.Process()

        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001) # 1msec

            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        mem_stats = memory_stats()

        self.end = mem_stats["InUse"]
        self.peak = mem_stats["MaxInUse"]
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example


def collate_fn(examples, with_prior_preservation=False):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }
    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    gaudi_config = GaudiConfig.from_pretrained(args.gaudi_config_name)
    gaudi_config.use_torch_autocast = gaudi_config.use_torch_autocast or args.mixed_precision == "bf16"
    accelerator = GaudiAccelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
        force_autocast=gaudi_config.use_torch_autocast,
    )
    if args.report_to == "wandb":
        import wandb

        wandb.login(key=args.wandb_key)
        wandb.init(project=args.wandb_project_name)
    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if args.sdp_on_bf16:
        torch._C._set_math_sdp_allow_fp16_bf16_reduction(True)

    # Generate class images if prior preservation is enabled.
    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            torch_dtype = torch.bfloat16 if accelerator.device.type == "hpu" else torch.float32
            if args.prior_generation_precision == "fp32":
                torch_dtype = torch.float32
            elif args.prior_generation_precision == "bf16":
                torch_dtype = torch.bfloat16
            pipeline = GaudiStableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=args.revision,
                use_hpu_graphs=args.use_hpu_graphs_for_inference,
                use_habana=True,
                gaudi_config=gaudi_config,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            api = HfApi(token=args.hub_token)
            # Create repo (repo_name from args or inferred)
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            repo_id = api.create_repo(repo_name, exist_ok=True).repo_id

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )  # DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )
    if args.adapter != "full":
        config = create_unet_adapter_config(args)
        unet = get_peft_model(unet, config)
        unet.print_trainable_parameters()
    unet.to(accelerator.device)
    vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)
    elif args.train_text_encoder and args.adapter != "full":
        config = create_text_encoder_adapter_config(args)
        text_encoder = get_peft_model(text_encoder, config)
        text_encoder.print_trainable_parameters()
    text_encoder.to(accelerator.device)
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder and not args.adapter != "full":
            text_encoder.gradient_checkpointing_enable()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    if gaudi_config.use_fused_adam:
        from habana_frameworks.torch.hpex.optimizers import FusedAdamW

        optimizer_class = FusedAdamW
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters()
    )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=1,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    def unwrap_model(model, training=False):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        if not training:
            return model
        else:
            if accelerator.distributed_type == DistributedType.MULTI_HPU:
                kwargs = {}
                kwargs["gradient_as_bucket_view"] = True
                accelerator.ddp_handler = DistributedDataParallelKwargs(**kwargs)
            if args.use_hpu_graphs_for_training:
                if _is_peft_model(model):
                    base_model = model.get_base_model()
                    htcore.hpu.ModuleCacher()(model=base_model, inplace=True)
                else:
                    htcore.hpu.ModuleCacher()(model=model, inplace=True)
            return model

    unwrap_model(model=unet, training=True)
    if args.train_text_encoder:
        unwrap_model(model=text_encoder, training=True)
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1]
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(args.output_dir, path))
        global_step = int(path.split("-")[1])

        resume_global_step = global_step * args.gradient_accumulation_steps
        first_epoch = resume_global_step // num_update_steps_per_epoch
        resume_step = resume_global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder.train()
        with TorchTracemalloc() as tracemalloc:
            for step, batch in enumerate(train_dataloader):
                # Skip steps until we reach the resumed step
                if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                        if args.report_to == "wandb":
                            accelerator.print(progress_bar)
                    continue

                with accelerator.accumulate(unet):
                    # Convert images to latent space
                    latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * 0.18215

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                    )
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                    # Predict the noise residual
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    if args.with_prior_preservation:
                        # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                        model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                        target, target_prior = torch.chunk(target, 2, dim=0)

                        # Compute instance loss
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                        # Compute prior loss
                        prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                        # Add the prior loss to the instance loss.
                        loss = loss + args.prior_loss_weight * prior_loss
                    else:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    accelerator.backward(loss)
                    htcore.mark_step()
                    if accelerator.sync_gradients:
                        params_to_clip = (
                            itertools.chain(unet.parameters(), text_encoder.parameters())
                            if args.train_text_encoder
                            else unet.parameters()
                        )
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    htcore.mark_step()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    if args.report_to == "wandb":
                        accelerator.print(progress_bar)
                    global_step += 1

                    # if global_step % args.checkpointing_steps == 0:
                    #     if accelerator.is_main_process:
                    #         save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    #         accelerator.save_state(save_path)
                    #         logger.info(f"Saved state to {save_path}")

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if (
                    args.validation_prompt is not None
                    and (step + num_update_steps_per_epoch * epoch) % args.validation_steps == 0
                ):
                    logger.info(
                        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                        f" {args.validation_prompt}."
                    )
                    # create pipeline
                    pipeline = GaudiStableDiffusionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        safety_checker=None,
                        revision=args.revision,
                        use_hpu_graphs=args.use_hpu_graphs_for_inference,
                        use_habana=True,
                        gaudi_config=gaudi_config,
                    )
                    # set `keep_fp32_wrapper` to True because we do not want to remove
                    # mixed precision hooks while we are still training
                    pipeline.unet = accelerator.unwrap_model(unet, keep_fp32_wrapper=True)
                    pipeline.text_encoder = accelerator.unwrap_model(text_encoder, keep_fp32_wrapper=True)
                    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
                    pipeline = pipeline.to(accelerator.device)
                    pipeline.set_progress_bar_config(disable=True)

                    # Set evaliation mode
                    pipeline.unet.eval()
                    pipeline.text_encoder.eval()

                    # run inference
                    if args.seed is not None:
                        if accelerator.device == torch.device("hpu"):
                            # torch.Generator() is unsupported on HPU
                            generator = set_seed(args.seed)
                        else:
                            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
                    else:
                        generator = None
                    images = []
                    for _ in range(args.num_validation_images):
                        image = pipeline(args.validation_prompt, num_inference_steps=25, generator=generator).images[0]
                        images.append(image)

                    for tracker in accelerator.trackers:
                        if tracker.name == "tensorboard":
                            np_images = np.stack([np.asarray(img) for img in images])
                            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
                        if tracker.name == "wandb":
                            import wandb

                            tracker.log(
                                {
                                    "validation": [
                                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                                        for i, image in enumerate(images)
                                    ]
                                }
                            )

                    # Set evaliation mode
                    pipeline.unet.train()
                    if args.train_text_encoder:
                        pipeline.text_encoder.train()

                    del pipeline

                if global_step >= args.max_train_steps:
                    break
        # Printing the HPU memory usage details such as allocated memory, peak memory, and total memory usage
        accelerator.print(f"HPU Memory before entering the train : {b2mb(tracemalloc.begin)}")
        accelerator.print(f"HPU Memory consumed at the end of the train (end-begin): {tracemalloc.used}")
        accelerator.print(f"HPU Peak Memory consumed during the train (max-begin): {tracemalloc.peaked}")
        accelerator.print(
            f"HPU Total Peak Memory consumed during the train (max): {tracemalloc.peaked + b2mb(tracemalloc.begin)}"
        )

        accelerator.print(f"CPU Memory before entering the train : {b2mb(tracemalloc.cpu_begin)}")
        accelerator.print(f"CPU Memory consumed at the end of the train (end-begin): {tracemalloc.cpu_used}")
        accelerator.print(f"CPU Peak Memory consumed during the train (max-begin): {tracemalloc.cpu_peaked}")
        accelerator.print(
            f"CPU Total Peak Memory consumed during the train (max): {tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)}"
        )

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.adapter != "full":
            unwarpped_unet = unwrap_model(unet)
            unwarpped_unet.save_pretrained(
                os.path.join(args.output_dir, "unet"), state_dict=accelerator.get_state_dict(unet)
            )
            if args.train_text_encoder:
                unwarpped_text_encoder = unwrap_model(text_encoder)
                unwarpped_text_encoder.save_pretrained(
                    os.path.join(args.output_dir, "text_encoder"),
                    state_dict=accelerator.get_state_dict(text_encoder),
                )
        else:
            pipeline = GaudiStableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=unwrap_model(unet),
                text_encoder=unwrap_model(text_encoder),
                revision=args.revision,
                use_hpu_graphs=args.use_hpu_graphs_for_inference,
                use_habana=True,
                gaudi_config=gaudi_config,
            )
            pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
            api.upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                run_as_future=True,
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
