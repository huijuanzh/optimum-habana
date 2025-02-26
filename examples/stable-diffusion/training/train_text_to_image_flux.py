#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
# limitations under the License.
"""
Fine-tuning script for Stable Diffusion models for text2image.
Adapted from the following sources:
https://github.com/huggingface/diffusers/blob/v0.25.1/examples/text_to_image/train_text_to_image_sdxl.py
"""
from typing import Any, Callable, Dict, List, Optional, Union

import argparse
import functools
import copy
import gc
import json
import logging
import math
import os
import random
import shutil
import time
from pathlib import Path

import accelerate
import datasets
import diffusers
import habana_frameworks.torch as htorch
import habana_frameworks.torch.core as htcore
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, GradientAccumulationPlugin
from datasets import load_dataset, Dataset
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    FluxTransformer2DModel,
    FluxPipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr, compute_loss_weighting_for_sd3, compute_density_for_timestep_sampling
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

from optimum.habana import GaudiConfig
from optimum.habana.accelerate import GaudiAccelerator
from optimum.habana.accelerate.utils.dataclasses import GaudiDistributedType
from optimum.habana.diffusers import (
    GaudiDDIMScheduler,
    GaudiEulerAncestralDiscreteScheduler,
    GaudiEulerDiscreteScheduler,
    GaudiFlowMatchEulerDiscreteScheduler,
    GaudiFluxPipeline,
)
from optimum.habana.utils import HabanaProfile, set_seed, to_gb_rounded


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.29.0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}


def save_model_card(
    repo_id: str,
    images=None,
    validation_prompt=None,
    base_model=str,
    dataset_name=str,
    repo_folder=None,
    vae_path=None,
):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
dataset: {dataset_name}
tags:
- flux
- flux-diffusers
- text-to-image
- diffusers
inference: true
---
    """
    model_card = f"""
# Text-to-image finetuning - {repo_id}

This pipeline was finetuned from **{base_model}** on the **{args.dataset_name}** dataset. Below are some example images generated with the finetuned pipeline using the following prompt: {validation_prompt}: \n
{img_str}

Special VAE used for training: {vae_path}.
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class =="T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


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
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing an image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
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
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="flux-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution."
        ),
    )
    parser.add_argument(
        "--crop_resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for cropping input images, all the images in the train/validation dataset will be resized to this"
            " resolution."
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
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
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
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
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="the FLUX.1 dev variant is a guidance distilled model",
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
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--timestep_bias_strategy",
        type=str,
        default="none",
        choices=["earlier", "later", "range", "none"],
        help=(
            "The timestep bias strategy, which may help direct the model toward learning low or high frequency details."
            " Choices: ['earlier', 'later', 'range', 'none']."
            " The default is 'none', which means no bias is applied, and training proceeds normally."
            " The value of 'later' will increase the frequency of the model's final training timesteps."
        ),
    )
    parser.add_argument(
        "--timestep_bias_multiplier",
        type=float,
        default=1.0,
        help=(
            "The multiplier for the bias. Defaults to 1.0, which means no bias is applied."
            " A value of 2.0 will double the weight of the bias, and a value of 0.5 will halve it."
        ),
    )
    parser.add_argument(
        "--timestep_bias_begin",
        type=int,
        default=0,
        help=(
            "When using `--timestep_bias_strategy=range`, the beginning (inclusive) timestep to bias."
            " Defaults to zero, which equates to having no specific bias."
        ),
    )
    parser.add_argument(
        "--timestep_bias_end",
        type=int,
        default=1000,
        help=(
            "When using `--timestep_bias_strategy=range`, the final timestep (inclusive) to bias."
            " Defaults to 1000, which is the number of timesteps that Stable Diffusion is trained on."
        ),
    )
    parser.add_argument(
        "--timestep_bias_portion",
        type=float,
        default=0.25,
        help=(
            "The portion of timesteps to bias. Defaults to 0.25, which 25% of timesteps will be biased."
            " A value of 0.5 will bias one half of the timesteps. The value provided for `--timestep_bias_strategy` determines"
            " whether the biased portions are in the earlier or later timesteps."
        ),
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
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
        "--bf16",
        action="store_true",
        default=False,
        help=("Whether to use bf16 mixed precision."),
    )
    parser.add_argument(
        "--sdp_on_bf16",
        action="store_true",
        default=False,
        help="Allow pyTorch to use reduced precision in the SDPA math backend",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
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
        "--image_save_dir",
        type=str,
        default="./stable-diffusion-generated-images",
        help="The directory where images will be saved.",
    )
    parser.add_argument(
        "--output_type",
        type=str,
        choices=["pil", "np"],
        default="pil",
        help="Whether to return PIL images or Numpy arrays.",
    )
    parser.add_argument(
        "--profiling_warmup_steps",
        default=0,
        type=int,
        help="Number of steps to ignore for profiling.",
    )
    parser.add_argument(
        "--profiling_steps",
        default=0,
        type=int,
        help="Number of steps to capture for profiling.",
    )
    parser.add_argument(
        "--logging_step",
        default=1,
        type=int,
        help="Print the loss for every logging_step.",
    )
    parser.add_argument(
        "--mediapipe",
        default="",
        type=str,
        help="Use gaudi2 HW mediapipe over regular dataloader. \
        case 1: nothing is passed to this argument -> regular torch dataloader is used\
        case 2: an empty or non existant path is passed -> images are dumped from dataset (passed in through dataset_name) in that location before first run \
        case 3: a non empty path is passed -> images from that location are used ",
    )
    parser.add_argument(
        "--adjust_throughput",
        default=False,
        action="store_true",
        help="Checkpoint saving takes a lot of time. Ignore time for checkpoint saving for throughput calculations",
    )
    parser.add_argument("--dataset_sz", type=int, default=32, help="set dataset size to avoid oom")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    return args


def _get_t5_prompt_embeds(
    tokenizer,
    text_encoder,
    prompt: Union[str, List[str]] = None,
    num_images_per_prompt: int = 1,
    max_sequence_length: int = 512,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    device = text_encoder.device
    dtype  = text_encoder.dtype

    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    with torch.no_grad():
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length- 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)[0]

        dtype = text_encoder.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _get_clip_prompt_embeds(
    tokenizer,
    text_encoder,
    prompt: Union[str, List[str]],
    num_images_per_prompt: int = 1,
    device: Optional[torch.device] = None,
):
    device = text_encoder.device 

    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    with torch.no_grad():
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {tokenizer.model_max_length} tokens: {removed_text}"
            )
        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds



# Adapted from diffusers.src.diffusers.pipelines.flux.pipeline_flux.encode_prompt
def encode_prompt(
    batch,
    text_encoders,
    tokenizers,
    proportion_empty_prompts,
    caption_column,
    is_train=True,
):
    prompt_embeds_list = []
    prompt_batch = batch[caption_column]
    

    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    pooled_prompt_embeds = _get_clip_prompt_embeds(tokenizers[0], text_encoders[0], captions)
    prompt_embeds = _get_t5_prompt_embeds(tokenizers[1], text_encoders[1], captions)

    #return {"prompt_embeds": prompt_embeds.cpu(), "pooled_prompt_embeds": pooled_prompt_embeds.cpu()}
    return {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds}


def compute_vae_encodings(batch, vae):
    images = batch.pop("pixel_values")
    pixel_values = torch.stack(list(images))
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    pixel_values = pixel_values.to(vae.device, dtype=vae.dtype)
    #print(f'baymax vae encodings pixel_values shape:{pixel_values.shape}')
    #show_traceback()

    with torch.no_grad():
        model_input = vae.encode(pixel_values).latent_dist.sample()
        #print(f' baymax vae encodings  model_input shape:{model_input.shape}')
    model_input = model_input * vae.config.scaling_factor

    # There might have slightly performance improvement
    return {"model_input": model_input}


def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)

def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents

def generate_timestep_weights(args, num_timesteps):
    weights = torch.ones(num_timesteps)

    # Determine the indices to bias
    num_to_bias = int(args.timestep_bias_portion * num_timesteps)

    if args.timestep_bias_strategy == "later":
        bias_indices = slice(-num_to_bias, None)
    elif args.timestep_bias_strategy == "earlier":
        bias_indices = slice(0, num_to_bias)
    elif args.timestep_bias_strategy == "range":
        # Out of the possible 1000 timesteps, we might want to focus on eg. 200-500.
        range_begin = args.timestep_bias_begin
        range_end = args.timestep_bias_end
        if range_begin < 0:
            raise ValueError(
                "When using the range strategy for timestep bias, you must provide a beginning timestep greater or equal to zero."
            )
        if range_end > num_timesteps:
            raise ValueError(
                "When using the range strategy for timestep bias, you must provide an ending timestep smaller than the number of timesteps."
            )
        bias_indices = slice(range_begin, range_end)
    else:  # 'none' or any other string
        return weights
    if args.timestep_bias_multiplier <= 0:
        return ValueError(
            "The parameter --timestep_bias_multiplier is not intended to be used to disable the training of specific timesteps."
            " If it was intended to disable timestep bias, use `--timestep_bias_strategy none` instead."
            " A timestep bias multiplier less than or equal to 0 is not allowed."
        )

    # Apply the bias
    weights[bias_indices] *= args.timestep_bias_multiplier

    # Normalize
    weights /= weights.sum()

    return weights


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Check for terminal SNR in combination with SNR Gamma
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=args.revision,
        variant=args.variant,
    )

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    gaudi_config = GaudiConfig.from_pretrained(args.gaudi_config_name)
    gaudi_config.use_torch_autocast = gaudi_config.use_torch_autocast or args.bf16

    ga_plugin = GradientAccumulationPlugin(num_steps=args.gradient_accumulation_steps, sync_each_batch=True)
    accelerator = GaudiAccelerator(
        #gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_accumulation_plugin = ga_plugin,
        mixed_precision="bf16" if gaudi_config.use_torch_autocast else "no",
        log_with=args.report_to,
        project_config=accelerator_project_config,
        force_autocast=gaudi_config.use_torch_autocast,
    )
    text_encoder_one = text_encoder_one.to(accelerator.device)
    text_encoder_two = text_encoder_two.to(accelerator.device)

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

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

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
            ).repo_id

    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )


    # Load scheduler and models
    noise_scheduler = GaudiFlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    if(args.sdp_on_bf16):
        torch._C._set_math_sdp_allow_fp16_bf16_reduction(True)

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if gaudi_config.use_torch_autocast:
        weight_dtype = torch.bfloat16

    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )

    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    transformer=transformer.to('hpu')

    # Freeze vae and text encoders.
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    # Set transformer as trainable.
    transformer.train()
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if gaudi_config.use_fused_adam:
        from habana_frameworks.torch.hpex.optimizers import FusedAdamW

        optimizer_class = FusedAdamW
    else:
        optimizer_class = torch.optim.AdamW

    if gaudi_config.use_fused_clip_norm:
        from habana_frameworks.torch.hpex.normalization import FusedClipNorm

        fused_clip_norm = FusedClipNorm(transformer.parameters(), args.max_grad_norm)

    # Optimizer creation
    params_to_optimize = transformer.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        if len(args.mediapipe) > 0:
            assert (
                args.resolution == args.crop_resolution
            ), f"To use hardware pipe, --resolution ({args.resolution}) must equal --crop_resolution ({args.crop_resolution})"
            if args.local_rank == 0:
                if not os.path.exists(args.mediapipe):
                    os.mkdir(args.mediapipe)
                if len(os.listdir(args.mediapipe)) == 0:
                    dataset = load_dataset(args.dataset_name, None)
                    with open(f"{args.mediapipe}/label.txt", "w") as f:
                        for idx, dt in enumerate(dataset["train"]):
                            dt["image"].save(f"{args.mediapipe}/{idx}.jpg")
                            f.write(dt["text"] + "\n")
            if accelerator.distributed_type != GaudiDistributedType.NO:
                torch.distributed.barrier()

            from media_pipe_imgdir import get_dataset_for_pipeline

            dt = get_dataset_for_pipeline(args.mediapipe)
            dataset = {"train": dt}
        else:
            # Downloading and loading a dataset from the hub.
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
            )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder
    
    ds = Dataset.from_dict(dataset['train'][:args.dataset_sz])
    dataset['train'] = ds


    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    text_encoder_one = text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two = text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    vae = vae.to(accelerator.device, dtype=weight_dtype)
    # Preprocessing the datasets.
    train_resize = transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR)
    train_crop = transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution)
    train_flip = transforms.RandomHorizontalFlip(p=1.0)
    train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    # Let's first compute all the embeddings so that we can free up the text encoders
    # from memory. We will pre-compute the VAE encodings too.
    text_encoders = [text_encoder_one, text_encoder_two]
    tokenizers = [tokenizer_one, tokenizer_two]

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        # image aug
        original_sizes = []
        all_images = []
        crop_top_lefts = []
        for image in images:
            original_sizes.append((image.height, image.width))
            image = train_resize(image)
            if args.random_flip and random.random() < 0.5:
                # flip
                image = train_flip(image)
            if args.center_crop:
                y1 = max(0, int(round((image.height - args.resolution) / 2.0)))
                x1 = max(0, int(round((image.width - args.resolution) / 2.0)))
                image = train_crop(image)
            else:
                y1, x1, h, w = train_crop.get_params(image, (args.resolution, args.resolution))
                image = crop(image, y1, x1, h, w)
            crop_top_left = (y1, x1)
            crop_top_lefts.append(crop_top_left)
            image = train_transforms(image)
            all_images.append(image)

        examples["original_sizes"] = original_sizes
        examples["crop_top_lefts"] = crop_top_lefts
        examples["pixel_values"] = all_images
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        train_dataset = dataset["train"]
        if len(args.mediapipe) == 0:
            # Set the training transforms
            train_dataset = train_dataset.with_transform(preprocess_train)

    compute_embeddings_fn = functools.partial(
        encode_prompt,
        text_encoders=text_encoders,
        tokenizers=tokenizers,
        proportion_empty_prompts=args.proportion_empty_prompts,
        caption_column=args.caption_column,
    )
    compute_vae_encodings_fn = functools.partial(compute_vae_encodings, vae=vae)

    # TODO : adding crop = (0,0) for now.
    # If we do random crop, we have to do this in mediapipe
    def attach_metadata(batch):
        import imagesize

        return {
            "original_sizes": imagesize.get(batch["image"]),
            "crop_top_lefts": (0, 0),
        }

    with accelerator.main_process_first():
        from datasets.fingerprint import Hasher

        # fingerprint used by the cache for the other processes to load the result
        # details: https://github.com/huggingface/diffusers/pull/4038#discussion_r1266078401
        new_fingerprint = Hasher.hash(args)
        train_dataset_with_embeddings = train_dataset.map(compute_embeddings_fn, batched=True, batch_size = 32, new_fingerprint=new_fingerprint)
        train_dataset_with_vae = train_dataset.map(
            compute_vae_encodings_fn,
            batched=True,
            batch_size=8
        )
        precomputed_dataset = concatenate_datasets(
            [train_dataset_with_embeddings, train_dataset_with_vae.remove_columns(["image", "text"])], axis=1
        )
        precomputed_dataset = precomputed_dataset.with_transform(preprocess_train)
        train_dataset = precomputed_dataset
        if len(args.mediapipe) > 0:
            train_dataset = train_dataset.map(attach_metadata, load_from_cache_file=False)

    #print('baymax debug compute embeddings done!')

    def collate_fn(examples):
        model_input = torch.stack([torch.tensor(example["model_input"]) for example in examples])
        original_sizes = [example["original_sizes"] for example in examples]
        crop_top_lefts = [example["crop_top_lefts"] for example in examples]
        prompt_embeds = torch.stack([torch.tensor(example["prompt_embeds"]) for example in examples])
        pooled_prompt_embeds = torch.stack([torch.tensor(example["pooled_prompt_embeds"]) for example in examples])

        return {
            "model_input": model_input,
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "original_sizes": original_sizes,
            "crop_top_lefts": crop_top_lefts,
        }

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    del text_encoders, tokenizers
    gc.collect()
    accelerator.wait_for_everyone()
    # Create EMA for the transformer.
    if args.use_ema:
        ema_transformer = FluxTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
            revision=args.revision,
            variant=args.variant,
        )
        ema_transformer = EMAModel(
            ema_transformer.parameters(),
            model_cls=FluxTransformer2DModel,
            model_config=ema_transformer.config,
        )

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_transformer.save_pretrained(os.path.join(output_dir, "transformer_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "transformer"))

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

        def load_model_hook(models, input_dir):
            '''
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "transformer_ema"), FluxTransformer2DModel)
                ema_transformer.load_state_dict(load_model.state_dict())
                ema_transformer.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = FluxTransformer2DModel.from_pretrained(input_dir, subfolder="transformer")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model
            '''
            transformer_ = None
            #if not accelerator.distributed_type == DistributedType.DEEPSPEED:
            if not accelerator.distributed_type == GaudiDistributedType.DEEPSPEED:
                if args.use_ema:
                    load_model = EMAModel.from_pretrained(os.path.join(input_dir, "transformer_ema"), FluxTransformer2DModel)
                    ema_transformer.load_state_dict(load_model.state_dict())
                    ema_transformer.to(accelerator.device)
                    del load_model
                while len(models) > 0:
                    model = models.pop()

                    if isinstance(unwrap_model(model), type(unwrap_model(flux_transformer))):
                        transformer_ = model  # noqa: F841
                    else:
                        raise ValueError(f"unexpected save model: {unwrap_model(model).__class__}")

            else:
                transformer_ = FluxTransformer2DModel.from_pretrained(input_dir, subfolder="transformer")  # noqa: F841

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    #transformer = transformer.to("hpu")
    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    if len(args.mediapipe) > 0:
        dataloader_params = {
            "batch_size": args.train_batch_size,
            "resolution": args.resolution,
        }
        from media_pipe_imgdir import MediaApiDataLoader

        train_dataloader = MediaApiDataLoader(train_dataset, **dataloader_params)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune-sdxl", config=vars(args))

    def unwrap_model(model, training=False):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        if not training:
            return model
        else:
            if accelerator.distributed_type == GaudiDistributedType.MULTI_HPU:
                kwargs = {}
                kwargs["gradient_as_bucket_view"] = True
                accelerator.ddp_handler = DistributedDataParallelKwargs(**kwargs)
            if args.use_hpu_graphs_for_training:
                htcore.hpu.ModuleCacher()(model=model, inplace=True)

    unwrap_model(model=transformer, training=True)
    hb_profiler = HabanaProfile(
        warmup=args.profiling_warmup_steps,
        active=args.profiling_steps,
        record_shapes=False,
    )
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    pipeline = None

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma    

    t0 = None
    t_start = time.perf_counter()
    train_loss = torch.tensor(0, dtype=torch.float, device="hpu")
    checkpoint_time = 0
    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss.zero_()
        if hb_profiler:
            hb_profiler.start()
        for step, batch in enumerate(train_dataloader):
            if t0 is None and global_step == args.throughput_warmup_steps:
                t0 = time.perf_counter()
            with accelerator.accumulate(transformer):
                # Move compute_vae_encoding here to reflect the transformed image input
                #model_input = compute_vae_encodings(batch["pixel_values"], vae)
                model_input = batch['model_input']
                model_input = model_input.to(accelerator.device, dtype=weight_dtype)

                vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
                bsz, num_channels_latents, height, width = model_input.shape
                latent_image_ids = _prepare_latent_image_ids(bsz, height // 2, width // 2, accelerator.device, model_input.dtype)

		        # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                packed_noisy_model_input = FluxPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=bsz,
                    num_channels_latents=num_channels_latents,
                    height=height,
                    width=height,
                )

                # handle guidance
                if accelerator.unwrap_model(transformer).config.guidance_embeds:
                    guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                    guidance = guidance.expand(bsz)
                else:
                    guidance = None

                # Predict the noise residual
                prompt_embeds = batch["prompt_embeds"].to(accelerator.device).to(dtype=transformer.dtype)

                pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(accelerator.device).to(dtype=transformer.dtype)
                #transformer_added_conditions.update({"text_embeds": pooled_prompt_embeds})
                text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)
                latent_image_ids = latent_image_ids.to(accelerator.device, dtype=weight_dtype)
                timesteps = timesteps.to(accelerator.device, dtype=weight_dtype)

                model_pred = transformer(
                    hidden_states = packed_noisy_model_input,
                    timestep = timesteps/1000,
                    guidance=guidance,
                    pooled_projections = pooled_prompt_embeds,
                    encoder_hidden_states = prompt_embeds,
                    txt_ids = text_ids,
                    img_ids = latent_image_ids,
                    return_dict=False,
                )[0]
		        # upscaling height & width as discussed in https://github.com/huggingface/diffusers/pull/9257#discussion_r1731108042
                model_pred = FluxPipeline._unpack_latents(
                    model_pred,
                    height=int(model_input.shape[2] * vae_scale_factor / 2),
                    width=int(model_input.shape[3] * vae_scale_factor / 2),
                    vae_scale_factor=vae_scale_factor,
                )
                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow matching loss
                target = noise - model_input
                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                htcore.mark_step()

                if accelerator.sync_gradients:
                    params_to_clip = transformer.parameters()
                    if gaudi_config.use_fused_clip_norm:
                        fused_clip_norm.clip_norm(params_to_clip)
                    else:
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                htcore.mark_step()
                hb_profiler.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                

                if accelerator.is_main_process or accelerator.distributed_type == GaudiDistributedType.DEEPSPEED:
                    if args.checkpointing_steps is not None and global_step % args.checkpointing_steps == 0:
                        t_chkpt_start = time.perf_counter()
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        t_chkpt_end = time.perf_counter()
                        checkpoint_time += t_chkpt_end - t_chkpt_start

                if (global_step - 1) % args.logging_step == 0 or global_step == args.max_train_steps:
                    train_loss_scalar = train_loss.item()
                    accelerator.log({"train_loss": train_loss_scalar}, step=global_step)

                    if args.gradient_accumulation_steps > 1:
                        logs = {
                            "step_loss": loss.item(),
                            "lr": lr_scheduler.get_last_lr()[0],
                            "mem_used": to_gb_rounded(htorch.hpu.memory_allocated()),
                        }
                    else:
                        logs = {
                            "step_loss": train_loss_scalar,
                            "lr": lr_scheduler.get_last_lr()[0],
                            "mem_used": to_gb_rounded(htorch.hpu.memory_allocated()),
                        }
                    progress_bar.set_postfix(**logs)
                train_loss.zero_()

            if global_step >= args.max_train_steps:
                break

        hb_profiler.stop()
        if accelerator.is_main_process:
            if args.validation_prompt is not None and (epoch + 1) % args.validation_epochs == 0:
                logger.info(
                    f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                    f" {args.validation_prompt}."
                )
                if args.use_ema:
                    # Store the transformer parameters temporarily and load the EMA parameters to perform inference.
                    ema_transformer.store(transformer.parameters())
                    ema_transformer.copy_to(transformer.parameters())

                # create pipeline
                pipeline = FluxPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        vae=vae,
                        transformer=unwrap_model(transformer),
                        revision=args.revision,
                        variant=args.variant,
                        )
                '''
                if pipeline is None:
                    #pipeline = GaudiFluxPipeline.from_pretrained(
                    pipeline = FluxPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        vae=vae,
                        transformer=unwrap_model(transformer),
                        revision=args.revision,
                        variant=args.variant,
                        #use_habana=True,
                        #use_hpu_graphs=args.use_hpu_graphs_for_inference,
                        #gaudi_config=args.gaudi_config_name,
                    )
                else:
                    # vae and text encoders are frozen, only need to update transformer
                    pipeline.transformer = unwrap_model(transformer)
                '''

                pipeline.scheduler = pipeline.scheduler.from_config(noise_scheduler.config)
                if args.prediction_type is not None:
                    scheduler_args = {"prediction_type": args.prediction_type}
                    pipeline.scheduler = pipeline.scheduler.from_config(pipeline.scheduler.config, **scheduler_args)
                pipeline = pipeline.to(accelerator.device)
                pipeline.set_progress_bar_config(disable=True)

                # run inference
                #generator = torch.Generator(device="cpu").manual_seed(args.seed) if args.seed else None
                if args.seed is not None:
                    if accelerator.device == torch.device("hpu"):
                        # torch.Generator() is unsupported on HPU
                        generator = set_seed(args.seed)
                    else:
                        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
                else:
                    generator = None
                pipeline_args = {"prompt": args.validation_prompt}

                with torch.autocast(
                    device_type="hpu",
                    dtype=torch.bfloat16,
                    enabled=gaudi_config.use_torch_autocast,
                ):
                    images = [
                        pipeline(**pipeline_args, generator=generator, num_inference_steps=25).images[0]
                        for _ in range(args.num_validation_images)
                    ]
                    for idx,image in enumerate(images):
                        file_name = f'./png/flux_epoch_{epoch}_idx_{idx}.png'
                        image.save(file_name)

                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        np_images = np.stack([np.asarray(img) for img in images])
                        tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
                    if tracker.name == "wandb":
                        tracker.log(
                            {
                                "validation": [
                                    wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                                    for i, image in enumerate(images)
                                ]
                            }
                        )
                #frer pipeline for memory 
                del pipeline
                gc.collect()

    if t0 is not None:
        duration = time.perf_counter() - t0 - (checkpoint_time if args.adjust_throughput else 0)
        ttt = time.perf_counter() - t_start
        throughput = (args.max_train_steps - args.throughput_warmup_steps) * total_batch_size / duration

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            logger.info(f"Throughput = {throughput} samples/s")
            logger.info(f"Train runtime = {duration} seconds")
            logger.info(f"Total Train runtime = {ttt} seconds")
            metrics = {
                "train_samples_per_second": throughput,
                "train_runtime": duration,
            }
            with open(f"{args.output_dir}/speed_metrics.json", mode="w") as file:
                json.dump(metrics, file)

        transformer = accelerator.unwrap_model(transformer)
        if args.use_ema:
            ema_transformer.copy_to(transformer.parameters())

        # Serialize pipeline.
        #pipeline = GaudiFluxPipeline.from_pretrained(
        pipeline = FluxPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            transformer=transformer,
            vae=vae,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
            #scheduler=noise_scheduler,
            #use_habana=True,
            #use_hpu_graphs=args.use_hpu_graphs_for_inference,
            #gaudi_config=args.gaudi_config_name,
        )
        if args.prediction_type is not None:
            scheduler_args = {"prediction_type": args.prediction_type}
            pipeline.scheduler = pipeline.scheduler.from_config(pipeline.scheduler.config, **scheduler_args)
        pipeline.save_pretrained(args.output_dir)

        # run inference
        images = []
        if args.validation_prompt and args.num_validation_images > 0:
            pipeline = pipeline.to(accelerator.device)
            #generator = torch.Generator(device="cpu").manual_seed(args.seed) if args.seed else None
            generator = torch.Generator("cpu").manual_seed(19)
            with torch.autocast(
                device_type="hpu",
                dtype=weight_dtype,
                enabled=gaudi_config.use_torch_autocast,
            ):
                images = [
                    pipeline(
                        args.validation_prompt,
                        height=1024,
                        width=1024,
                        guidance_scale=3.5,
                        num_inference_steps=32,
                        max_sequence_length = 512,
                        generator=generator,
                    ).images[0]
                    for _ in range(args.num_validation_images)
                ]
            # Save images in the specified directory if not None and if they are in PIL format
            if args.image_save_dir is not None:
                if args.output_type == "pil":
                    image_save_dir = Path(args.image_save_dir)
                    image_save_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Saving images in {image_save_dir.resolve()}...")
                    for i, image in enumerate(images):
                        image.save(image_save_dir / f"image_{epoch}_{i+1}.png")
                else:
                    logger.warning("--output_type should be equal to 'pil' to save images in --image_save_dir.")

            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images("test", np_images, epoch, dataformats="NHWC")
                if tracker.name == "wandb":
                    tracker.log(
                        {
                            "test": [
                                wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                                for i, image in enumerate(images)
                            ]
                        }
                    )

        if args.push_to_hub:
            save_model_card(
                repo_id=repo_id,
                images=images,
                validation_prompt=args.validation_prompt,
                base_model=args.pretrained_model_name_or_path,
                dataset_name=args.dataset_name,
                repo_folder=args.output_dir,
                vae_path=args.pretrained_vae_model_name_or_path,
            )
            
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)



