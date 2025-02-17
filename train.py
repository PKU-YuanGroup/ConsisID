# Copyright 2024 ConsisID Authors and The HuggingFace Team. All rights reserved.
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

import logging
import math
import os
import pickle
import random
import shutil
import threading
from datetime import timedelta
from pathlib import Path

import accelerate
import insightface
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    DistributedType,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)
from consisid_eva_clip import create_model_and_transforms
from consisid_eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from insightface.app import FaceAnalysis
from models.consisid_utils import (
    compute_prompt_embeddings,
    prepare_rotary_positional_embeddings,
    process_face_embeddings,
    resize_numpy_image_long,
    tensor_to_pil,
)
from models.dataloader import ConsisID_Dataset, RandomSampler, SequentialSampler
from models.pipeline_cogvideox import CogVideoXPipeline
from models.pipeline_consisid import ConsisIDPipeline, draw_kps
from models.transformer_consisid import ConsisIDTransformer3DModel
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, T5EncoderModel
from transformers.utils import ContextManagers
from util.utils import get_args, pixel_values_to_pil, resize_mask

import diffusers
from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, cast_training_params, free_memory
from diffusers.utils import convert_unet_state_dict_to_peft, deprecate, export_to_video, is_wandb_available, load_image
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.video_processor import VideoProcessor


lock = threading.Lock()

if is_wandb_available():
    import wandb

logger = get_logger(__name__)


def log_validation(
    pipe,
    args,
    accelerator,
    pipeline_args,
    global_step,
    is_final_validation: bool = False,
    id_vit_hidden=None,
    id_cond=None,
    kps_cond=None,
):
    """
    Run the validation process for generating videos using the provided pipeline.

    This function performs a validation step by generating a specified number of videos using
    a given prompt and pipeline configuration. The generated videos are then logged to
    appropriate trackers (e.g., Weights and Biases).

    Args:
        pipe: The video generation pipeline object used to generate the videos.
        args: Arguments object containing various parameters such as the number of validation videos,
              output directory, and seed for reproducibility.
        accelerator: The accelerator object used to manage device placement and distributed training utilities.
        pipeline_args: Dictionary containing arguments for the video generation pipeline, including the prompt.
        is_final_validation (bool, optional): Flag indicating if this is the final validation step. Defaults to False.
        id_vit_hidden (optional): Hidden state input for identity preservation during video generation. Defaults to None.
        id_cond (optional): Condition input for identity preservation. Defaults to None.

    Returns:
        list: A list of generated video frames.
    """
    # Log the start of the validation process with the given prompt and number of videos
    logger.info(
        f"Running validation... \n Generating {args.num_validation_videos} videos with prompt: {pipeline_args['prompt']}."
    )

    # Set up scheduler arguments based on scheduler configuration
    scheduler_args = {}
    if "variance_type" in pipe.scheduler.config:
        variance_type = pipe.scheduler.config.variance_type

        # Adjust variance_type if necessary to ensure compatibility with the scheduler
        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    # Update the pipeline scheduler with the modified configuration
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, **scheduler_args)
    pipe = pipe.to(accelerator.device)

    # Initialize the random generator with a specified seed, if provided
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

    # Perform video generation without tracking gradients (inference mode)
    videos = []
    with torch.no_grad():
        for _ in range(args.num_validation_videos):
            # Generate video frames using the pipeline and provided conditions
            video = pipe(**pipeline_args, generator=generator, output_type="np",
                            id_vit_hidden=id_vit_hidden,
                            id_cond=id_cond,
                            kps_cond=kps_cond,
                        ).frames[0]
            videos.append(video)

    # Log generated videos to the appropriate tracker(s)
    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "wandb":
            video_filenames = []
            for i, video in enumerate(videos):
                # Create a unique filename for each video generated during validation
                prompt = (
                    pipeline_args["prompt"][:25]
                    .replace(" ", "_")
                    .replace("'", "_")
                    .replace('"', "_")
                    .replace("/", "_")
                )
                filename = os.path.join(args.output_dir, f"{global_step}_video_{i}_{prompt}.mp4")
                # Export the video as an MP4 file
                export_to_video(video, filename, fps=8)
                video_filenames.append(filename)

            # Log the generated videos to Weights and Biases (wandb) tracker
            tracker.log(
                {
                    phase_name: [
                        wandb.Video(filename, caption=f"{i}: {pipeline_args['prompt']}")
                        for i, filename in enumerate(video_filenames)
                    ]
                }
            )

    # Clean up the pipeline to free memory
    del pipe
    free_memory()

    # Return the list of generated videos
    return videos


def get_optimizer(args, params_to_optimize, use_deepspeed: bool = False):
    """
    Create and return an optimizer based on the specified arguments.

    Parameters:
    args : object
        An object containing various hyperparameters such as optimizer type, learning rate, betas, etc.
    params_to_optimize : iterable
        Parameters to be optimized by the optimizer.
    use_deepspeed : bool, optional (default=False)
        Flag to indicate whether to use the DeepSpeed optimizer.

    Returns:
    optimizer : Optimizer
        The optimizer instance based on the provided configuration.
    """
    # Use DeepSpeed optimizer if specified by the use_deepspeed flag
    if use_deepspeed:
        from accelerate.utils import DummyOptim

        return DummyOptim(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )

    # Define the list of supported optimizers
    supported_optimizers = ["adam", "adamw", "prodigy"]
    # Warn and default to AdamW if the provided optimizer is not supported
    if args.optimizer not in supported_optimizers:
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}. Supported optimizers include {supported_optimizers}. Defaulting to AdamW"
        )
        args.optimizer = "adamw"

    # Check if 8-bit Adam is being used with an unsupported optimizer
    if args.use_8bit_adam and args.optimizer.lower() not in ["adam", "adamw"]:
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'Adam' or 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    # Attempt to import bitsandbytes if 8-bit Adam is specified
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

    # Create optimizer instance based on the specified type
    if args.optimizer.lower() == "adamw":
        # Use AdamW from bitsandbytes if using 8-bit precision, otherwise use PyTorch's AdamW
        optimizer_class = bnb.optim.AdamW8bit if args.use_8bit_adam else torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    elif args.optimizer.lower() == "adam":
        # Use Adam from bitsandbytes if using 8-bit precision, otherwise use PyTorch's Adam
        optimizer_class = bnb.optim.Adam8bit if args.use_8bit_adam else torch.optim.Adam

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    elif args.optimizer.lower() == "prodigy":
        # Attempt to import the Prodigy optimizer library
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        # Warn if learning rate is lower than recommended for Prodigy
        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using Prodigy, it's generally better to set learning rate around 1.0"
            )

        # Create an instance of the Prodigy optimizer with additional specific arguments
        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    # Return the created optimizer
    return optimizer



def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    init_kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=args.nccl_timeout))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs, init_kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Prepare models and scheduler
    load_dtype = torch.bfloat16 if "5b" in args.pretrained_model_name_or_path.lower() else torch.float16

    transformer_additional_kwargs={
        'torch_dtype': load_dtype,
        'revision': args.revision,
        'variant': args.variant,
        'is_train_face': args.is_train_face,
        'is_kps': args.is_kps,
        'cross_attn_interval': args.cross_attn_interval,
        'cross_attn_dim_head': args.cross_attn_dim_head,
        'cross_attn_num_heads': args.cross_attn_num_heads,
        'LFE_id_dim': args.LFE_id_dim,
        'LFE_vit_dim': args.LFE_vit_dim,
        'LFE_depth': args.LFE_depth,
        'LFE_dim_head': args.LFE_dim_head,
        'LFE_num_heads': args.LFE_num_heads,
        'LFE_num_id_token': args.LFE_num_id_token,
        'LFE_num_querie': args.LFE_num_querie,
        'LFE_ff_mult': args.LFE_ff_mult,
        'LFE_num_scale': args.LFE_num_scale,
        'local_face_scale': args.local_face_scale,
    }

    transformer = ConsisIDTransformer3DModel.from_pretrained_cus(
        args.pretrained_model_name_or_path if args.pretrained_weight is None else args.pretrained_weight,
        subfolder="transformer",
        config_path=args.config_path,
        transformer_additional_kwargs=transformer_additional_kwargs,
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = accelerator.state.deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]


    scheduler = CogVideoXDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )

    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = T5EncoderModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
        )

        vae = AutoencoderKLCogVideoX.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
        )

        if args.enable_slicing:
            vae.enable_slicing()
        if args.enable_tiling:
            vae.enable_tiling()

        # detect face in the videos
        face_clip_model = None
        face_main_model = None
        face_helper_1     = None
        face_helper_2    = None
        if args.is_train_face:
            face_helper_1 = FaceRestoreHelper(
                upscale_factor=1,
                face_size=512,
                crop_ratio=(1, 1),
                det_model='retinaface_resnet50',
                save_ext='png',
                device=accelerator.device,
                model_rootpath=os.path.join(args.pretrained_model_name_or_path, "face_encoder")
            )
            face_helper_1.face_parse = None
            face_helper_1.face_parse = init_parsing_model(model_name='bisenet', device=accelerator.device, model_rootpath=os.path.join(args.pretrained_model_name_or_path, "face_encoder"))
            face_helper_1.face_det.eval()
            face_helper_1.face_parse.eval()

            model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', os.path.join(args.pretrained_model_name_or_path, "face_encoder", "EVA02_CLIP_L_336_psz14_s6B.pt"), force_custom_clip=True)
            face_clip_model = model.visual
            face_clip_model.eval()

            eva_transform_mean = getattr(face_clip_model, 'image_mean', OPENAI_DATASET_MEAN)
            eva_transform_std = getattr(face_clip_model, 'image_std', OPENAI_DATASET_STD)
            if not isinstance(eva_transform_mean, (list, tuple)):
                eva_transform_mean = (eva_transform_mean,) * 3
            if not isinstance(eva_transform_std, (list, tuple)):
                eva_transform_std = (eva_transform_std,) * 3
            eva_transform_mean = eva_transform_mean
            eva_transform_std = eva_transform_std

            device_id = accelerator.process_index % torch.cuda.device_count()
            face_main_model = FaceAnalysis(name='antelopev2', root=os.path.join(args.pretrained_model_name_or_path, "face_encoder"), providers=['CUDAExecutionProvider'], provider_options=[{"device_id": device_id}])
            face_helper_2 = insightface.model_zoo.get_model(f'{args.pretrained_model_name_or_path}/face_encoder/models/antelopev2/glintr100.onnx', providers=['CUDAExecutionProvider'], provider_options=[{"device_id": device_id}])

    # Freeze all the components
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    transformer.requires_grad_(False)

    VAE_SCALING_FACTOR = vae.config.scaling_factor
    VAE_SCALE_FACTOR_SPATIAL = 2 ** (len(vae.config.block_out_channels) - 1)
    RoPE_BASE_HEIGHT = transformer.config.sample_height * VAE_SCALE_FACTOR_SPATIAL
    RoPE_BASE_WIDTH = transformer.config.sample_width * VAE_SCALE_FACTOR_SPATIAL

    if args.is_train_face:
        face_clip_model.requires_grad_(False)
        face_helper_1.face_det.requires_grad_(False)
        face_helper_1.face_parse.requires_grad_(False)

    weight_dtype = torch.bfloat16
    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    # Move everything to device
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)

    if args.is_train_face:
        face_clip_model.to(accelerator.device, dtype=weight_dtype)
        face_helper_1.face_det.to(accelerator.device)
        face_helper_1.face_parse.to(accelerator.device)
        face_main_model.prepare(ctx_id=device_id if device_id is not None else 0, det_size=(640, 640))
        face_helper_2.prepare(ctx_id=device_id if device_id is not None else 0)
        free_memory()

    # enable gradient checkpointing
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params([transformer], dtype=torch.float32)

    # Create EMA for the transformer3d.
    if args.use_ema:
        ema_transformer3d = ConsisIDTransformer3DModel.from_pretrained_cus(
            args.pretrained_model_name_or_path if args.pretrained_weight is None else args.pretrained_weight,
            subfolder="transformer",
            config_path=args.config_path,
            transformer_additional_kwargs=transformer_additional_kwargs,
        )

        ema_transformer3d = EMAModel(ema_transformer3d.parameters(), model_cls=ConsisIDTransformer3DModel, model_config=ema_transformer3d.config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None

            if args.use_ema:
                ema_transformer3d.save_pretrained(os.path.join(output_dir, "transformer_ema"))

            for i, model in enumerate(models):
                if args.is_train_lora:
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                    model.save_face_modules(os.path.join(output_dir, "face_modules.pt"))
                else:
                    model.save_pretrained(os.path.join(output_dir, "transformer"))

                if weights:
                    weights.pop()

            if args.is_train_lora:
                ConsisIDPipeline.save_lora_weights(
                    output_dir,
                    transformer_lora_layers=transformer_lora_layers_to_save,
                )

            with open(os.path.join(output_dir, "sampler_pos_start.pkl"), 'wb') as file:
                pickle.dump([sampler._pos_start, first_epoch], file)

    def load_model_hook(models, input_dir):
        if args.use_ema:
            ema_path = os.path.join(input_dir, "transformer_ema")
            _, ema_kwargs = ConsisIDTransformer3DModel.load_config(ema_path, return_unused_kwargs=True)
            load_model = ConsisIDTransformer3DModel.from_pretrained_cus(
                input_dir,
                subfolder="transformer_ema",
                transformer_additional_kwargs=transformer_additional_kwargs,
            )

            load_model = EMAModel(load_model.parameters(), model_cls=ConsisIDTransformer3DModel, model_config=load_model.config)
            load_model.load_state_dict(ema_kwargs)

            ema_transformer3d.load_state_dict(load_model.state_dict())
            ema_transformer3d.to(accelerator.device)
            del load_model

        for i in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            if args.is_train_lora:
                lora_state_dict = ConsisIDPipeline.lora_state_dict(input_dir)
                transformer_state_dict = {
                    f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
                }
                transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
                incompatible_keys = set_peft_model_state_dict(model, transformer_state_dict, adapter_name="default")
                if incompatible_keys is not None:
                    # check only for unexpected keys
                    unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                    if unexpected_keys:
                        logger.warning(
                            f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                            f" {unexpected_keys}. "
                        )
                model.load_face_modules(os.path.join(input_dir, "face_modules.pt"))
            else:
                # load diffusers style into model
                load_model = ConsisIDTransformer3DModel.from_pretrained_cus(
                    input_dir,
                    subfolder="transformer",
                    transformer_additional_kwargs=transformer_additional_kwargs,
                )

            model.register_to_config(**load_model.config)
            model.load_state_dict(load_model.state_dict())
            del load_model

        pkl_path = os.path.join(input_dir, "sampler_pos_start.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as file:
                loaded_number, _ = pickle.load(file)
                sampler._pos_start = max(loaded_number - args.dataloader_num_workers * accelerator.num_processes * 2, 0)
            print(f"Load pkl from {pkl_path}. Get loaded_number = {loaded_number}.")

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params([model])

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # set trainable parameter
    trainable_modules = ["."]
    for name, param in transformer.named_parameters():
        for trainable_module_name in trainable_modules:
            if trainable_module_name in name:
                param.requires_grad = True
                break

    if args.is_train_face:
        unfreeze_modules = ["local_facial_extractor", "perceiver_cross_attention"]

        for module_name in unfreeze_modules:
            try:
                for param in getattr(transformer, module_name).parameters():
                    param.requires_grad = True
            except AttributeError:
                continue

        if args.is_train_lora:
            transformer_lora_config = LoraConfig(
                r=args.rank,
                lora_alpha=args.lora_alpha,
                init_lora_weights=True,
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                # Need to check 'exclude_modules'
                # exclude_modules=unfreeze_modules,
            )
            transformer.add_adapter(transformer_lora_config)

    # Optimization parameters
    if args.is_diff_lr:
        fuse_face_ca_params = list(filter(lambda p: p.requires_grad, transformer.perceiver_cross_attention.parameters()))
        fuse_face_ca_param_ids = {id(p) for p in fuse_face_ca_params}
        transformer_params = [p for p in transformer.parameters() if p.requires_grad and id(p) not in fuse_face_ca_param_ids]
        fuse_face_ca_params_with_lr = {"params": fuse_face_ca_params, "lr": args.learning_rate * 10}
        transformer_params_with_lr = {"params": transformer_params, "lr": args.learning_rate * 0.1}
        params_to_optimize = [fuse_face_ca_params_with_lr, transformer_params_with_lr]
    else:
        trainable_params = list(filter(lambda p: p.requires_grad, transformer.parameters()))
        transformer_parameters_with_lr = {"params": trainable_params, "lr": args.learning_rate}
        params_to_optimize = [transformer_parameters_with_lr]

    use_deepspeed_optimizer = (
        accelerator.state.deepspeed_plugin is not None
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    args.use_deepspeed = accelerator.state.deepspeed_plugin is not None
    optimizer = get_optimizer(args, params_to_optimize, use_deepspeed=use_deepspeed_optimizer)

    # Dataset and DataLoader
    train_dataset = ConsisID_Dataset(
        instance_data_root=args.instance_data_root,
        height=args.height,
        width=args.width,
        max_num_frames=args.max_num_frames,
        id_token=args.id_token,
        sample_stride=args.sample_stride,
        skip_frames_start_percent=args.skip_frames_start_percent,
        skip_frames_end_percent=args.skip_frames_end_percent,
        skip_frames_start=args.skip_frames_start,
        skip_frames_end=args.skip_frames_end,
        is_train_face=args.is_train_face,
        is_single_face=args.is_single_face,
        miss_tolerance=args.miss_tolerance,
        min_distance=args.min_distance,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        is_cross_face=args.is_cross_face,
        is_reserve_face=args.is_reserve_face
    )

    batch_sampler_generator = torch.Generator().manual_seed(args.seed)
    if args.is_shuffle_data:
        sampler = RandomSampler(train_dataset, generator=batch_sampler_generator)
    else:
        sampler = SequentialSampler(train_dataset, generator=batch_sampler_generator)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=sampler,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        prefetch_factor=2 if args.dataloader_num_workers != 0 else None,
        persistent_workers=True if args.dataloader_num_workers != 0 else False,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if use_deepspeed_scheduler:
        from accelerate.utils import DummyScheduler

        lr_scheduler = DummyScheduler(
            name=args.lr_scheduler,
            optimizer=optimizer,
            total_num_steps=args.max_train_steps * accelerator.num_processes,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        )
    else:
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_transformer3d.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = args.tracker_name or "consisid-ipt2v"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_trainable_parameters = sum(param.numel() for model in params_to_optimize for param in model["params"])

    logger.info("***** Running training *****")
    logger.info(f"  Num trainable parameters = {num_trainable_parameters}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if not args.resume_from_checkpoint:
        initial_global_step = 0
    else:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
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

            if args.is_accelerator_state_dict:
                # way_1
                accelerator.load_state(os.path.join(args.output_dir, path))
            else:
                # way_2
                pretrained_model_path = os.path.join(args.output_dir, path)

                if args.use_ema:
                    ema_path = os.path.join(pretrained_model_path, "transformer_ema")
                    _, ema_kwargs = ConsisIDTransformer3DModel.load_config(ema_path, return_unused_kwargs=True)
                    load_model = ConsisIDTransformer3DModel.from_pretrained_cus(
                        pretrained_model_path,
                        subfolder="transformer_ema",
                        transformer_additional_kwargs=transformer_additional_kwargs,
                    )

                    load_model = EMAModel(load_model.parameters(), model_cls=ConsisIDTransformer3DModel, model_config=load_model.config)
                    load_model.load_state_dict(ema_kwargs)

                    ema_transformer3d.load_state_dict(load_model.state_dict())
                    ema_transformer3d.to(accelerator.device)
                    del load_model

                load_model = ConsisIDTransformer3DModel.from_pretrained_cus(
                    pretrained_model_path,
                    subfolder="transformer",
                    transformer_additional_kwargs=transformer_additional_kwargs,
                )

                transformer.register_to_config(**load_model.config)
                m, u = accelerator.unwrap_model(transformer).load_state_dict(load_model.state_dict(), strict=False)
                print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
                del load_model

            global_step = int(path.split("-")[1])
            initial_global_step = global_step

            pkl_path = os.path.join(os.path.join(args.output_dir, path), "sampler_pos_start.pkl")
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as file:
                    _, first_epoch = pickle.load(file)
            else:
                first_epoch = global_step // num_update_steps_per_epoch
            print(f"Load pkl from {pkl_path}. Get first_epoch = {first_epoch}.")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # For DeepSpeed training
    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config

    def process_image(image, vae):
        image_noise_sigma = torch.normal(mean=-3.0, std=0.5, size=(1,), device=image.device)
        image_noise_sigma = torch.exp(image_noise_sigma).to(dtype=image.dtype)
        noisy_image = torch.randn_like(image) * image_noise_sigma[:, None, None, None, None]
        input_image = image + noisy_image
        image_latent_dist = vae.encode(input_image).latent_dist
        return image_latent_dist

    def encode_video(video, get_image_latent=True):
        video = video.to(accelerator.device, dtype=vae.dtype).unsqueeze(0) # [F, C, H, W] -> [B, F, C, H, W]
        video = video.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
        image = video[:, :, :1].clone()

        latent_dist = vae.encode(video).latent_dist

        if get_image_latent:
            image_latent_dist = process_image(image, vae)
        else:
            image_latent_dist = None

        return latent_dist, image_latent_dist

    def compute_prompt_embeddings_for_batch(prompts):
        return [
                compute_prompt_embeddings(
                tokenizer,
                text_encoder,
                [prompt],
                model_config.max_text_seq_length,
                accelerator.device,
                weight_dtype,
                requires_grad=False,
            )
            for prompt in prompts
        ]

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        sampler.generator = torch.Generator().manual_seed(args.seed + epoch)
        for step, batch in enumerate(train_dataloader):
            free_memory()
            models_to_accumulate = [transformer]

            with accelerator.accumulate(models_to_accumulate):
                if args.low_vram:
                    free_memory()
                    vae.to(accelerator.device)
                    text_encoder.to(accelerator.device)
                    if args.is_train_face:
                        face_clip_model.to(accelerator.device)
                        face_helper_1.face_det.to(accelerator.device)
                        face_helper_1.face_parse.to(accelerator.device)

                with torch.no_grad():
                    id_cond = None
                    id_vit_hidden = None
                    dense_masks = None
                    face_kps_latents = None

                    batch_instance_video = batch["instance_video"]
                    batch_instance_prompt = batch["instance_prompt"]

                    if args.is_train_face and args.enable_mask_loss and random.random() < 0.5:
                        enable_mask_loss_flag = True
                    else:
                        enable_mask_loss_flag = False

                    if args.is_train_face:
                        expand_face_imgs = batch['expand_face_imgs'] if batch['expand_face_imgs'] is not None else None                  # [B, self.max_frames, C=3, H=112, W=112]
                        original_face_imgs = batch['original_face_imgs'] if batch['original_face_imgs'] is not None else None            # [B, self.max_frames, C=3, H=112, W=112]
                        if args.is_reserve_face:
                            reserve_face_imgs = batch["reserve_face_imgs"]                                                               # torch.Size([2, 5, 3, 480, 480])

                        if args.enable_mask_loss and enable_mask_loss_flag:
                            dense_masks = batch['dense_masks_tensor'].to(memory_format=torch.contiguous_format).to(dtype=weight_dtype) if batch['dense_masks_tensor'] is not None else None   # B 1 T H W

                        if face_clip_model is not None and face_main_model is not None:
                            B, T = expand_face_imgs.shape[:2]
                            expand_valid_face_imgs = []
                            original_valid_face_imgs = []
                            B_valid_num = torch.zeros(B, dtype=torch.int32)
                            for i in range(B):
                                valid_mask = torch.any(torch.any(torch.any(expand_face_imgs[i] != 0, dim=1), dim=1), dim=1)
                                B_valid_num[i] = valid_mask.sum()
                                expand_valid_face_imgs.extend(expand_face_imgs[i][valid_mask])
                                original_valid_face_imgs.extend(original_face_imgs[i][valid_mask])
                            B_valid_num.to(accelerator.device)
                            expand_face_imgs = torch.stack(expand_valid_face_imgs)      # torch.Size([2, 3, 480, 480])
                            original_face_imgs = torch.stack(original_valid_face_imgs)  # torch.Size([2, 3, 480, 480])

                            align_crop_face_imgs = []
                            valid_id_conds = []
                            valid_id_vit_hiddens = []
                            valid_indices = []
                            face_kps_list = []
                            for idx, id_image in enumerate(expand_face_imgs):           # id_image: torch.Size([3, 480, 480])
                                align_crop_face_image = None
                                id_cond = None
                                id_vit_hidden = None

                                try:
                                    id_image = np.array(tensor_to_pil(id_image).convert("RGB"))
                                    id_image = resize_numpy_image_long(id_image, 1024)
                                    original_id_image = None
                                    if not args.is_align_face:
                                        original_id_image = np.array(tensor_to_pil(original_face_imgs[idx]).convert("RGB"))
                                        original_id_image = resize_numpy_image_long(original_id_image, 1024)
                                    id_cond, id_vit_hidden, align_crop_face_image, face_kps = process_face_embeddings(face_helper_1, face_clip_model, face_helper_2, eva_transform_mean, eva_transform_std, face_main_model, accelerator.device, weight_dtype, id_image, original_id_image=original_id_image, is_align_face=args.is_align_face)
                                except Exception as e:
                                    processed = False

                                    if args.is_reserve_face:
                                        print(f"Initial processing failed for image {idx}, attempting to process reserve images. Error: {e}")
                                        original_id_image = None
                                        if not args.is_align_face:
                                            original_id_image = np.array(tensor_to_pil(original_face_imgs[idx]).convert("RGB"))
                                            original_id_image = resize_numpy_image_long(original_id_image, 1024)
                                        for reserve_idx, reserve_id_image in enumerate(reserve_face_imgs[idx]):
                                            id_image = np.array(tensor_to_pil(reserve_id_image).convert("RGB"))
                                            id_image = resize_numpy_image_long(id_image, 1024)
                                            try:
                                                id_cond, id_vit_hidden, align_crop_face_image, face_kps = process_face_embeddings(
                                                    face_helper_1, face_clip_model, face_helper_2, eva_transform_mean, eva_transform_std,
                                                    face_main_model, accelerator.device, weight_dtype, id_image, original_id_image=original_id_image,
                                                    is_align_face=args.is_align_face
                                                )
                                                processed = True
                                                break
                                            except Exception as inner_e:
                                                print(f"Reserve image {reserve_idx} processing failed, trying next reserve image. Error: {inner_e}")
                                                continue

                                    if not processed:
                                        print(f"All reserve images failed, attempting to process frames from video. Error: {e}")
                                        Len_frame = batch_instance_video.shape[1]
                                        original_id_image = None
                                        if not args.is_align_face:
                                            original_id_image = np.array(tensor_to_pil(original_face_imgs[idx]).convert("RGB"))
                                            original_id_image = resize_numpy_image_long(original_id_image, 1024)
                                        for frame_idx in range(0, Len_frame, 5):
                                            try:
                                                temp_image = pixel_values_to_pil(batch_instance_video[idx].clone().cpu(), frame_index=frame_idx)
                                                id_image = np.array(temp_image.convert("RGB"))
                                                id_image = resize_numpy_image_long(id_image, 1024)
                                                id_cond, id_vit_hidden, align_crop_face_image, face_kps = process_face_embeddings(
                                                    face_helper_1, face_clip_model, face_helper_2, eva_transform_mean, eva_transform_std,
                                                    face_main_model, accelerator.device, weight_dtype, id_image, original_id_image=original_id_image,
                                                    is_align_face=args.is_align_face
                                                )
                                                processed = True
                                                break
                                            except Exception as inner_e:
                                                print(f"Frame {frame_idx} processing failed, trying next frame. Error: {inner_e}")
                                                continue

                                    if not processed:

                                        print(f"All attempts failed for image {idx}. No valid embeddings could be generated.")

                                if id_cond is not None:
                                    valid_id_conds.append(id_cond)
                                    valid_id_vit_hiddens.append(id_vit_hidden)
                                    align_crop_face_imgs.append(align_crop_face_image)
                                    valid_indices.append(idx)
                                    if args.is_kps:
                                        face_kps_list.append(face_kps)

                            if len(valid_id_conds) != 0:
                                valid_id_conds = torch.stack(valid_id_conds)
                                valid_id_vit_hiddens = [torch.cat(tensor_group, dim=0) for tensor_group in zip(*valid_id_vit_hiddens)]
                                align_crop_face_imgs = torch.stack(align_crop_face_imgs)

                                if len(valid_id_conds) != B:
                                    batch_instance_video = torch.stack([batch_instance_video[i] for i in valid_indices])
                                    batch_instance_prompt = [batch_instance_prompt[i] for i in valid_indices]
                                    if args.enable_mask_loss and enable_mask_loss_flag:
                                        dense_masks = torch.stack([dense_masks[i] for i in valid_indices])
                                    print(f"Adjusted batch size from {B} to {len(valid_id_conds)} due to mismatched valid_id_conds length.")

                    if not args.is_train_face or len(valid_indices) != 0:
                        video_latents_list = []
                        image_latents_list = []
                        for idx, (video, face_img) in enumerate(zip(batch_instance_video, align_crop_face_imgs)):
                            if args.is_train_face:
                                latent_dist, _ = encode_video(video, get_image_latent=False)  # input [F, C, H, W] -> [B, C, F, H, W]
                                vae_scale_factor_spatial = (VAE_SCALE_FACTOR_SPATIAL if vae is not None else 8)
                                video_processor = VideoProcessor(vae_scale_factor=vae_scale_factor_spatial)
                                tensor = face_img.cpu().detach()
                                tensor = tensor.squeeze()
                                tensor = tensor.permute(1, 2, 0)
                                tensor = tensor.numpy() * 255
                                tensor = tensor.astype(np.uint8)
                                pil_img = Image.fromarray(tensor)
                                if args.train_type == 'i2v':
                                    image = video_processor.preprocess(pil_img, height=args.height, width=args.width).to(memory_format=torch.contiguous_format).to(accelerator.device, dtype=weight_dtype) # torch [B, C, H, W]
                                    image_latent_dist = process_image(image.unsqueeze(2), vae) # torch [B, C, H, W] -> torch [B, C, 1, H, W]
                                if args.is_kps:
                                    valid_face_kps = draw_kps(pil_img, face_kps_list[idx])
                                    valid_face_kps = video_processor.preprocess(valid_face_kps, height=args.height, width=args.width).to(memory_format=torch.contiguous_format).to(accelerator.device, dtype=weight_dtype) # torch [B, C, H, W]
                                    face_kps_latent_dist = process_image(valid_face_kps.unsqueeze(2), vae) # torch [B, C, H, W] -> torch [B, C, 1, H, W]
                            else:
                                if args.train_type == 'i2v':
                                    latent_dist, image_latent_dist = encode_video(video, get_image_latent=True)
                                else:
                                    latent_dist, _ = encode_video(video, get_image_latent=False)

                            video_latents = latent_dist.sample() * VAE_SCALING_FACTOR
                            video_latents = video_latents.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W] -> [B, F, C, H, W]
                            video_latents_list.append(video_latents)

                            if args.train_type == 'i2v':
                                image_latents = image_latent_dist.sample() * VAE_SCALING_FACTOR  # torch.Size([1, 16, 1, 60, 90])
                                image_latents = image_latents.permute(0, 2, 1, 3, 4)  # [B, C, 1, H, W] -> [B, 1, C, H, W]  torch.Size([1, 1, 16, 60, 90])
                                if args.is_kps:
                                    face_kps_latents = face_kps_latent_dist.sample() * VAE_SCALING_FACTOR  # torch.Size([1, 16, 1, 60, 90])
                                    face_kps_latents = face_kps_latents.permute(0, 2, 1, 3, 4)  # [B, C, 1, H, W] -> [B, 1, C, H, W]  torch.Size([1, 1, 16, 60, 90])
                                    image_latents = torch.cat([image_latents, face_kps_latents], dim=1)
                                    padding_shape = (video_latents.shape[0], video_latents.shape[1] - 2, *video_latents.shape[2:])
                                else:
                                    padding_shape = (video_latents.shape[0], video_latents.shape[1] - 1, *video_latents.shape[2:])
                                latent_padding = image_latents.new_zeros(padding_shape)
                                image_latents = torch.cat([image_latents, latent_padding], dim=1)
                                if random.random() < args.noised_image_dropout:
                                    image_latents = torch.zeros_like(image_latents)
                                image_latents_list.append(image_latents)

                        video_latents = torch.cat(video_latents_list).to(memory_format=torch.contiguous_format).float()
                        video_latents = video_latents.to(dtype=weight_dtype)  # [B, F, C, H, W]

                        prompt_embeds = torch.cat(compute_prompt_embeddings_for_batch(batch_instance_prompt))

                        if args.train_type == 'i2v':
                            image_latents = torch.cat(image_latents_list).to(memory_format=torch.contiguous_format).float()
                            image_latents = image_latents.to(dtype=weight_dtype)  # [B, F, C, H, W]

                        # enable mask loss
                        if args.is_train_face and args.enable_mask_loss and enable_mask_loss_flag:
                            # way 1
                            dense_masks = dense_masks.unsqueeze(1)  # B F H W -> B 1 F H W
                            temp_video_latents = video_latents.clone().permute(0, 2, 1, 3, 4)  # B F C H W -> B C F H W
                            dense_masks = resize_mask(dense_masks, temp_video_latents, process_first_frame_only=False)  # torch.Size([2, 1, 13, 60, 90]),  B C F H W
                            dense_masks = dense_masks.repeat(1, temp_video_latents.shape[1], 1, 1, 1).permute(0, 2, 1, 3, 4).float()  # B C F H W -> B F C H W
                            dense_masks = dense_masks.reshape(video_latents.shape[0], -1)
                            dense_masks = dense_masks.to(accelerator.device)

                if args.low_vram:
                    vae.to('cpu')
                    text_encoder.to('cpu')
                    if args.is_train_face:
                        face_clip_model.to('cpu')
                        face_helper_1.face_det.to('cpu')
                        face_helper_1.face_parse.to('cpu')
                    free_memory()

                if not args.is_train_face or len(valid_indices) != 0:
                    batch_size, num_frames, num_channels, height, width = video_latents.shape

                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0, scheduler.config.num_train_timesteps, (batch_size,), device=video_latents.device
                    )
                    timesteps = timesteps.long()

                    # Sample noise that will be added to the latents
                    patch_size_t = model_config.patch_size_t if hasattr(model_config, "patch_size_t") else None
                    if patch_size_t is not None:
                        ncopy = video_latents.shape[1] % patch_size_t
                        # Copy the first frame ncopy times to match patch_size_t
                        first_frame = video_latents[:, :1, :, :, :]  # Get first frame [B, 1, C, H, W]
                        video_latents = torch.cat([first_frame.repeat(1, ncopy, 1, 1, 1), video_latents], dim=1)
                        assert video_latents.shape[1] % patch_size_t == 0
                    
                    # Padding image_latents to the same frame number as latent
                    padding_shape = (video_latents.shape[0], video_latents.shape[1] - image_latents.shape[1], * video_latents.shape[2:])
                    latent_padding = image_latents.new_zeros(padding_shape)
                    image_latents = torch.cat([image_latents, latent_padding], dim=1)

                    # Sample noise that will be added to the latents
                    noise = torch.randn_like(video_latents)

                    # Add noise to the model input according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_video_latents = scheduler.add_noise(video_latents, noise, timesteps)
                    if args.train_type == 'i2v':
                        noisy_model_input = torch.cat([noisy_video_latents, image_latents], dim=2)
                    else:
                        noisy_model_input = noisy_video_latents
                    model_config.patch_size_t if hasattr(model_config, "patch_size_t") else None,
                    ofs_embed_dim = model_config.ofs_embed_dim if hasattr(model_config, "ofs_embed_dim") else None,
                    ofs_emb = None if ofs_embed_dim is None else noisy_model_input.new_full((1,), fill_value=2.0)
                
                    # Prepare rotary embeds
                    image_rotary_emb = (
                        prepare_rotary_positional_embeddings(
                            height=height * VAE_SCALE_FACTOR_SPATIAL,
                            width=width * VAE_SCALE_FACTOR_SPATIAL,
                            num_frames=num_frames,
                            vae_scale_factor_spatial=VAE_SCALE_FACTOR_SPATIAL,
                            patch_size=model_config.patch_size,
                            patch_size_t=model_config.patch_size_t if hasattr(model_config, "patch_size_t") else None,
                            attention_head_dim=model_config.attention_head_dim,
                            device=accelerator.device,
                            base_height=RoPE_BASE_HEIGHT,
                            base_width=RoPE_BASE_WIDTH,
                        )
                        if model_config.use_rotary_positional_embeddings
                        else None
                    )
                else:
                    if args.train_type == 'i2v':
                        temp_dim = 32
                    else:
                        temp_dim = 16
                    batch_size = 1
                    patch_size_t = model_config.patch_size_t if hasattr(model_config, "patch_size_t") else None
                    if patch_size_t is not None:
                        noisy_video_latents = torch.zeros(1, 14, temp_dim, 60, 90).to(accelerator.device, dtype=weight_dtype)
                        noisy_model_input = torch.zeros(1, 14, temp_dim, 60, 90).to(accelerator.device, dtype=weight_dtype)
                        image_rotary_emb     = (torch.zeros(9450, 64).to(accelerator.device, dtype=weight_dtype), torch.zeros(9450, 64).to(accelerator.device, dtype=weight_dtype))
                    else:
                        noisy_video_latents = torch.zeros(1, 13, temp_dim, 60, 90).to(accelerator.device, dtype=weight_dtype)
                        noisy_model_input = torch.zeros(1, 13, temp_dim, 60, 90).to(accelerator.device, dtype=weight_dtype)
                        image_rotary_emb     = (torch.zeros(17550, 64).to(accelerator.device, dtype=weight_dtype), torch.zeros(17550, 64).to(accelerator.device, dtype=weight_dtype))
                    prompt_embeds = torch.zeros(1, 226, 4096).to(accelerator.device, dtype=weight_dtype)
                    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=video_latents.device)
                    timesteps = timesteps.long()
                    ofs_emb   = torch.tensor([2.]).to(accelerator.device, dtype=weight_dtype)
                    valid_id_conds       = torch.zeros(1, 1280).to(accelerator.device, dtype=weight_dtype)
                    valid_id_vit_hiddens = [torch.zeros([1, 577, 1024]).to(accelerator.device, dtype=weight_dtype)] * 5

                # Predict the noise residual
                model_output = transformer(
                        hidden_states=noisy_model_input,
                        encoder_hidden_states=prompt_embeds,
                        timestep=timesteps,
                        ofs=ofs_emb,
                        image_rotary_emb=image_rotary_emb,
                        return_dict=False,
                        id_cond=valid_id_conds if valid_id_conds is not None else None,
                        id_vit_hidden=valid_id_vit_hiddens if valid_id_vit_hiddens is not None else None,
                    )[0]
                model_pred = scheduler.get_velocity(model_output, noisy_video_latents, timesteps)

                alphas_cumprod = scheduler.alphas_cumprod[timesteps]
                weights = 1 / (1 - alphas_cumprod)
                while len(weights.shape) < len(model_pred.shape):
                    weights = weights.unsqueeze(-1)

                target = video_latents

                if not args.is_train_face or len(valid_indices) != 0:
                    loss = (weights * (model_pred - target) ** 2).reshape(batch_size, -1)
                    if args.is_train_face and args.enable_mask_loss and enable_mask_loss_flag:
                        loss = (loss * dense_masks).sum() / dense_masks.sum()
                    else:
                        loss = torch.mean(loss, dim=1).mean()
                else:
                    loss = (weights * (model_pred - model_pred) ** 2).reshape(batch_size, -1)
                    loss = torch.mean(loss, dim=1).mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                if accelerator.state.deepspeed_plugin is None:
                    optimizer.step()

                lr_scheduler.step()
                optimizer.zero_grad()

            loss = loss.detach()
            model_output = None
            model_pred = None
            noisy_model_input = None
            prompt_embeds = None
            timesteps = None
            image_rotary_emb = None
            valid_id_conds = None
            valid_id_vit_hiddens = None
            face_kps_latents = None
            del model_pred
            del model_output
            del noisy_model_input
            del prompt_embeds
            del timesteps
            del image_rotary_emb
            del valid_id_conds
            del valid_id_vit_hiddens
            del face_kps_latents
            free_memory()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_transformer3d.step(transformer.parameters())
                progress_bar.update(1)
                global_step += 1

                if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
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
                                logger.info(f"Removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")

                        if args.is_accelerator_state_dict:
                            accelerator.save_state(save_path)
                        else:
                            unwrap_model(transformer).save_pretrained(os.path.join(save_path, "transformer"))
                            if args.use_ema:
                                ema_transformer3d.save_pretrained(os.path.join(save_path, "transformer_ema"))

                        logger.info(f"Saved state to {save_path}")

                        if accelerator.is_main_process:
                            if args.validation_prompt is not None and args.is_validation:
                                if args.use_ema:
                                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                                    ema_transformer3d.store(transformer.parameters())
                                    ema_transformer3d.copy_to(transformer.parameters())

                                with torch.no_grad():
                                    # Create pipeline
                                    if args.train_type == 'i2v':
                                        pipe = ConsisIDPipeline.from_pretrained(
                                            args.pretrained_model_name_or_path,
                                            transformer=unwrap_model(transformer),
                                            scheduler=scheduler,
                                            revision=args.revision,
                                            variant=args.variant,
                                            torch_dtype=weight_dtype,
                                        )
                                    else:
                                        pipe = CogVideoXPipeline.from_pretrained(
                                            args.pretrained_model_name_or_path,
                                            transformer=unwrap_model(transformer),
                                            scheduler=scheduler,
                                            revision=args.revision,
                                            variant=args.variant,
                                            torch_dtype=weight_dtype,
                                        )

                                    validation_prompts = args.validation_prompt.split(args.validation_prompt_separator)
                                    validation_images = args.validation_images.split(args.validation_prompt_separator)

                                    for validation_image, validation_prompt in zip(validation_images, validation_prompts):
                                        if args.is_train_face:
                                            B_valid_num = torch.tensor([1], dtype=torch.int32)
                                            id_image = np.array(Image.open(validation_image).convert("RGB"))
                                            id_image = resize_numpy_image_long(id_image, 1024)
                                            id_cond, id_vit_hidden, align_crop_face_image, kps_cond = process_face_embeddings(face_helper_1, face_clip_model, face_helper_2, eva_transform_mean, eva_transform_std, face_main_model, accelerator.device, weight_dtype, id_image, original_id_image=id_image, is_align_face=args.is_align_face)
                                            tensor = align_crop_face_image.cpu().detach()
                                            tensor = tensor.squeeze()
                                            tensor = tensor.permute(1, 2, 0)
                                            tensor = tensor.numpy() * 255
                                            tensor = tensor.astype(np.uint8)
                                            pil_img = ImageOps.exif_transpose(Image.fromarray(tensor))
                                        else:
                                            pil_img = load_image(validation_image)

                                        if args.train_type == 'i2v':
                                            pipeline_args = {
                                                "image": pil_img,
                                                "prompt": validation_prompt,
                                                "guidance_scale": args.guidance_scale,
                                                "use_dynamic_cfg": args.use_dynamic_cfg,
                                                "height": args.height,
                                                "width": args.width,
                                            }
                                        else:
                                            pipeline_args = {
                                                "prompt": validation_prompt,
                                                "guidance_scale": args.guidance_scale,
                                                "use_dynamic_cfg": args.use_dynamic_cfg,
                                                "height": args.height,
                                                "width": args.width,
                                            }

                                        validation_outputs = log_validation(
                                            pipe=pipe,
                                            args=args,
                                            accelerator=accelerator,
                                            global_step=global_step,
                                            pipeline_args=pipeline_args,
                                            id_vit_hidden=id_vit_hidden if id_vit_hidden is not None else None,
                                            id_cond=id_cond if id_cond is not None else None,
                                            kps_cond=kps_cond if kps_cond is not None else None,
                                        )

                                del pipe
                                del validation_outputs
                                free_memory()
                                if args.use_ema:
                                    # Switch back to the original transformer3d parameters.
                                    ema_transformer3d.restore(transformer.parameters())

            try:
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            except AttributeError:
                logs = {"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, f"checkpoint-epoch-{global_step}")
        if args.is_accelerator_state_dict:
            accelerator.save_state(save_path)
        else:
            transformer.save_pretrained(os.path.join(save_path, "transformer"))
            if args.use_ema:
                ema_transformer3d.save_pretrained(os.path.join(save_path, "transformer_ema"))

        logger.info(f"Saved state to {save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    args = get_args()
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision
    main(args)