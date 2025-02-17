# export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE="offline"
# export WANDB_MODE="online"

# For training from scratch
# export MODEL_PATH="THUDM/CogVideoX-5b-I2V"    # "THUDM/CogVideoX1.5-5b-I2V"
# export CONFIG_PATH="THUDM/CogVideoX-5b-I2V"   # "THUDM/CogVideoX1.5-5b-I2V"
# For finetuning ConsisID
export MODEL_PATH="BestWishYsh/ConsisID-preview"
export CONFIG_PATH="BestWishYsh/ConsisID-preview"
export TYPE="i2v"
export DATASET_PATH="asserts/demo_train_data/total_train_data.txt"
export OUTPUT_PATH="consisid_finetune_single_rank"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# if you are not using wth 8 gus, change `accelerate_config_machine_single.yaml` num_processes as your gpu number
accelerate launch --config_file util/deepspeed_configs/accelerate_config_machine_single.yaml \
  train.py \
  --config_path $CONFIG_PATH \
  --dataloader_num_workers 8 \
  --gradient_checkpointing \
  --pretrained_model_name_or_path $MODEL_PATH \
  --instance_data_root $DATASET_PATH \
  --validation_prompt "The video features a woman standing next to an airplane, engaged in a conversation on her cell phone. She is wearing sunglasses and a black top, and she appears to be talking seriously. The airplane has a green stripe running along its side, and there is a large engine visible behind her. The woman seems to be standing near the entrance of the airplane, possibly preparing to board or just having disembarked. The setting suggests that she might be at an airport or a private airfield. The overall atmosphere of the video is professional and focused, with the woman's attire and the presence of the airplane indicating a business or travel context." \
  --validation_images "asserts/example_images/5.png" \
  --validation_prompt_separator ::: \
  --num_validation_videos 1 \
  --validation_epochs 1000 \
  --seed 42 \
  --mixed_precision bf16 \
  --output_dir $OUTPUT_PATH \
  --height 480 \
  --width 720 \
  --fps 8 \
  --max_num_frames 49 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --checkpointing_steps 250 \
  --num_train_epochs 15 \
  --learning_rate 3e-6 \
  --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 200 \
  --lr_num_cycles 1 \
  --gradient_checkpointing \
  --optimizer AdamW \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --allow_tf32 \
  --resume_from_checkpoint="latest" \
  --report_to wandb \
  --sample_stride 3 \
  --skip_frames_start 7 \
  --skip_frames_end 7 \
  --miss_tolerance 6 \
  --min_distance 3 \
  --min_frames 1 \
  --max_frames 1 \
  --cross_attn_interval 2 \
  --cross_attn_dim_head 128 \
  --cross_attn_num_heads 16 \
  --LFE_id_dim 1280 \
  --LFE_vit_dim 1024 \
  --LFE_depth 10 \
  --LFE_dim_head 64 \
  --LFE_num_heads 16 \
  --LFE_num_id_token 5 \
  --LFE_num_querie 32 \
  --LFE_output_dim 2048 \
  --LFE_ff_mult 4 \
  --LFE_num_scale 5 \
  --local_face_scale 1.0 \
  --is_train_face \
  --is_single_face \
  --enable_mask_loss \
  --is_accelerator_state_dict \
  --is_validation \
  --is_align_face \
  --train_type $TYPE \
  --is_shuffle_data
  
  # --is_kps \
  # --pretrained_weight "checkpoint-1250" \
  # --is_diff_lr \
  # --low_vram \
  # --is_cross_face
  # --enable_slicing \
  # --enable_tiling \
  # --use_ema