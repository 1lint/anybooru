#!/bin/bash

# Example to run image generation script

export DATASET_PATH='lint/danbooru_tags/2021_0_pruned.parquet'
export PRETRAINED_PATH='andite/pastel-mix'
export POSITIVE_PROMPT="masterpiece, best quality, ultra-detailed, illustration, mksks style, best quality, CG, HDR, high quality, high-definition, extremely detailed, mature female, earring, gown, looking at viewer, detailed eyes"
export NEGATIVE_PROMPT="lowres, ((bad anatomy)), ((bad hands)), text, missing finger, extra digits, fewer digits, blurry, ((mutated hands and fingers)), (poorly drawn face), ((mutation)), ((deformed face)), (ugly), ((bad proportions)), ((extra limbs)), extra face, (double head), (extra head), ((extra feet)), monster, logo, cropped, worst quality, low quality, normal quality, jpeg, humpbacked, long body, long neck, ((jpeg artifacts))"
export OUTPUT_DIR="/mnt/g/data/anybooru_diffusers"
export SKIP_BATCHES=$((5626/4))

python generate_images.py "$DATASET_PATH" --pretrained_path="$PRETRAINED_PATH" --batch_size=4 --positive_prompt="$POSITIVE_PROMPT" --negative_prompt="$NEGATIVE_PROMPT" --output_dir="$OUTPUT_DIR" --sampler_name="DPMSolverMultistepScheduler" --sampler_steps=30 --cfg_scale=7 --skip_batches=$SKIP_BATCHES --use_xformers=True --use_fp16=True