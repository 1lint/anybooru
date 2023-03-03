#!/bin/bash

# Example to run image generation script

export DATASET_PATH='lint/danbooru_tags/2021_0_pruned.parquet'
export POSITIVE_PROMPT="masterpiece, best quality, ultra-detailed, illustration, mksks style, best quality, CG, HDR, high quality, high-definition, extremely detailed, mature female, earring, gown, looking at viewer, detailed eyes"
export NEGATIVE_PROMPT="<bad-artist>, (<bad-hands-5>:1), <bad_prompt>, <bad-image-v2-39000>, lowres, ((bad anatomy)), ((bad hands)), text, missing finger, extra digits, fewer digits, blurry, ((mutated hands and fingers)), (poorly drawn face), ((mutation)), ((deformed face)), (ugly), ((bad proportions)), ((extra limbs)), extra face, (double head), (extra head), ((extra feet)), monster, logo, cropped, worst quality, low quality, normal quality, jpeg, humpbacked, long body, long neck, ((jpeg artifacts))"
export OUTPUT_DIR="/mnt/g/data/anybooru"

python generate_images.py "$DATASET_PATH" --use_webui_api=True --batch_size=4 --positive_prompt="$POSITIVE_PROMPT" --negative_prompt="$NEGATIVE_PROMPT" --output_dir="$OUTPUT_DIR" --sampler_name="DPM++ 2M Karras" --sampler_steps=30 --cfg_scale=8 --skip_batches=5626