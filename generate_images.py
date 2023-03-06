from diffusers import StableDiffusionPipeline, AutoencoderKL, schedulers
from datasets import load_dataset, Dataset
import webuiapi
import pandas as pd
import random
from PIL.PngImagePlugin import PngInfo
from pathlib import Path
import os
import torch
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dataset_path", help="path to tags dataset")

parser.add_argument("--use_webui_api", help="whether to use A1111 webui api or HF diffusers", type=bool, default=False)
parser.add_argument("--positive_prompt", help="positive prompt added to each generation", default="")
parser.add_argument("--negative_prompt", help="negative prompt added to each generation", default="")
parser.add_argument("--batch_size", help="batch size for each generation iteration", type=int, default=2)
parser.add_argument("--output_dir", help="directory to save generated images", default="./anybooru")
parser.add_argument("--sampler_steps", help="Diffusion sampling steps", type=int, default=30)
parser.add_argument("--cfg_scale", help="Guidance scale", type=int, default=7)
parser.add_argument("--skip_batches", help="number of batches to skip", type=int, default=0)

parser.add_argument("--sampler_name", help="Diffusion sampler, see diffusion_sampler_options.txt for options", default="DDIMScheduler")


# arguments for diffusers pipeline
parser.add_argument("--pretrained_path", help="Huggingface pretrained path", default="'hakurei/waifu-diffusion'")
parser.add_argument("--use_fp16", help="whether to use fp16", default=True)
parser.add_argument("--use_xformers", help="whether to use xformers", default=False)


# arguments for A1111 Webui
parser.add_argument("--webui_host", help="A1111 Webui address", default='127.0.0.1')
parser.add_argument("--webui_port", help="A1111 Webui port", type=int, default=7860)


args = parser.parse_args()

dataset_path = Path(args.dataset_path)

if not os.path.isfile(args.dataset_path):
    repo_id = str(Path().joinpath(*dataset_path.parts[:2]))

    if len(dataset_path.parts) > 2:
        file_path = str(Path().joinpath(*dataset_path.parts[2:]))
        data_files = {"train": file_path}
    else:
        data_files = None
    dataset = load_dataset(repo_id, data_files=data_files)['train']

else:
    file_ext = dataset_path.name.rsplit('.', 1)[-1]
    if file_ext in ['pqt', 'parquet']:
        df = pd.read_parquet(dataset_path)
    elif file_ext in ['pkl', 'pickle']:
        df = pd.read_pickle(dataset_path)
    elif file_ext == 'csv':
        df = pd.read_csv(dataset_path)
    else:
        raise Exception(f"Unhandled file extension: {file_ext}")
    dataset = Dataset.from_pandas(df)

os.makedirs(args.output_dir, exist_ok=True)

sampler_settings = {
    "sampler_name": args.sampler_name,
    "steps": args.sampler_steps,
    "cfg_scale": args.cfg_scale
}


if args.use_webui_api:
    # generate with A1111 Webui API
    api = webuiapi.WebUIApi(host=args.webui_host, port=args.webui_port)

    # API can only take on prompt per generation, so batch consists of single tags string
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None)

    if args.sampler_name == "DDIMScheduler":
        args.sampler_name = "DDIM"

else:
    # existing diffusers checkpoints use the standard VAE producing desaturated anime images
    # use better VAE for anime, converted pl vae from https://huggingface.co/andite/pastel-mix to diffusers format
    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_path,
        vae=AutoencoderKL.from_pretrained('lint/anime_vae'), 
        scheduler=getattr(schedulers, args.sampler_name).from_pretrained(args.pretrained_path, subfolder='scheduler'),
        requires_safety_checker=False,
        safety_checker=None,
        feature_extractor=None,
    )

    for component in pipe.components.values():
        if hasattr(component, 'device'):
            if args.use_fp16:
                component.to('cuda', torch.float16)
            else:
                component.to('cuda')

    if args.use_xformers:
        pipe.unet.enable_xformers_memory_efficient_attention()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)



for batch_idx, batch in enumerate(tqdm(dataloader)):

    if batch_idx < args.skip_batches:
        continue

    seed = random.randint(0,2**32)
    id = batch['id']
    tags = batch['tags']

    # for A1111 API, batch_size creates multiple generations of the same prompt using sequential seeds
    # since API can only take one prompt per generation
    if args.use_webui_api:

        positive_prompt = tags + ', ' + args.positive_prompt

        result = api.txt2img(prompt=positive_prompt,
            negative_prompt=args.negative_prompt,
            seed=seed,
            batch_size=args.batch_size,
            **sampler_settings,
        )

        tags_batch = [tags for _ in range(args.batch_size)]
        seeds = [str(seed+i) for i in range(args.batch_size)]
        ids = [id for _ in range(args.batch_size)]

    else:
        tags_batch = tags
        positive_prompt = [tags + ', ' + args.positive_prompt for tags in tags_batch]
        negative_prompt = [args.negative_prompt for _ in range(len(tags_batch))]

        print(positive_prompt)
        print(negative_prompt)

        result = pipe(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=args.sampler_steps,
            guidance_scale=args.cfg_scale,
            num_images_per_prompt=1,
            generator=torch.Generator().manual_seed(seed),
        )

        seeds = [str(seed) for _ in range(args.batch_size)]
        ids = id

    for i, image in enumerate(result.images):
        metadata = PngInfo()
        metadata.add_text("tags", tags_batch[i])
        metadata.add_text("seed", seeds[i])
        for k, v in sampler_settings.items():
            metadata.add_text(k, str(v))
        image.save(Path(args.output_dir)/f'{ids[i]}_{i}.png', pnginfo=metadata)

