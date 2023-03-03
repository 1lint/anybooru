---
license: mit
---

# Anybooru 

The actual dataset is at https://huggingface.co/datasets/lint/anybooru
This repository includes code/instructions for generating your own variant of the dataset.

## Synthetic Anime Image Dataset

Generate a synthetic anime image dataset using an anime style Stable Diffusion checkpoint with Danbooru2021 tags collected in https://gwern.net/danbooru2021. By setting strong global positive and negative prompts with additional embeddings, you can generate a homogenous, high quality image dataset annotated with your input prompts. 

![](./anybooru_examples/image_grid.png)

The Stable Diffusion checkpoint can be in Huggingface diffusers format or in the standard Pytorch-Lightning format hosted on an A1111 webui api. 
I extracted the tag strings from the Danbooru2021 metadata file and uploaded them at https://huggingface.co/datasets/lint/danbooru_tags for use with the included `generate_images.py` script.

## Quick Start

```
pip install -r requirements.txt
```

### Quick Start with Huggingface diffusers
```
python generate_images.py "lint/danbooru_tags/2021_0_pruned.parquet" --pretrained_path="andite/pastel-mix" --use_fp16=True --batch_size=1 --output_dir="./anybooru"
```

### Quick Start with A1111 SD Webui API
Launch your Webui on 127.0.0.1:7860 with `--api` flag, and select an anime style checkpoint. Then run the following command:
```
python generate_images.py "lint/danbooru_tags/2021_0_pruned.parquet" --use_webui_api=True --use_fp16=True --batch_size=1 --output_dir="./anybooru"
```

## Further Details

See the included `.sh` scripts for more advanced options. 
I used the A1111 Webui settings in `./generate_with_A1111_webui.sh` to generate the examples

Advantages of Diffusers generation
-convenient to download files from HF Hub
-no need to launch API server
-can use different prompts within same batch

Advantages of A1111 Webui generation
-easy to test generation settings
-easy to include embeddings/LORA
-can pass unbounded prompt size

## Prominent Issues

Tags annotating multiple people may fail to generate an image with the correct number of people. YMMV depending on which checkpoint you use.
Generating high quality hands is a challenge as always; recommend using a negative embedding to mitigate the issue. 

## Pytorch Dataset

I include a rudimentary Pytorch Dataset implementation for your reference. Note the tag strings/generation metadata are stored inside the image file metadata. 

```
from anybooru_data import PNGDataset

dataset = PNGDataset('./anybooru_examples/')
sample = dataset[0]

image_tensor = sample['image']
print(image_tensor.shape)

tag_string = sample['cond']
print(tag_string)

```

## Citations

```
@misc{danbooru2021, author = {Anonymous and Danbooru community and Gwern Branwen}, title = {Danbooru2021: A Large-Scale Crowdsourced and Tagged Anime Illustration Dataset}, howpublished = {\url{https://gwern.net/danbooru2021}}, url = {https://gwern.net/danbooru2021}, type = {dataset}, year = {2022}, month = {January}, timestamp = {2022-01-21}, note = {Accessed: 03/01/2023} }
```


