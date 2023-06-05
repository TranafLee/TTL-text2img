# TTL text2img

Text-to-image generation using diffusion models involves iteratively refining images based on textual descriptions. This approach gradually enhances image quality, ensuring coherence with the given text. It enables the generation of visually appealing and contextually relevant images that accurately represent the textual input. <br />
Demo in `huggingface`: [TTL-text2img](https://huggingface.co/spaces/ttranaflee/TTL-text2img)

# Installation
Instructions on how to install and set up TTL-text2img
```sh
git clone https://github.com/TranafLee/TTL-text2img.git
```
```sh
cd TTL-text2img/
```
```sh
# create a virtual environment to keep global install clean
python -m venv .venv 
source .venv/bin/activate
```
```sh
# install all necessary packages
pip install -r requirements.txt
```

# Usage
### Stage 1: Fine-tune the base model
Exact path leading to your dataset folder has to be `./data`. Images and text files has to be all together in the folder. If an image is named `001.jpg` its relative txt file should be named `001.txt` and so on.
```sh
python train.py \
  --data_dir './data' \
  --train_upsample False \
  --project_name 'base_tuning_wandb' \
  --batch_size 4 \
  --learning_rate 1e-04 \
  --side_x 64 \
  --side_y 64 \
  --resize_ratio 1.0 \
  --uncond_p 0.2 \
  --resume_ckpt 'ckpt_to_resume_from.pt' \
  --checkpoints_dir 'my_local_checkpoint_directory' \
```

### Stage 2: Fine-tune the super-resolution model
```sh
python train.py \
  --data_dir '/userdir/data/mscoco' \
  --train_upsample True \
  --image_to_upsample './images/low_res_img.png' \
  --upscale_factor 4 \
  --side_x 64 \
  --side_y 64 \
  --uncond_p 0.0 \
  --resume_ckpt 'ckpt_to_resume_from.pt' \
  --checkpoints_dir 'my_local_checkpoint_directory' \
```
### Full Usage
```sh
usage: train.py [-h] 
                [--data_dir DATA_DIR] 
                [--batch_size BATCH_SIZE]
                [--learning_rate LEARNING_RATE]
                [--adam_weight_decay ADAM_WEIGHT_DECAY] 
                [--side_x SIDE_X]
                [--side_y SIDE_Y] 
                [--resize_ratio RESIZE_RATIO]
                [--uncond_p UNCOND_P] 
                [--train_upsample]
                [--resume_ckpt RESUME_CKPT]
                [--checkpoints_dir CHECKPOINTS_DIR] [--use_fp16]
                [--device DEVICE] 
                [--log_frequency LOG_FREQUENCY]
                [--freeze_transformer] 
                [--freeze_diffusion]
                [--project_name PROJECT_NAME] [--activation_checkpointing]
                [--use_captions] 
                [--epochs EPOCHS] 
                [--test_prompt TEST_PROMPT]
                [--test_batch_size TEST_BATCH_SIZE]
                [--test_guidance_scale TEST_GUIDANCE_SCALE] 
                [--use_webdataset]
                [--wds_image_key WDS_IMAGE_KEY]
                [--wds_caption_key WDS_CAPTION_KEY]
                [--wds_dataset_name WDS_DATASET_NAME] 
                [--seed SEED]
                [--cudnn_benchmark] 
                [--upscale_factor UPSCALE_FACTOR]
```

# Reference
[OpenAI/glide-text2im](https://github.com/openai/glide-text2im) <br />
[OpenAI/guided-diffusion](https://github.com/openai/guided-diffusion)
