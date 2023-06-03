from PIL import Image
from IPython.display import display
import torch as th
import torch.nn as nn

from glide_text2im.clip.model_creation import create_clip_model
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
)
from glide_text2im.tokenizer.simple_tokenizer import SimpleTokenizer


has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda')


# Sampling parameters
prompt = "A yellow flower" 
batch_size =  2 
guidance_scale =  8

# A value of 1.0 is sharper, but sometimes results in grainy artifacts.
upsample_temp = 1.0

base_timestep_respacing = '40'

sr_timestep_respacing = 'fast27'


#@title Create base model.
glide_path = '/content/drive/MyDrive/itE10-se6/project2/glide_text2img/glide-finetune/checkpoints/glide-ft-2x249.pt'
import os
options = model_and_diffusion_defaults()
options['use_fp16'] = has_cuda
options['timestep_respacing'] = base_timestep_respacing # use 100 diffusion steps for fast sampling
model, diffusion = create_model_and_diffusion(**options)

if len(glide_path) > 0:
    assert os.path.exists(
        glide_path
    ), f"Failed to resume from {glide_path}, file does not exist."
    weights = th.load(glide_path, map_location="cpu")
    model, diffusion = create_model_and_diffusion(**options)
    model.load_state_dict(weights)
    print(f"Resumed from {glide_path} successfully.")
else:
    model, diffusion = create_model_and_diffusion(**options)
    model.load_state_dict(load_checkpoint("base", device))
model.eval()
if has_cuda:
    model.convert_to_fp16()
model.to(device)
print('total base parameters', sum(x.numel() for x in model.parameters()))


def show_images(batch: th.Tensor):
    """ Display a batch of images inline. """
    scaled = ((batch + 1)*127.5).round().clamp(0,255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    display(Image.fromarray(reshaped.numpy()))



tokens = model.tokenizer.encode(prompt)
tokens, mask = model.tokenizer.padded_tokens_and_mask(
    tokens, options["text_ctx"]
)
uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(
    [], options["text_ctx"]
)
model_kwargs = dict(
    tokens=th.tensor(
        [tokens] * batch_size + [uncond_tokens] * batch_size, device=device
    ),
    mask=th.tensor(
        [mask] * batch_size + [uncond_mask] * batch_size,
        dtype=th.bool,
        device=device,
    ),
)

def cfg_model_fn(x_t, ts, **kwargs):
    half = x_t[: len(x_t) // 2]
    combined = th.cat([half, half], dim=0)
    model_out = model(combined, ts, **kwargs)
    eps, rest = model_out[:, :3], model_out[:, 3:]
    cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
    half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
    eps = th.cat([half_eps, half_eps], dim=0)
    return th.cat([eps, rest], dim=1)
# Sample from the base model.


full_batch_size = batch_size * 2
model.del_cache()
samples = diffusion.plms_sample_loop(
    cfg_model_fn,
    (full_batch_size, 3, options["image_size"], options["image_size"]),
    device=device,
    clip_denoised=True,
    progress=True,
    model_kwargs=model_kwargs,
    cond_fn=None,
)[:batch_size]
model.del_cache()

# Show the output
show_images(samples)