import argparse, os, gc
from pathlib import Path
import argparse
from random import randint
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from stable_diffusion.ldm.util import instantiate_from_config
from stable_diffusion.ldm.models.diffusion.ddim import DDIMSampler
from stable_diffusion.ldm.models.diffusion.plms import PLMSSampler
from transformers import logging
from stable_diffusion.scripts.upscale import upscale
from stable_diffusion.scripts.gfpgan import main as gfpgan


logging.set_verbosity_error()
DEFAULT_CKPT = os.path.join(Path(__file__).parent.parent, "checkpoints", "v1-5-pruned-emaonly.ckpt")
CONFIG = os.path.join(Path(__file__).parent.parent, "configs", "stable-diffusion", "v1-inference.yaml")


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    checkpoint = torch.load(ckpt, map_location="cpu")
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(state_dict, strict=False)

    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def main(        
        prompt: str,
        negative_prompt:str = "",
        outdir: str = "outputs/txt2img/",
        ddim_steps: int = 50,
        fixed_code: bool = False,    
        ddim_eta:float = 0.0,
        n_iter:int = 1,
        H:int = 512,
        W:int=512,
        C:int=4,
        f:int=8,
        n_samples:int=5,    
        scale:float=7.5,
        device:str="cuda",
        from_file:str=None,
        seed:int=None,    
        unet_bs:int=1,
        turbo:bool=False,
        precision:str="full",
        format:str="png",
        ckpt:str=DEFAULT_CKPT,
        plms:bool = True,
        n_rows=0,
        upscale_strength=0.3,
        upscale_steps=150,
        upscale_scale=None,
        upscale_overlap=128,
        upscale_passes=1,
        real_ersgan_executable_path=None,
        gfpgan_model=None,
        only_keep_final=True,
        **kwargs
):
    '''
    Args:
        prompt: the prompt to render,
        negative_prompt: what not to look like
        outdir: dir to write results to
        ddim_steps: number of ddim sampling steps
        fixed_code: if enabled, uses the same starting code across samples
        ddim_eta: ddim eta (eta=0.0 corresponds to deterministic sampling
        n_iter: sample this often
        H: image height, in pixel space
        W: image width, in pixel space
        C: latent channels
        f: downsampling factor
        n_samples: how many samples to produce for each given prompt. A.k.a. batch size
        scale: unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))
        device: specify GPU (cuda/cuda:0/cuda:1/...)
        from_file: if specified, load prompts from this file
        seed: the seed (for reproducible sampling)
        unet_bs: Slightly reduces inference time at the expense of high VRAM (value > 1 not recommended )
        turbo: Reduces inference time on the expense of 1GB VRAM"
        precision: evaluate at this precision
        format: output image format
        plms: sampler plms or ddim
        real_ersgan_executable_path: if provided, will be used to upscale the image 4x
        ckpt: path to checkpoint of model
    '''
    if seed == None:
        seed = randint(0, 1000000)
    seed_everything(seed)

    config = OmegaConf.load(f"{CONFIG}")
    model = load_model_from_config(config, f"{ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(outdir, exist_ok=True)
    outpath = outdir

    batch_size = n_samples
    n_rows = n_rows if n_rows > 0 else batch_size
    if not from_file:
        prompt = prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {from_file}")
        with open(from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    base_count = len(os.listdir(outpath))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if fixed_code:
        start_code = torch.randn([n_samples, C, H // f, W // f], device=device)

    generated_imgs = []

    precision_scope = autocast if precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if negative_prompt is not None and negative_prompt != "":
                            uc = model.get_learned_conditioning(batch_size * [negative_prompt])
                        elif scale != 1.0 :
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [C, H // f, W // f]
                        samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                         conditioning=c,
                                                         batch_size=n_samples,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=scale,
                                                         unconditional_conditioning=uc,
                                                         eta=ddim_eta,
                                                         x_T=start_code)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                for x_sample in x_samples_ddim:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    img_path = os.path.join(outpath, "seed_" + str(seed) + "_" + f"{base_count:05}.{format}")
                    Image.fromarray(x_sample.astype(np.uint8)).save(
                        img_path
                    )
                    generated_imgs.append(img_path)
                    base_count += 1

                toc = time.time()
                torch.cuda.empty_cache()
                gc.collect()
    if real_ersgan_executable_path:
        if gfpgan_model:
            gfpgan_generated_imgs = gfpgan(
                input=generated_imgs,
                output=outdir,
                version='1.4',
                model_path=gfpgan_model,
                upscale=1,
            )
            if only_keep_final:
                for file in generated_imgs:
                    os.remove(file)
            generated_imgs = gfpgan_generated_imgs

        generated_imgs = upscale(
            images=generated_imgs,
            model=model,
            W=W,
            H=H,
            realesrgan_executable=real_ersgan_executable_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=upscale_strength,
            detail_steps=upscale_steps,
            detail_scale=upscale_scale if upscale_scale else scale,
            gobig_overlap=upscale_overlap,
            device='cuda',
            precision='full',
            passes=upscale_passes,
            only_keep_final=only_keep_final
        )

    if gfpgan_model:
        gfpgan(input=generated_imgs, output=outdir, version='1.4', model_path=gfpgan_model)
        if only_keep_final:
            for file in generated_imgs:
                os.remove(file)
        
    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt", type=str, nargs="?", default="a painting of a virus monster playing guitar", help="the prompt to render"
    )
    parser.add_argument(
        "--negative_prompt", type=str, nargs="?", default="", help="the prompt to not render"
    )
    parser.add_argument("--outdir", type=str, nargs="?", help="dir to write results to", default="outputs/txt2img-samples")
    parser.add_argument(
        "--skip_grid",
        action="store_true",
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action="store_true",
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--fixed_code",
        action="store_true",
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="specify GPU (cuda/cuda:0/cuda:1/...)",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--unet_bs",
        type=int,
        default=1,
        help="Slightly reduces inference time at the expense of high VRAM (value > 1 not recommended )",
    )
    parser.add_argument(
        "--turbo",
        action="store_true",
        help="Reduces inference time on the expense of 1GB VRAM",
    )
    parser.add_argument(
        "--precision", 
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="full"
    )
    parser.add_argument(
        "--format",
        type=str,
        help="output image format",
        choices=["jpg", "png"],
        default="png",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        help="sampler",
        choices=["ddim", "plms","heun", "euler", "euler_a", "dpm2", "dpm2_a", "lms"],
        default="plms",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="path to checkpoint of model",
        default=DEFAULT_CKPT,
    )
    opt = parser.parse_args()
    
    main(**vars(opt))
