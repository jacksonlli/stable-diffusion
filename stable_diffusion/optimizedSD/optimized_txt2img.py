from pathlib import Path
import argparse, os, re
import torch
import numpy as np
from random import randint
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
from stable_diffusion.optimizedSD.optimUtils import split_weighted_subprompts, logger
from transformers import logging
# from samplers import CompVisDenoiser
logging.set_verbosity_error()



def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd


CONFIG = os.path.join(Path(__file__).parent, "v1-inference.yaml")
DEFAULT_CKPT = os.path.join(Path(__file__).parent.parent, "checkpoints", "v1-5-pruned-emaonly.ckpt")

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
        sampler:str="plms",
        ckpt:str=DEFAULT_CKPT,
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
        f: downsampling facto
        n_samples: how many samples to produce for each given prompt. A.k.a. batch size
        scale: unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))
        device: specify GPU (cuda/cuda:0/cuda:1/...)
        from_file: if specified, load prompts from this file
        seed: the seed (for reproducible sampling)
        unet_bs: Slightly reduces inference time at the expense of high VRAM (value > 1 not recommended )
        turbo: Reduces inference time on the expense of 1GB VRAM"
        precision: evaluate at this precision
        format: output image format
        sampler: sampler
        ckpt: path to checkpoint of model
    '''

    tic = time.time()
    os.makedirs(outdir, exist_ok=True)
    outpath = outdir

    if seed == None:
        seed = randint(0, 1000000)
    seed_everything(seed)


    sd = load_model_from_config(f"{ckpt}")
    li, lo = [], []
    for key, _ in sd.items():
        sp = key.split(".")
        if (sp[0]) == "model":
            if "input_blocks" in sp:
                li.append(key)
            elif "middle_block" in sp:
                li.append(key)
            elif "time_embed" in sp:
                li.append(key)
            else:
                lo.append(key)
    for key in li:
        sd["model1." + key[6:]] = sd.pop(key)
    for key in lo:
        sd["model2." + key[6:]] = sd.pop(key)

    config = OmegaConf.load(f"{CONFIG}")

    model = instantiate_from_config(config.modelUNet)
    _, _ = model.load_state_dict(sd, strict=False)
    model.eval()
    model.unet_bs = unet_bs
    model.cdevice = device
    model.turbo = turbo

    modelCS = instantiate_from_config(config.modelCondStage)
    _, _ = modelCS.load_state_dict(sd, strict=False)
    modelCS.eval()
    modelCS.cond_stage_model.device = device

    modelFS = instantiate_from_config(config.modelFirstStage)
    _, _ = modelFS.load_state_dict(sd, strict=False)
    modelFS.eval()
    del sd

    if device != "cpu" and precision == "autocast":
        model.half()
        modelCS.half()

    start_code = None
    if fixed_code:
        start_code = torch.randn([n_samples, C, H // f, W // f], device=device)


    batch_size = n_samples
    if not from_file:
        assert prompt is not None
        prompt = prompt
        print(f"Using prompt: {prompt}")
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {from_file}")
        with open(from_file, "r") as f:
            text = f.read()
            print(f"Using prompt: {text.strip()}")
            data = text.splitlines()
            data = batch_size * list(data)
            data = list(chunk(sorted(data), batch_size))


    if precision == "autocast" and device != "cpu":
        precision_scope = autocast
    else:
        precision_scope = nullcontext

    seeds = ""
    with torch.no_grad():

        for n in trange(n_iter, desc="Sampling"):
            for prompts in tqdm(data, desc="data"):

                sample_path = os.path.join(outpath, "_".join(re.split(":| ", prompts[0])))[:150]
                os.makedirs(sample_path, exist_ok=True)
                base_count = len(os.listdir(sample_path))

                with precision_scope("cuda"):
                    modelCS.to(device)
                    uc = None
                    if negative_prompt is not None and negative_prompt != "":
                        uc = modelCS.get_learned_conditioning(batch_size * [negative_prompt])
                    elif scale != 1.0 :
                        uc = modelCS.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)

                    subprompts, weights = split_weighted_subprompts(prompts[0])
                    if len(subprompts) > 1:
                        c = torch.zeros_like(uc)
                        totalWeight = sum(weights)
                        # normalize each "sub prompt" and add it
                        for i in range(len(subprompts)):
                            weight = weights[i]
                            # if not skip_normalize:
                            weight = weight / totalWeight
                            c = torch.add(c, modelCS.get_learned_conditioning(subprompts[i]), alpha=weight)
                    else:
                        c = modelCS.get_learned_conditioning(prompts)

                    shape = [n_samples, C, H // f, W // f]

                    if device != "cpu":
                        mem = torch.cuda.memory_allocated() / 1e6
                        modelCS.to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)

                    samples_ddim = model.sample(
                        S=ddim_steps,
                        conditioning=c,
                        seed=seed,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=scale,
                        unconditional_conditioning=uc,
                        eta=ddim_eta,
                        x_T=start_code,
                        sampler = sampler,
                    )

                    modelFS.to(device)

                    print(samples_ddim.shape)
                    print("saving images")
                    for i in range(batch_size):

                        x_samples_ddim = modelFS.decode_first_stage(samples_ddim[i].unsqueeze(0))
                        x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
                        Image.fromarray(x_sample.astype(np.uint8)).save(
                            os.path.join(sample_path, "seed_" + str(seed) + "_" + f"{base_count:05}.{format}")
                        )
                        seeds += str(seed) + ","
                        seed += 1
                        base_count += 1

                    if device != "cpu":
                        mem = torch.cuda.memory_allocated() / 1e6
                        modelFS.to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)
                    del samples_ddim
                    print("memory_final = ", torch.cuda.memory_allocated() / 1e6)

    toc = time.time()

    time_taken = (toc - tic) / 60.0

    print(
        (
            "Samples finished in {0:.2f} minutes and exported to "
            + sample_path
            + "\n Seeds used = "
            + seeds[:-1]
        ).format(time_taken)
    )


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
    
    # Logging
    logger(vars(opt), log_csv = "logs/txt2img_logs.csv")

    main(**vars(opt))
    