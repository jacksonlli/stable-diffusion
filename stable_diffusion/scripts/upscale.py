import subprocess
import os
import gc
import torch
from torch import autocast
import numpy as np
from tqdm import tqdm, trange
import PIL
from PIL import Image, ImageDraw
from einops import rearrange, repeat
from contextlib import contextmanager, nullcontext, redirect_stdout

from stable_diffusion.ldm.models.diffusion.ddim import DDIMSampler


def upscale(
    prompt,
    negative_prompt,
    images,
    W,
    H,
    model,
    realesrgan_executable,
    strength=0.3,
    detail_steps=150,
    detail_scale=10,
    gobig_overlap=128,
    passes=1,
    device='cuda',
    precision='full',
    only_keep_final=True
):
    """
    Args:
        W: subimage w
        H: subimage h
        strength: strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image
        detail_steps: number of sampling steps when detailing
        detail_scale: unconditional guidance scale when detailing: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))
        passes: number of upscales/details
    """
    torch.cuda.empty_cache()
    gc.collect()
    sampler = DDIMSampler(model)
    batch_size = len(images)
    data = [batch_size * [prompt]]
    upscaled_files = []
    for img_file in images:
        file = img_file
        for _ in trange(passes, desc="Passes"):
            real_ersgan_output_file = file.split('.png')[0]+'_realersgan.png'
            run_real_ersgan(realesrgan_executable, file, real_ersgan_output_file)

            source_image = Image.open(real_ersgan_output_file)
            og_size = (H, W)
            slices, _ = grid_slice(source_image, gobig_overlap, og_size, False)

            betterslices = []
            
            for chunk_w_coords in tqdm(slices, "Slices"):
                chunk, coord_x, coord_y = chunk_w_coords
                init_image = convert_pil_img(chunk).to(device)
                init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
                init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

                sampler.make_schedule(ddim_num_steps=detail_steps, ddim_eta=0, verbose=False)

                assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
                t_enc = int(strength * detail_steps)
                precision_scope = autocast if precision=="autocast" else nullcontext

                with torch.inference_mode():
                    with precision_scope("cuda"):
                        with model.ema_scope():
                            with redirect_stdout(None):
                                for prompts in data:
                                    uc = None
                                    if negative_prompt is not None and negative_prompt != "":
                                        uc = model.get_learned_conditioning(batch_size * [negative_prompt])
                                    elif detail_scale != 1.0 :
                                        uc = model.get_learned_conditioning(batch_size * [""])
                                    if isinstance(prompts, tuple):
                                        prompts = list(prompts)
                                    c = model.get_learned_conditioning(prompts)

                                    # encode (scaled latent)
                                    z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                                    # decode it
                                    samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=detail_scale,
                                                            unconditional_conditioning=uc, disable_tqdm=True)

                                    x_samples = model.decode_first_stage(samples)
                                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                                    for x_sample in x_samples:
                                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                        resultslice = Image.fromarray(x_sample.astype(np.uint8)).convert('RGBA')
                                        betterslices.append((resultslice.copy(), coord_x, coord_y))

            alpha = Image.new('L', og_size, color=0xFF)
            alpha_gradient = ImageDraw.Draw(alpha)
            a = 0
            i = 0
            overlap = gobig_overlap
            shape = (og_size, (0,0))
            while i < overlap:
                alpha_gradient.rectangle(shape, fill = a)
                a += 4
                i += 1
                shape = ((og_size[0] - i, og_size[1]- i), (i,i))
            mask = Image.new('RGBA', og_size, color=0)
            mask.putalpha(alpha)
            finished_slices = []
            for betterslice, x, y in betterslices:
                finished_slice = addalpha(betterslice, mask)
                finished_slices.append((finished_slice, x, y))
            # # Once we have all our images, use grid_merge back onto the source, then save
            final_output = grid_merge(source_image.convert("RGBA"), finished_slices).convert("RGB")
            if only_keep_final:
                os.remove(real_ersgan_output_file)
                os.remove(file)
            file = file.split('.png')[0]+'_upscaled.png'
            final_output.save(file)

            torch.cuda.empty_cache()
            gc.collect()
        upscaled_files.append(file)
    return upscaled_files

def run_real_ersgan(real_ersgan_executable_path, upscale_input, upscale_output):
    subprocess.run([real_ersgan_executable_path, '-i', upscale_input, '-o', upscale_output, '-n', 'realesrgan-x4plus'])
    final_output = Image.open(upscale_output)
    final_output = final_output.resize((int(final_output.size[0] / 2), int(final_output.size[1] / 2)), get_resampling_mode())
    final_output.save(upscale_output)

def grid_slice(source, overlap, og_size, maximize=False):
    """
    Chop our source into a grid of images that each equal the size of the original render
    """
    width, height = og_size # size of the slices to be rendered
    coordinates, new_size = grid_coords(source.size, og_size, overlap)
    if maximize == True:
        source = source.resize(new_size, get_resampling_mode()) # minor concern that we're resizing twice
        coordinates, new_size = grid_coords(source.size, og_size, overlap) # re-do the coordinates with the new canvas size
    # loc_width and loc_height are the center point of the goal size, and we'll start there and work our way out
    slices = []
    for coordinate in coordinates:
        x, y = coordinate
        slices.append(((source.crop((x, y, x+width, y+height))), x, y))
    global slices_todo
    slices_todo = len(slices) - 1
    return slices, new_size

def grid_coords(target, original, overlap):
    """
    generate a list of coordinate tuples for our sections, in order of how they'll be rendered
    target should be the size for the gobig result, original is the size of each chunk being rendered
    """
    
    center = []
    target_x, target_y = target
    center_x = int(target_x / 2)
    center_y = int(target_y / 2)
    original_x, original_y = original
    x = center_x - int(original_x / 2)
    y = center_y - int(original_y / 2)
    center.append((x,y)) #center chunk
    uy = y #up
    uy_list = []
    dy = y #down
    dy_list = []
    lx = x #left
    lx_list = []
    rx = x #right
    rx_list = []
    while uy > 0: #center row vertical up
        uy = uy - original_y + overlap
        uy_list.append((lx, uy))
    while (dy + original_y) <= target_y: #center row vertical down
        dy = dy + original_y - overlap
        dy_list.append((rx, dy))
    while lx > 0:
        lx = lx - original_x + overlap
        lx_list.append((lx, y))
        uy = y
        while uy > 0:
            uy = uy - original_y + overlap
            uy_list.append((lx, uy))
        dy = y
        while (dy + original_y) <= target_y:
            dy = dy + original_y - overlap
            dy_list.append((lx, dy))
    while (rx + original_x) <= target_x:
        rx = rx + original_x - overlap
        rx_list.append((rx, y))
        uy = y
        while uy > 0:
            uy = uy - original_y + overlap
            uy_list.append((rx, uy))
        dy = y
        while (dy + original_y) <= target_y:
            dy = dy + original_y - overlap
            dy_list.append((rx, dy))
    # calculate a new size that will fill the canvas, which will be optionally used in grid_slice and go_big
    last_coordx, last_coordy = dy_list[-1:][0]
    render_edgey = last_coordy + original_y # outer bottom edge of the render canvas
    render_edgex = last_coordx + original_x # outer side edge of the render canvas
    scalarx = render_edgex / target_x
    scalary = render_edgey / target_y
    if scalarx <= scalary:
        new_edgex = int(target_x * scalarx)
        new_edgey = int(target_y * scalarx)
    else:
        new_edgex = int(target_x * scalary)
        new_edgey = int(target_y * scalary)
    # now put all the chunks into one master list of coordinates (essentially reverse of how we calculated them so that the central slices will be on top)
    result = []
    for coords in dy_list[::-1]:
        result.append(coords)
    for coords in uy_list[::-1]:
        result.append(coords)
    for coords in rx_list[::-1]:
        result.append(coords)
    for coords in lx_list[::-1]:
        result.append(coords)
    result.append(center[0])
    return result, (new_edgex, new_edgey)

def get_resampling_mode():
    try:
        from PIL import __version__, Image
        major_ver = int(__version__.split('.')[0])
        if major_ver >= 9:
            return Image.Resampling.LANCZOS
        else:
            return Image.LANCZOS
    except Exception as ex:
        return 1  # 'Lanczos' irrespective of version.

def convert_pil_img(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def addalpha(im, mask):
    imr, img, imb, ima = im.split()
    mmr, mmg, mmb, mma = mask.split()
    im = Image.merge('RGBA', [imr, img, imb, mma])  # we want the RGB from the original, but the transparency from the mask
    return(im)

def grid_merge(source, slices):
    source.convert("RGBA")
    for slice, posx, posy in slices: # go in reverse to get proper stacking
        source.alpha_composite(slice, (posx, posy))
    return source
