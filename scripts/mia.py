import argparse, os, sys, glob
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms
import sys
sys.path.append('..')
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from skimage.metrics import structural_similarity


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model



def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))



def ddim_singlestep(model, x, t_c, t_target, prompt, scale):
    device = x.device
    uc = model.get_learned_conditioning(1 * [""])
    c = model.get_learned_conditioning(1 * [prompt])

    t_c = x.new_ones([x.shape[0], ], dtype=torch.long) * (t_c)
    t_target = x.new_ones([x.shape[0], ], dtype=torch.long) * (t_target)

    betas = torch.linspace(model.betas[0], model.betas[-1], 1000).double().to(device)
    alphas = 1. - betas
    alphas = torch.cumprod(alphas, dim=0)
    alphas_t_c = extract(alphas, t=t_c, x_shape=x.shape)
    alphas_t_target = extract(alphas, t=t_target, x_shape=x.shape)

    x_in = torch.cat([x] * 2)
    t_in = torch.cat([t_c] * 2)
    c_in = torch.cat([uc, c])
    e_t_uncond, e_t = model.apply_model(x_in, t_in, cond=c_in)
    epsilon = e_t_uncond + scale * (e_t - e_t_uncond)
    # epsilon = model.apply_model(x, t_c, cond=c)

    pred_x_0 = (x - ((1 - alphas_t_c).sqrt() * epsilon)) / (alphas_t_c.sqrt())
    x_t_target = alphas_t_target.sqrt() * pred_x_0 \
                 + (1 - alphas_t_target).sqrt() * epsilon

    return x_t_target



def naive_inversion(model, x_0, prompt, scale, use_ddim=False, steps=None):

    if use_ddim:
        assert steps is not None
    else:
        x = x_0
        t_c = 0
        target_steps = list(range(0, 150, 50))[1:]

        for idx, t_targets in enumerate(target_steps):
            result = ddim_singlestep(model, x, t_c, t_targets, prompt, scale)
            x = result
            t_c = t_targets
        x_sec = x

        return x_sec



def cal_loss(raw_image, x_sample):
    raw_image = np.array(raw_image)
    x_sample = np.array(x_sample.cpu())
    ssim = structural_similarity(raw_image, x_sample, data_range=1.0, channel_axis=0)

    return ssim



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--imgname",
        type=str,
        help="dir to image",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=256,
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=256,
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="textual prompt",
    )

    parser.add_argument(
        "--yaml",
        type=str,
        default="../configs/latent-diffusion.yaml",
        help="dir to LDM's yaml file",
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default="../models/ldm/model.ckpt",
        help="dir to LDM",
    )

    opt = parser.parse_args()

    config = OmegaConf.load(opt.yaml)  # TODO: Optionally download from same location as ckpt and chnage this logic
    model = load_model_from_config(config, opt.ckpt)  # TODO: check path
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    H = opt.H
    W = opt.W
    prompt = opt.prompt
    img_url = opt.imgname
    scale = opt.scale

    with torch.no_grad():
        with model.ema_scope():

            raw_image = Image.open(img_url).convert('RGB')
            img = np.array(raw_image).astype(np.uint8)
            crop = min(img.shape[0], img.shape[1])
            h, w, = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]
            image = Image.fromarray(img)
            image = image.resize((H, W), resample=Image.BICUBIC)
            image = np.array(image).astype(np.uint8)
            image = (image / 127.5 - 1.0).astype(np.float32)
            raw_image = transforms.ToTensor()(image)

            c = model.get_learned_conditioning(1 * [prompt])
            m = {}
            m['image'] = raw_image
            m['image_path'] = img_url
            m['caption'] = c

            x_0, cond = model.get_input(m, model.first_stage_key)
            x_T = naive_inversion(model, x_0, prompt, scale, use_ddim=False, steps=None)
            x_samples = model.decode_first_stage(x_T)

            for x_sample in x_samples:
                ssim = cal_loss(raw_image, x_sample)
                print("The value of ssim is: ")
                print(ssim)

                loss_file_ssim_name = 'ssim.txt'
                loss_file_ssim = open(loss_file_ssim_name, 'a')
                loss_str_ssim = str(ssim) + '\n'
                loss_file_ssim.write(loss_str_ssim)
                loss_file_ssim.close()


