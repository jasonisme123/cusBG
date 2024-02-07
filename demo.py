from diffusers.utils import load_image
from diffusers import AutoPipelineForInpainting, LCMScheduler
from skimage import io
import torch
import os
from PIL import Image
from briarmbg import BriaRMBG
from utilities import preprocess_image, postprocess_image
from PIL import Image
import argparse
from PyDeepLX import PyDeepLX


def remove_bg(img_name):
    model_path = './weight/model.pth'
    im_path = f"{os.path.dirname(os.path.abspath(__file__))}/{img_name}"

    net = BriaRMBG()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()

    # prepare input
    model_input_size = [1024, 1024]
    orig_im = io.imread(im_path)
    orig_im_size = orig_im.shape[0:2]
    image = preprocess_image(orig_im, model_input_size).to(device)

    # inference
    result = net(image)

    # post process
    result_image = postprocess_image(result[0][0], orig_im_size)

    # save result
    pil_im = Image.fromarray(result_image)
    # no_bg_image = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
    # orig_image = Image.open(im_path)
    # no_bg_image.paste(orig_image, mask=pil_im)
    pil_im.save("no_bg_image.png")
    return 'no_bg_image.png'


def resize_img(w, h, resolution):
    if w > h:
        h = int(h * resolution / w)
        w = resolution
    else:
        w = int(w * resolution / h)
        h = resolution

    if w % 8 != 0:
        w1 = w + (8 - w % 8)
        w2 = w - w % 8
        w = min(w1, w2)
    if h % 8 != 0:
        h1 = h + (8 - h % 8)
        h2 = h - h % 8
        h = min(h1, h2)

    return w, h


def mask2img(orig_img, mask, prompt, resolution):
    pipe = AutoPipelineForInpainting.from_pretrained(
        "Uminosachi/realisticVisionV51_v51VAE-inpainting",
        safety_checker=None,
        requires_safety_checker=False
    )
    # set scheduler
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    # load LCM-LoRA
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    # load base and mask image
    init_image = load_image(orig_img)
    mask_image = load_image(mask)
    w = init_image.width
    h = init_image.height
    if w > resolution or h > resolution:
        w, h = resize_img(w, h, resolution)

    generator = torch.manual_seed(0)
    image = pipe(
        prompt=prompt,
        image=init_image,
        mask_image=mask_image,
        width=w,
        height=h,
        generator=generator,
        num_inference_steps=8,
        guidance_scale=1.5,
    ).images[0]

    image.save("inpainting.png")


def main(imgName: str, prompt: str, resolution: int):
    remove_bg_name = remove_bg(imgName)
    mask2img(imgName, remove_bg_name, prompt, resolution)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--resolution", type=int, required=False, default=512)
    args = parser.parse_args()
    en_prompt = PyDeepLX.translate(args.prompt, "ZH", "EN")

    if args.resolution % 8 != 0:
        raise ValueError("resolution must be a multiple of 8")
    elif not os.path.exists(args.image):
        raise ValueError("image does not exist")
    else:
        main(args.image, en_prompt, args.resolution)
