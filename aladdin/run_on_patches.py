"""
Let's say you have an extremely large image but you can't run
it all through the network (and you don't want to resize etc)
then you can split the image into crops, run each cropped part
through the model and then stitch them back together. Perhaps
you would also want to use overlapping images to get a smoother 
output, then this can be useful. PyTorch Unfold/Fold is also a
way to also vectorize doing sliding windows approach, I guess
there are a ton of useful ways to use those functions nevermind me 
I'm just randomly babbling here
"""

import torch
from PIL import Image
import math
import numpy as np
import cv2
import albumentations as A
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2

def run_on_patches(image, model, kernel_size=256, stride=64, device="cuda", toggle_eval=True):
    model = model.to(device)
    if toggle_eval:
        model.eval()

    #image = Image.open(image_path).convert("RGB")
    #width, height = image.size
    #max_size = math.ceil(max(width, height) / kernel_size) * kernel_size
    #pad_height = max_size - height
    #pad_width = max_size - width
    #image = np.array(image)
    #augment = A.Compose([
    #    A.PadIfNeeded(min_width=max_size, min_height=max_size, border_mode=cv2.BORDER_REFLECT),
    #    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0, ),
    #    ToTensorV2(),
    #])
    #image = augment(image=image)["image"].to(device)
    #img_size = image.shape[2]

    image = np.array(image).permute(1,2,0)
    patches = image.unfold(0, kh, dh).unfold(1, kernel_size, kernel_size)
    patches = patches.contiguous().view(-1, 3, stride, stride)

    # Run on patch
    with torch.no_grad():
        model = model.to("cuda")
        batch_size = 32
        for id in tqdm(range(math.ceil(patches.shape[0]/16))):
            from_idx = id*batch_size
            to_idx = min((id+1)*batch_size, patches.shape[0])
            curr_patch = patches[from_idx:to_idx].to("cuda")
            new_patch = model(curr_patch)*0.5+0.5
            patches[from_idx:to_idx] = new_patch.to("cpu")

    patches = patches.view(1, patches.shape[0], 3*kernel_size*kernel_size).permute(0, 2, 1)
    output = F.fold(patches, output_size=(img_size, img_size),
                    kernel_size=kernel_size, stride=dh)
    recovery_mask = F.fold(torch.ones_like(patches), output_size=(
        img_size, img_size), kernel_size=kernel_size, stride=dh)
    output /= recovery_mask
    #augment_back = A.Compose([
    #    A.CenterCrop(height=max_size-int(pad_height), width=max_size - int(pad_width)),
    #    ToTensorV2(),
    #    ])
    #x = augment_back(image=output.squeeze(0).detach().cpu().permute(1,2,0).numpy())["image"]
    return output
    #save_image(x, f"{image_path}_.png")

model.train()






