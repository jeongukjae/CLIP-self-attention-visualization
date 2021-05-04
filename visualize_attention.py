import argparse
import os
import sys
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.datasets import Places365, CIFAR100
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
import clip

parser = argparse.ArgumentParser()
parser.add_argument("--index", default=0, type=int, required=True)
parser.add_argument("--dataset", type=str, required=True)

model, preprocess = clip.load("ViT-B/32", jit=False)

args = parser.parse_args()

if args.dataset == 'place365':
    ds = Places365(root=os.path.expanduser("~/.cache"), small=True, split='val')
elif args.dataset == 'cifar100':
    ds = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
else:
    raise ValueError

transform_image = Compose([
    Resize(model.visual.input_resolution, interpolation=Image.BICUBIC),
    CenterCrop(model.visual.input_resolution),
    lambda image: image.convert("RGB"),
])

# Prepare the inputs
image, class_id = ds[args.index]
image_input = preprocess(image).unsqueeze(0)

with torch.no_grad():
    image_attention = model.encode_image_attention(image_input)

fig = plt.figure(figsize=[10, 5], frameon=False)
ax = fig.add_subplot(1, 2, 1)
ax.axis("off")
ax.imshow(transform_image(image))
ax = fig.add_subplot(1, 2, 2)
ax.axis("off")
ax.pcolor(image_attention[0].reshape(7, 7))
fig.subplots_adjust(hspace=0, wspace=0)
fig.savefig(f"{args.dataset}_{args.index}")
