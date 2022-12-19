# %%
%load_ext autoreload
%autoreload 2


from tqdm import tqdm
import yaml
import torch
import os
from utils.dataloaders import LoadImagesAndLabels
from utils.plots import plot_images
import random
import numpy as np
from arguments import get_test_args, get_train_args, ROOT
from utils.general import init_seeds
import logging


args = get_train_args()

args.data = ROOT / 'data/eye.yaml'
    
# Hyperparameters
with open(args.hyp) as f:
    hyp = yaml.load(f, Loader=yaml.SafeLoader)


# %%

init_seeds(1)

from utils.dataloaders import LoadImagesAndLabels

# no clip augmentation
args.no_clip = True

path = "/local/yermaand/yolov5/datasets/eye/val.txt"
dataset = LoadImagesAndLabels(path, hyp=hyp, augment=True, args=args)
batch_size = 16

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, collate_fn=LoadImagesAndLabels.collate_fn,
    # shuffle=True
)

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader)):
    folder = f"outputs/eye"
    os.makedirs(folder, exist_ok=True)
    fname = f"{folder}/{batch_i}.jpg"
    plot_images(img, targets, paths, fname)

    if batch_i == 10:
        break
 
 

# %%
