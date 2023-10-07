# ------------------------------------------------------------------------------ #
# Author: Zhenwei Shao (https://github.com/ParadoxZW)
# Description: Tool for extracting image features
# ------------------------------------------------------------------------------ #

import os, sys

sys.path.append(os.getcwd())

import glob
import numpy as np
import torch
from torch import nn
from PIL import Image
import clip
from tqdm import tqdm
import argparse
from pathlib import Path

from configs.task_cfgs import Cfgs
from configs.task_to_split import *

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode


@torch.no_grad()
def _extract_feat(img_path, net, T, save_path):
    # print(img_path)
    img = Image.open(img_path)
    # W, H = img.size
    img = T(img).unsqueeze(0).cuda()
    clip_feats = net(img).cpu().numpy()[0]
    clip_feats = clip_feats.transpose(1, 2, 0)
    # print(clip_feats.shape, save_path)
    # return
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        save_path,

        x=clip_feats,
    )


class ExtractModel:
    def __init__(self, __C, encoder) -> None:
        encoder.attnpool = nn.Identity()
        self.__C = __C
        self.backbone = encoder

        self.backbone.cuda().eval()

        if self.__C.N_GPU > 1:
            self.backbone = nn.DataParallel(self.backbone, device_ids=self.__C.GPU_IDS)

    @torch.no_grad()
    def __call__(self, img):
        x = self.backbone(img)
        return x


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def transform(n_px, pad=False, crop=False):
    return Compose([
        Resize([n_px, n_px], interpolation=InterpolationMode.BICUBIC),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def main(__C):
    # find imgs
    img_name_list = ['train2014', 'val2014', 'test2015']
    img_dir_list = []
    for name in img_name_list:
        img_dir_list.append('datasets/img_vqav2/' + name + '/')
    print('image dirs:', img_dir_list)
    img_path_list = []
    for img_dir in img_dir_list:
        img_path_list += glob.glob(img_dir + '*.jpg')
    print('total images:', len(img_path_list))

    # load model
    clip_model, _ = clip.load(__C.CLIP_VERSION, device='cpu')
    img_encoder = clip_model.visual

    model = ExtractModel(__C, img_encoder)
    T = transform(__C.IMG_RESOLUTION)

    for img_path in tqdm(img_path_list):
        img_path_sep = img_path.split('/')
        img_path_sep[-3] += '_feats'
        save_path = '/'.join(img_path_sep).replace('.jpg', '.npz')
        if Path(save_path).is_file():
            continue
        _extract_feat(img_path, model, T, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Tool for extracting CLIP image features.')
    parser.add_argument('--gpu', dest='GPU', help='gpu id', type=str, default='0')
    parser.add_argument('--clip_model', dest='CLIP_VERSION', help='clip model name or local model checkpoint path', type=str, default='RN50x64')
    parser.add_argument('--img_resolution', dest='IMG_RESOLUTION', help='image resolution', type=int, default=512)
    args = parser.parse_args()
    __C = Cfgs(args)
    main(__C)
