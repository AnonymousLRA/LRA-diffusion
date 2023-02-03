import os
import clip
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import numpy as np
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _transform(n_px):
    return Compose([
        Normalize(mean=[-0.4914 / 0.2023, -0.4822 / 0.1994, -0.4465 / 0.2010],
                  std=[1 / 0.2023, 1 / 0.1994, 1 / 0.2010]),
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        # ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class clip_img_wrap(nn.Module):
    def __init__(self, clip_model='ViT-L/14', device='cpu'):
        super().__init__()

        self.model, self.preprocess = clip.load(clip_model, device)
        self.name = '-'.join(clip_model.split('/'))
        self.device = device
        self.dim = self.model.text_projection.shape[1]

        self.inv_normalize = _transform(self.model.visual.input_resolution)

    def forward(self, image):

        image = self.inv_normalize(image)
        with torch.no_grad():
            image_features = self.model.encode_image(image)

        return image_features.float()



# clip_model = clip_img_wrap('ViT-L/14', 'cpu')

