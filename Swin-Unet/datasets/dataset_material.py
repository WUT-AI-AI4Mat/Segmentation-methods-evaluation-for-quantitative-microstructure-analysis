import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class MaterialDataset(Dataset):
    def __init__(self, base_dir, list_dir=None, split="train", transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform


        self.image_dir = os.path.join(self._base_dir, split, 'images')
        self.label_dir = os.path.join(self._base_dir, split, 'masks')

        self.sample_list = [f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.png', '.tif', '.bmp'))]
        print(f"[{split}] 发现 {len(self.sample_list)} 张图片。")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        img_path = os.path.join(self.image_dir, case)
        label_path = os.path.join(self.label_dir, case)

        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L')


        image = np.array(image)
        label = np.array(label)


        label[label > 0] = 1

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
