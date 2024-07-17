import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class VOC2007Dataset(Dataset):
    def __init__(self, root_dir, image_set='train', image_transform=None, mask_transform=None):
        self.root_dir = root_dir
        self.image_set = image_set
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.image_dir = os.path.join(root_dir, 'JPEGImages')
        self.mask_dir = os.path.join(root_dir, 'SegmentationClass')
        self.image_list = self._get_image_list()

    def _get_image_list(self):
        image_set_file = os.path.join(self.root_dir, 'ImageSets', 'Segmentation', self.image_set + '.txt')
        with open(image_set_file, 'r') as f:
            image_list = f.read().splitlines()
        return image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name + '.jpg')
        mask_path = os.path.join(self.mask_dir, img_name + '.png')

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            mask = mask.squeeze(0)  # 移除第2个维度，使其变为 (H, W)
            mask = mask.long()  # 转换为 Long 类型

        return image, mask



