import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
class Task1Dataset(Dataset):
    def __init__(self,file_path):
        self.file_path = file_path
    def __len__(self):
        return len(self.file_path)
    def __getitem__(self,idx):
        # print(self.file_path[idx])
        img = Image.open(self.file_path[idx]).convert('RGB')
        transform = transforms.Compose([
            # 裁剪成 160 x 160 中心裁剪
            transforms.CenterCrop(160),
            transforms.ToTensor(),
            # 归一化,输出的时候记得transform回来
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        img = transform(img)
        return img
