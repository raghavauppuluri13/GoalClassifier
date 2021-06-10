import numpy as np
import glob
import torch 
from skimage import io

from torch.utils.data import Dataset

class TransformDataset(Dataset):
    'Wrapper class to transform goal classfier dataset'
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if self.transform:
            image, target = self.dataset[idx]
            image = self.transform(image)
            sample = (image, target)

        return sample
