from dataset import BinaryClassifierDataset 
import torch
from transforms import ToTensor 
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import numpy as np

import os

train_video_path = os.path.abspath('Data/Clean/clean_1.avi')
train_success_frame_idx = 165 # determined from finding last case of occlusion or clear task completion

train_dataset = BinaryClassifierDataset(train_video_path, "clean_1_dataset", success_frame_idx=train_success_frame_idx, transform=transforms.Compose([ToTensor()]))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4,
                                          shuffle=True, num_workers=2)

dataset_classes = train_dataset.classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def imshow(inp, title=None):
	"""Imshow for Tensor."""
	plt.imshow(inp.permute(1, 2, 0))
	if title is not None:
		plt.title(title)
	plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(train_loader))

print(classes)

for i, label in enumerate(classes):
	imshow(inputs[i], title=[dataset_classes[label]])
	plt.show()
