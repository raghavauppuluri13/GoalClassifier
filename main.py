from dataset import BinaryRewardClassifierDataset 
import torch
from transforms import ToTensor 
from model import BinaryRewardClassifier
from torchvision import transforms, utils, models
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import wandb

import os

def main():

	# Hyperparameters
	config = {
		'num_epochs': 50,
		'batch_size': 4,
		'num_workers': 2,
		'learning_rate': 1e-3,
		'optimizer': 'adam',
		'step_size': 7,
		'gamma': 0.1,
	}

	# Data loading + Preprocessing

	video_paths = ['Data/Clean/clean_1.avi', 'Data/Clean/clean_2.avi', 'Data/Clean/clean_3.avi']
	video_paths = [ os.path.abspath(path) for path in video_paths ]

	data_dirs = ['data/train', 'data/test', 'data/val']

	success_frame_idxs = [165, 104, 176] # determined from finding last case of occlusion or clear task completion

	datasets = [BinaryRewardClassifierDataset(video_paths[i], data_dirs[i], success_frame_idxs[i], transform=transforms.Compose([ToTensor()])) 
								for i in range(len(video_paths))]

	dataset_sizes = {x: len(datasets[i]) for i, x in enumerate(['train', 'val'])}
	class_names = datasets[0].classes

	dataset_names = ['train', 'test', 'val']
	dataloaders = {}

	for i, name in enumerate(dataset_names):
		dataloaders[name] = torch.utils.data.DataLoader(datasets[i], batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
	
	classifier = BinaryRewardClassifier(config)

	classifier.train_model(dataloaders, dataset_sizes)

	#classifier.visualize_model(dataloaders,class_names)
	#plt.ioff()
	#plt.show()

if __name__ == "__main__":
    main()
