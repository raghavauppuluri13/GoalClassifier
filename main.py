from dataset import BinaryRewardClassifierDataset 
import torch
from transforms import ToTensor 
from model import train_model, visualize_model
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
		'epochs': 25,
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

	dataset_names = ['train', 'test', 'val']
	dataloaders = {}

	for i, name in enumerate(dataset_names):
		dataloaders[name] = torch.utils.data.DataLoader(datasets[i], batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# Model + Training

	wandb.init(config=config)

	model = models.resnet50(pretrained=True)
	for param in model.parameters():
		param.requires_grad = False

	num_ftrs = model.fc.in_features
	model.fc = nn.Sequential(
			nn.Linear(num_ftrs, 2),
			nn.Softmax(1),
	)

	model = model.to(device)

	criterion = nn.CrossEntropyLoss()

	optimizer_conv = optim.SGD(model.fc.parameters(), lr=config['learning_rate'], momentum=0.9)

	# Decay LR by a factor of 0.1 every 7 epochs
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=config['step_size'], gamma=config['gamma'])	

	train_model(model, dataloaders, criterion, optimizer_conv, exp_lr_scheduler, device, num_epochs=config['epochs'])

	visualize_model(model, dataloaders, device, num_images=6)

	plt.ioff()
	plt.show()

	save_path = 'last_weights.pt'
	torch.save(model.state_dict(), save_path)	

if __name__ == "__main__":
    main()
