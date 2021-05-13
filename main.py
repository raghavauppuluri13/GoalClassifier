import os

import torch
import torchvision
from torchvision import transforms, utils, models
from torchvision.transforms import ToTensor, Compose, Normalize

from torch.optim import lr_scheduler
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import wandb

from utility import create_bc_dataset_from_videos
from dataset import TransformDataset
from model import BinaryRewardClassifier
from tensorboard_vis import Visualizer

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

    video_paths = [os.path.abspath(path) for path in
            ['Data/Clean/clean_1.avi', 'Data/Clean/clean_2.avi', 'Data/Clean/clean_3.avi']]

    target_dir = 'clean_dataset'

    split_idxs = [165, 104, 176] # determined from finding last case of occlusion or clear task completion

    #create_bc_dataset_from_videos(video_paths, split_idxs, target_dir)
    # Split dataset

    dataset = torchvision.datasets.ImageFolder(root=target_dir, transform=ToTensor())
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        (round(0.6 * len(dataset)), round(0.2 * len(dataset)), round(0.2 * len(dataset))),
    )
    data_transform = transforms.Compose(
        [
            Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    classes = dataset.class_to_idx
    train_dataset = TransformDataset(train_dataset, data_transform)

    train_dataset_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers']
    )
    valid_dataset_loader = torch.utils.data.DataLoader(
                valid_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers']
    )
    test_dataset_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers']
    )

    dataloaders = {
        "train": train_dataset_loader,
        "val": valid_dataset_loader,
        "test": test_dataset_loader,
    }



    # Model + Training

    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 10),
        nn.Softmax(1),
    )

    # Pretraining visualization

    vis = Visualizer()

    for phase in dataloaders:
        vis.visualize_batch(dataloaders[phase], phase)

    vis.visualize_model(model, dataloaders['train'])

    '''
    classifier = BinaryRewardClassifier(model, config)

    dataset_sizes = [len(train_dataset), len(valid_dataset), len(test_dataset)]
    print(dataset_sizes)

    classifier.train_model(dataloaders, dataset_sizes)
    for param in classifier.model.parameters():
        param.requires_grad = True
    classifier.train_model(dataloaders, dataset_sizes)
    '''

    vis.close()

if __name__ == "__main__":
    main()
