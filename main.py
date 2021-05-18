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

from utility import create_bc_dataset_from_videos, dataset_per_channel_mean
from dataset import TransformDataset
from model import BinaryRewardClassifier
from tensorboard_vis import Visualizer

from test import test_batch
def main():

    # Hyperparameters
    hparams = {
        'num_epochs': 50,
        'layer_size': 50,
        'batch_size': 5,
        'num_workers': 2,
        'learning_rate': 1e-3,
        'dropout_prob': 0.5,
        'weight_decay': 0.001,
        'optimizer': 'adam',
        'step_size': 7,
        'gamma': 0.1,
    }

    torch.manual_seed(0)

    # Data loading + Preprocessing

    video_paths = [os.path.abspath(path) for path in
            ['Data/Clean/clean_1.avi', 'Data/Clean/clean_2.avi', 'Data/Clean/clean_3.avi']]

    target_dir = 'clean_dataset'

    split_idxs = [165, 104, 176] # determined from finding last case of occlusion or clear task completion

    #create_bc_dataset_from_videos(video_paths, split_idxs, target_dir)
    # Split dataset

    dataset = torchvision.datasets.ImageFolder(root=target_dir, transform=ToTensor())
    print(len(dataset))
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        (round(0.4 * len(dataset)), round(0.4 * len(dataset)), round(0.199 * len(dataset))),
        #(5, 10, len(dataset) - 15),
    )

    mean = dataset_per_channel_mean(dataset)
    data_transform = transforms.Compose(
        [
            Normalize(
                mean=mean, std=[1,1,1]
            ),
        ]
    )

    classes = dataset.class_to_idx
    train_dataset = TransformDataset(train_dataset, data_transform)

    train_dataset_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=hparams['batch_size'], shuffle=True, num_workers=hparams['num_workers']
    )
    valid_dataset_loader = torch.utils.data.DataLoader(
                valid_dataset, batch_size=hparams['batch_size'], shuffle=True, num_workers=hparams['num_workers']
    )
    test_dataset_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=hparams['batch_size'], shuffle=False, num_workers=hparams['num_workers']
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
        nn.Linear(num_ftrs, hparams['layer_size']),
        nn.BatchNorm1d(hparams['layer_size']),
        nn.ReLU(),
        nn.Dropout(hparams['dropout_prob']),
        nn.Linear(hparams['layer_size'], 2),
    )

    def init_normal(m):
        if type(m) == nn.Linear:
            nn.init.uniform_(m.weight)

    # use the modules apply function to recursively apply the initialization
    #model.apply(init_normal)

    # Pretraining visualization
    print(model)

    vis = Visualizer()
    phase = 'train' 
    batch = next(iter(dataloaders[phase]))
    vis.visualize_batch(batch, phase)
    vis.visualize_model(model, batch)

    classifier = BinaryRewardClassifier(model, hparams, vis)

    dataset_sizes = {"train":len(train_dataset), "val":len(valid_dataset), "test":len(test_dataset)}
    print(dataset_sizes)
    classifier.train_model(dataloaders, dataset_sizes)

    vis.close()

if __name__ == "__main__":
    main()
