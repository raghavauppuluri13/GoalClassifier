import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

import torchvision
from torchvision import transforms, utils, models
from torchvision.transforms import ToTensor, Compose, Normalize

from dataset import TransformDataset
from trainer import Trainer
from tensorboard_vis import Visualizer

def main():

    # Hyperparameters
    hparams = {
        'num_epochs': 30,
        'layer_size': 50,
        'batch_size': 30,
        'num_workers': 2,
        'learning_rate': 1e-4,
        'dropout_prob': 0.5,
        'weight_decay': 0.001,
        'optimizer': 'adam',
        'step_size': 7,
        'gamma': 0.1,
    }

    torch.manual_seed(32)

    # Data loading + Preprocessing

    data_dir = "clean_dataset"

    dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=ToTensor())
    test_dataset = torchvision.datasets.ImageFolder(root='test_' + data_dir, transform=ToTensor())
    train_dataset, valid_dataset = torch.utils.data.random_split(
        dataset,
        (round(0.6 * len(dataset)), round(0.4 * len(dataset)), 
           # round(0.2 * len(dataset))
        ),
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), num_workers=1)
    data = next(iter(train_loader))[0]
    train_mean = data.mean(dim=(0,2,3))
    train_std = data.std(dim=(0,2,3))

    '''
    TODO: Add gaussian noise transform
    '''
    data_transform = transforms.Compose(
        [
            Normalize(
               mean=train_mean,
               std=train_std 
            ),
        ]
    )

    classes = dataset.class_to_idx

    '''
    BUG: Somehow trains well without normalization and doesn't train at all with normalization
    train_dataset = TransformDataset(train_dataset, data_transform)
    '''

    train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=hparams['batch_size'], shuffle=True, num_workers=hparams['num_workers']
    )
    valid_loader = torch.utils.data.DataLoader(
                valid_dataset, batch_size=hparams['batch_size'], shuffle=True, num_workers=hparams['num_workers']
    )
    test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=hparams['batch_size'], shuffle=False, num_workers=hparams['num_workers']
    )

    dataloaders = {
        "train": train_loader,
        "val": valid_loader,
        "test": test_loader,
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
        nn.Linear(hparams['layer_size'], hparams['layer_size']),
        nn.BatchNorm1d(hparams['layer_size']),
        nn.ReLU(),
        nn.Dropout(hparams['dropout_prob']),
        nn.Linear(hparams['layer_size'], 2),
    )

    def init_normal(m):
        if type(m) == nn.Linear:
            nn.init.uniform_(m.weight)

    # use the modules apply function to recursively apply the initialization
    # model.apply(init_normal)

    # Pretraining visualization

    vis = Visualizer()
    phase = 'train' 
    batch = next(iter(dataloaders[phase]))
    vis.visualize_batch(batch, phase)
    vis.visualize_model(model, batch)

    trainer = Trainer(model, hparams, vis)

    dataset_sizes = {"train":len(train_dataset), "val":len(valid_dataset), "test":len(test_dataset)}
    trainer.train_model(dataloaders, dataset_sizes)
    trainer.test(dataloaders['test'])

    vis.close()

if __name__ == "__main__":
    main()
