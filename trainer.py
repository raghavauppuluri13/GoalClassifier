import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt
import time
import os
import copy

class Trainer:
    def __init__(self, model, hparams, vis):

        self.hparams = hparams
        self.vis = vis

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.save_path = "final_weights.pt"

        self.criterion = nn.CrossEntropyLoss()

        if hparams['optimizer'] == 'adam':
            self.optimizer = optim.Adam(self.model.fc.parameters(), lr=hparams['learning_rate'], weight_decay=hparams['weight_decay'])
        else:
            self.optimizer = optim.SGD(self.model.fc.parameters(), lr=hparams['learning_rate'], weight_decay=hparams['weight_decay'])

        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['gamma'])

    def train_model(self, dataloaders, dataset_sizes, save_path="final_weights.pt"):
        self.save_path = save_path
        since = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(self.hparams['num_epochs']):
            print('Epoch {}/{}'.format(epoch, self.hparams['num_epochs'] - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_reg_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        l2_loss = 0
                        for param in self.model.parameters() :
                            l2_loss += self.hparams['weight_decay'] * torch.sum(param ** 2)
                            
                        reg_loss = loss + l2_loss

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_reg_loss += reg_loss * inputs.size(0)

                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    self.scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_reg_loss = running_reg_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                self.vis.tb.add_scalar('{} loss'.format(phase), epoch_loss, epoch)
                self.vis.tb.add_scalar('{} reg loss'.format(phase), epoch_reg_loss, epoch)
                self.vis.tb.add_scalar('{} correct'.format(phase), running_corrects.double(), epoch)
                self.vis.tb.add_scalar('{} accuracy'.format(phase), epoch_acc, epoch)

                if phase == 'train':
                    for name, weight in self.model.named_parameters():
                        self.vis.tb.add_histogram(name,weight.data, epoch)

                        if weight.data.grad:
                            self.vis.tb.add_histogram(f'{name}.grad',weight.data.grad, epoch)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                        phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_loss = epoch_loss 
                    best_model_wts = copy.deepcopy(self.model.state_dict())

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        self.vis.tb.add_hparams(self.hparams, {
                "accuracy": best_acc,
                "loss": best_loss,
            },
        )

        torch.save(best_model_wts, save_path)

        # load best model weights
        self.model.load_state_dict(best_model_wts)

    def test(self, testloader, from_save_path=True):
        if from_save_path:
            self.model.load_state_dict(torch.load(self.save_path))
        self.model.eval()

        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for i, data in enumerate(testloader):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # calculate outputs by running images through the network
                outputs = self.model(inputs)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                print('Accuracy of the network on test batch %d: %d %%' % (
                    i, 100 * correct / total))
