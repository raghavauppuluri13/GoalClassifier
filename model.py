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
import wandb

class BinaryRewardClassifier:
	def __init__(self, model, config):

		self.config = config
		wandb.init(config=self.config)

		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.model = model.to(self.device)

		self.criterion = nn.CrossEntropyLoss()

		if config['optimizer'] == 'adam':
			self.optimizer = optim.Adam(self.model.fc.parameters(), lr=config['learning_rate'])
		else:
			self.optimizer = optim.SGD(self.model.fc.parameters(), lr=config['learning_rate'], momentum=0.9)

		self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=config['step_size'], gamma=config['gamma'])

	def train_model(self, dataloaders, dataset_sizes, save_path="final_weights.pt"):
		since = time.time()

		best_model_wts = copy.deepcopy(self.model.state_dict())
		best_acc = 0.0

		for epoch in range(self.config['num_epochs']):
			print('Epoch {}/{}'.format(epoch, self.config['num_epochs'] - 1))
			print('-' * 10)

			# Each epoch has a training and validation phase
			for phase in ['train', 'val']:
				if phase == 'train':
					self.model.train()  # Set model to training mode
				else:
					self.model.eval()   # Set model to evaluate mode

				running_loss = 0.0
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

						# backward + optimize only if in training phase
						if phase == 'train':
							loss.backward()
							self.optimizer.step()

					# statistics
					running_loss += loss.item() * inputs.size(0)

					running_corrects += torch.sum(preds == labels.data)
				if phase == 'train':
					self.scheduler.step()

				epoch_loss = running_loss / dataset_sizes[phase]

				epoch_acc = running_corrects.double() / dataset_sizes[phase]

				wandb.log({"{} loss".format(phase): epoch_loss})
				wandb.log({"{} accuracy".format(phase): epoch_acc})

				print('{} Loss: {:.4f} Acc: {:.4f}'.format(
						phase, epoch_loss, epoch_acc))

				# deep copy the model
				if phase == 'val' and epoch_acc > best_acc:
					best_acc = epoch_acc
					best_model_wts = copy.deepcopy(self.model.state_dict())

		time_elapsed = time.time() - since
		print('Training complete in {:.0f}m {:.0f}s'.format(
				time_elapsed // 60, time_elapsed % 60))
		print('Best val Acc: {:4f}'.format(best_acc))

		torch.save(self.model.state_dict(), save_path)

		# load best model weights
		self.model.load_state_dict(best_model_wts)
		return self.model

	def visualize_model(self, dataloaders, class_names, num_images=4):
		was_training = self.model.training
		self.model.eval()
		images_so_far = 0
		fig = plt.figure()

		with torch.no_grad():
			for i, (inputs, labels) in enumerate(dataloaders['val']):
				inputs = inputs.to(self.device)
				labels = labels.to(self.device)

				outputs = self.model(inputs)
				_, preds = torch.max(outputs, 1)

				for j in range(inputs.size()[0]):
					images_so_far += 1
					ax = plt.subplot(num_images//2, 2, images_so_far)
					ax.axis('off')
					ax.set_title('predicted: {}'.format(class_names[preds[j]]))
					imshow(inputs.cpu().data[j])

					if images_so_far == num_images:
						self.model.train(mode=was_training)
						return
					self.model.train(mode=was_training)
