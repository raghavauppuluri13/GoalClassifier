import numpy as np
import glob
import torch
from skimage import io

from torch.utils.data import Dataset

class BinaryRewardClassifierDataset(Dataset):
	'Dataset class for a binary classifier for computing reward'
	
	def __init__(self, image_dir):
		'''
		ARGS:
		image_dir (str): absolute path to directory with labeled frames of format "*_{idx}.jpg" 
		'''

		self.classes = ['Fail', 'Success']
		self.image_dir = image_dir

	def __len__(self):
		return self.num_frames

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		idx = idx % len(self)

		img_glob = 	"{}/*_{}.*".format(self.image_dir, idx)
		img_path = [name for name in glob.glob(img_glob)][0]

		image = io.imread(img_path) 
		label = int(img_path.split('/')[-1].split('_')[0])

		sample = { "image": image, "label": label } 

		return sample

class TransformDataset(Dataset):
	'Wrapper class to transform binary classfier dataset'
	def __init__(self, dataset, transform=None):
		self.dataset = dataset
		self.transform = transform

	def __len__(self):
		return len(self.dataset)

	def __getitem(self, idx):
		if self.transform:
			image = self.transform(self.dataset[idx])	

		self.dataset[idx]
