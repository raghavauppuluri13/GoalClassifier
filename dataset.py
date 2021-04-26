import numpy as np
import glob
import torch
from PIL import Image
import pathlib
import cv2

import torchvision
from skimage import io
import skvideo.io  
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

import os

class BinaryRewardClassifierDataset(Dataset):
	'Dataset class for data preprocessing on video for a binary classifier for computing reward'
	
	def __init__(self, video_path, target_dir=None, success_frame_idx=None, transform=None):
		'''
		ARGS:
		video_path: absolute path to video
		target_dir: absolute path to directory to save extracted frames
		success_frame (int): first frame to include as success label
		transform (torchvision.transform): transforms on data
		'''

		self.video_path = video_path
		self.transform = transform
		self.num_frames = None
		self.classes = ['Fail', 'Success']

		if success_frame_idx is not None:
			self.success_frame_idx = success_frame_idx
		else:
			self.success_frame_idx = self.num_frames - 1

		if target_dir is not None:
			self.target_dir = target_dir
		else:
			self.target_dir = os.path.join(os.getcwd(), 'extracted_frames')

		self._extract_and_write_frames()

		if success_frame_idx >= len(self):
			raise IndexError('success frame is out of range')

	def _extract_and_write_frames(self):
		vidcap = cv2.VideoCapture(self.video_path)
		num_frames = 0
		success = True
		pathlib.Path(self.target_dir).mkdir(parents=True, exist_ok=True) 
		
		while success:
			success, image = vidcap.read()
			if(success):
				num_frames += 1
				label = int(num_frames - 1 >= self.success_frame_idx)
				file_name = str(label) + '_' + str(num_frames - 1) + ".jpg"
				path = os.path.join(self.target_dir, file_name)
				cv2.imwrite(path, image)
			else:
				print("Error reading frame: ", num_frames)
			if vidcap.get(cv2.CAP_PROP_POS_FRAMES) == vidcap.get(cv2.CAP_PROP_FRAME_COUNT):
				break
		self.num_frames = num_frames 

	def __len__(self):
		return self.num_frames

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		idx = idx % len(self)

		img_glob = 	"{}/*_{}.*".format(self.target_dir, idx)
		img_path = [name for name in glob.glob(img_glob)][0]

		image = io.imread(img_path) 
		label = int(img_path.split('/')[-1].split('_')[0])

		sample = (image, label)

		if self.transform:
			sample = self.transform(sample)

		return sample
