import numpy as np
import torch

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return (torch.from_numpy(image), label)

class VaryLighting(object):
		def __init__(self):	
			pass
		def __call__(self):	
			pass
