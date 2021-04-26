import matplotlib as plt

def visualize_batch(dataloader):
	def imshow(inp, title=None):
		"""Imshow for Tensor."""
		plt.imshow(inp.permute(1, 2, 0))
		if title is not None:
			plt.title(title)
		plt.pause(0.001)  # pause a bit so that plots are updated

	# Get a batch of training data
	inputs, classes = next(iter(dataloader))

	for i, label in enumerate(classes):
		imshow(inputs[i], title=[dataset_classes[label]])
		plt.show()
