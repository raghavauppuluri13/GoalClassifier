
def test_batch(batch, model):
	'''
	I - 28x28 uint8 numpy array
	'''

	# test phase
	model.eval()

	# We don't need gradients for test, so wrap in 
	# no_grad to save memory
	with torch.no_grad():
			batch = batch.to(device)

			# forward propagation
			output = model( batch )

			# get prediction
			output = torch.argmax(output, 1)

	return output
