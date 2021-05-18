import torch
import numpy as np
import utility

def test_batch(model, batch):
    inputs, labels = batch 
    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        inputs = inputs.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(inputs)
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    print(output[0])
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    for i in output:
        probabilities = torch.nn.functional.softmax(i, dim=0)
        print(probabilities)
