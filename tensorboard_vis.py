from torch.utils.tensorboard import SummaryWriter
import torchvision

class Visualizer:
    def __init__(self, logdir="runs"):
        self.tb = SummaryWriter(logdir)

    def visualize_batch(self, batch, heading="Images"):
        images, labels = batch
        img_grid = torchvision.utils.make_grid(images)
        self.tb.add_image(heading, img_grid)

    def visualize_model(self, model, batch):
        images, labels = batch
        self.tb.add_graph(model, images)

    def close(self):
        self.tb.close()
