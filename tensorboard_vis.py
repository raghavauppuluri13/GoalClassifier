from torch.utils.tensorboard import SummaryWriter
import torchvision

class Visualizer:
    def __init__(self, logdir="runs/run_1"):
        self.writer = SummaryWriter(logdir)

    def visualize_batch(self, dataloader, heading="Images"):
        images, labels = next(iter(dataloader))
        img_grid = torchvision.utils.make_grid(images)
        self.writer.add_image(heading, img_grid)

    def visualize_model(self, model, dataloader):
        images, labels = next(iter(dataloader))
        self.writer.add_graph(model, images)

    def close(self):
        self.writer.close()
