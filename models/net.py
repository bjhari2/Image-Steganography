import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.layer1 = nn.Conv2d(9, 50, kernel_size=3, padding=1)

        self.layer2 = nn.Sequential(
            nn.BatchNorm2d(150),
            nn.ReLU(),
            nn.Conv2d(150, )
        )