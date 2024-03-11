import torch.nn as nn

class RevealNetwork(nn.Module):
    def __init__(self):
        super(RevealNetwork, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        
        return out

# Create an instance of the Reveal Network
# reveal_net = RevealNetwork()