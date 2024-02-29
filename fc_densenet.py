import torch
import torch.nn as nn

class FC_DenseNet(nn.Module):
    def __init__(self):
        super(FC_DenseNet, self).__init__()
        
        # Initial 3x3 convolutional layer
        self.conv1 = nn.Conv2d(3, 50, kernel_size=3, padding=1)
        
        # Downsampling block
        self.downsample_block = nn.Sequential(
            nn.BatchNorm2d(50),
            nn.ReLU(inplace=True),
            nn.Conv2d(50, 150, kernel_size=3, padding=1),
            nn.BatchNorm2d(150),
            nn.ReLU(inplace=True),
            nn.Conv2d(150, 250, kernel_size=3, padding=1),
            nn.BatchNorm2d(250),
            nn.ReLU(inplace=True),
            nn.Conv2d(250, 350, kernel_size=3, padding=1)
        )
        
        # Transition down
        self.transition_down = nn.Sequential(
            nn.Conv2d(350, 250, kernel_size=1),
            nn.BatchNorm2d(250),
            nn.ReLU(inplace=True),
            nn.Conv2d(250, 350, kernel_size=3, padding=1)
        )
        
        # Upsampling block
        self.upsample_block = nn.Sequential(
            nn.BatchNorm2d(350),
            nn.ReLU(inplace=True),
            nn.Conv2d(350, 250, kernel_size=3, padding=1),
            nn.BatchNorm2d(250),
            nn.ReLU(inplace=True),
            nn.Conv2d(250, 350, kernel_size=3, padding=1)
        )
        
        # Final 1x1 convolutional layer
        self.conv_final = nn.Conv2d(350, 3, kernel_size=1)
        
    def forward(self, x):
        # Initial convolutional layer
        out = self.conv1(x)
        
        # Downsampling block
        out1 = self.downsample_block(out)
        
        # Transition down
        out2 = self.transition_down(out1)
        
        # Upsampling block
        out3 = self.upsample_block(out2)
        
        # Final convolutional layer
        out_final = self.conv_final(out3)
        
        return out_final

# Instantiate the FC-DenseNet model
# model = FC_DenseNet()