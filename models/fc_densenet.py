import torch
import torch.nn as nn

class FC_DenseNet(nn.Module):
    def __init__(self):
        super(FC_DenseNet, self).__init__()
        
        # Initial 3x3 convolutional layer
        self.conv1 = nn.Conv2d(9, 50, kernel_size=3, padding=1)
        nn.init.normal_(self.conv1.weight, mean=0.0, std=0.02)  # Initialize weights with normal distribution
        
        # BatchNorm for the initial convolutional layer
        self.bn_conv1 = nn.BatchNorm2d(50)
        nn.init.normal_(self.bn_conv1.weight, mean=1.0, std=0.02)  # Initialize BatchNorm weights with normal distribution
        
        # Downsampling block
        self.downsample_block = nn.Sequential(
            self.bn_conv1,
            nn.ReLU(inplace=True),
            nn.Conv2d(50, 150, kernel_size=3, padding=1),
            
            # BatchNorm for the first convolution in the downsampling block
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
        out1 = self.conv1(x)
        out2 = self.bn_conv1(out1)  # Applying BatchNorm after convolution
        
        # Downsampling block
        out3 = self.downsample_block(out2)
        
        # Transition down
        out4 = self.transition_down(out3)
        
        # Upsampling block
        out5 = self.upsample_block(out4)
        
        # Final convolutional layer
        out_final = self.conv_final(out5)
        
        return out_final