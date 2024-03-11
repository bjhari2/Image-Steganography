import torch.nn as nn
import torch.nn.init as init

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.layer1 = nn.Conv2d(9, 50, kernel_size=3, padding=1)
        init.normal_(self.layer1.weight, mean=0.0, std=0.02)

        self.layer2 = nn.Sequential(
            # 2(BN + ReLU + 3x3 Conv)
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.Conv2d(50, 150, kernel_size=3, padding=1),

            nn.BatchNorm2d(150),
            nn.ReLU(),
            nn.Conv2d(150, 150, kernel_size=3, padding=1)
        )

        for layer in self.layer2:
            if isinstance(layer, nn.BatchNorm2d):
                init.normal_(layer.weight, mean=1.0, std=0.02)
        
        for layer in self.layer2:
            if isinstance(layer, nn.Conv2d):
                init.normal_(layer.weight, mean=0.0, std=0.02)

        self.layer3 = nn.Sequential(
            # TD block
            nn.BatchNorm2d(150),
            nn.ReLU(),
            nn.Conv2d(150, 150, kernel_size=1),
            nn.MaxPool2d(kernel_size=2, stride=2), # TD block till here

            # 2(BN + ReLU + 3x3 Conv)
            nn.BatchNorm2d(150),
            nn.ReLU(),
            nn.Conv2d(150, 150, kernel_size=3, padding=1),

            nn.BatchNorm2d(150),
            nn.ReLU(),
            nn.Conv2d(150, 250, kernel_size=3, padding=1)
        )
        
        for layer in self.layer3:
            if isinstance(layer, nn.BatchNorm2d):
                init.normal_(layer.weight, mean=1.0, std=0.02)
        
        for layer in self.layer3:
            if isinstance(layer, nn.Conv2d):
                init.normal_(layer.weight, mean=0.0, std=0.02)

        self.layer4 = nn.Sequential(
            # TD block
            nn.BatchNorm2d(250),
            nn.ReLU(),
            nn.Conv2d(250, 250, kernel_size=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 2(BN + ReLU + 3x3 Conv)
            nn.BatchNorm2d(250),
            nn.ReLU(),
            nn.Conv2d(250, 250, kernel_size=3, padding=1),

            nn.BatchNorm2d(250),
            nn.ReLU(),
            nn.Conv2d(250, 250, kernel_size=3, padding=1),

            # TU block
            nn.ConvTranspose2d(250, 350, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        
        for layer in self.layer4:
            if isinstance(layer, nn.BatchNorm2d):
                init.normal_(layer.weight, mean=1.0, std=0.02)
        
        for layer in self.layer4:
            if isinstance(layer, nn.Conv2d):
                init.normal_(layer.weight, mean=0.0, std=0.02)

        self.layer5 = nn.Sequential(
            # 2(BN + ReLU + 3x3 Conv)
            nn.BatchNorm2d(350),
            nn.ReLU(),
            nn.Conv2d(350, 350, kernel_size=3, padding=1),

            nn.BatchNorm2d(350),
            nn.ReLU(),
            nn.Conv2d(350, 350, kernel_size=3, padding=1),

            # TU block
            nn.ConvTranspose2d(350, 250, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        
        for layer in self.layer5:
            if isinstance(layer, nn.BatchNorm2d):
                init.normal_(layer.weight, mean=1.0, std=0.02)
        
        for layer in self.layer5:
            if isinstance(layer, nn.Conv2d):
                init.normal_(layer.weight, mean=0.0, std=0.02)

        self.layer6 = nn.Sequential(
            # 2(BN + ReLU + 3x3 Conv)
            nn.BatchNorm2d(250),
            nn.ReLU(),
            nn.Conv2d(250, 250, kernel_size=3, padding=1),

            nn.BatchNorm2d(250),
            nn.ReLU(),
            nn.Conv2d(250, 350, kernel_size=3, padding=1),
        )
        
        for layer in self.layer6:
            if isinstance(layer, nn.BatchNorm2d):
                init.normal_(layer.weight, mean=1.0, std=0.02)
        
        for layer in self.layer6:
            if isinstance(layer, nn.Conv2d):
                init.normal_(layer.weight, mean=0.0, std=0.02)

        self.layer7 = nn.Sequential(
            # 1x1 Conv
            nn.Conv2d(350, 3, kernel_size=3, padding=1)
        )
        
        for layer in self.layer7:
            if isinstance(layer, nn.BatchNorm2d):
                init.normal_(layer.weight, mean=1.0, std=0.02)
        
        for layer in self.layer7:
            if isinstance(layer, nn.Conv2d):
                init.normal_(layer.weight, mean=0.0, std=0.02)
    
    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)

        return out7
