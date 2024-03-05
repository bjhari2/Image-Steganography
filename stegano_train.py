import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from PIL import Image
from models.fc_densenet import FC_DenseNet
from models.reveal_network import RevealNetwork

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Initialize the convolution operation weight with Normal(0.0, 0.02)
conv_weight = nn.init.normal_(torch.empty(out_channels, in_channels, kernel_size, kernel_size), mean=0.0, std=0.02)

# Initialize the BatchNorm operation weight with Normal(1.0, 0.02)
batchnorm_weight = nn.init.normal_(torch.empty(num_features), mean=1.0, std=0.02)

dataset = torchvision.datasets.ImageFolder(root='dataset', transform=transform)

data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=64, shuffle=True)

rn = RevealNetwork()

print(rn)