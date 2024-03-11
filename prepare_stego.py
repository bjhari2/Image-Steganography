import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.reveal_network import RevealNetwork
from models.net import Net

# Importing the saved trained model
encoder = Net()
decoder = RevealNetwork()
r1 = RevealNetwork()
r2 = RevealNetwork()

encoder.load_state_dict(torch.load('steganocnn_model.pth', map_location=torch.device('cpu'))['encoder_state_dict'])
decoder.load_state_dict(torch.load('steganocnn_model.pth', map_location=torch.device('cpu'))['decoder_state_dict'])
r1.load_state_dict(torch.load('steganocnn_model.pth', map_location=torch.device('cpu'))['reveal_net1_state_dict'])
r2.load_state_dict(torch.load('steganocnn_model.pth', map_location=torch.device('cpu'))['reveal_net2_state_dict'])

# Transform function
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder('dataset', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=15)

directory = 'stego'
if not os.path.exists(directory):
    os.mkdir(directory)

save_path = 'stego/'

count = 0
iter = 0
for images, _ in train_loader:
    # images = images.cuda()
    iter += 1
    for i in range(0, 15, 3):
        count += 1
        carrier_image, secret1, secret2 = images[i], images[i+1], images[i+2]

        concatenated_image = torch.cat((carrier_image, secret1, secret2)).unsqueeze(0)

        stego = encoder(concatenated_image)

        stego = stego.squeeze(0)
        im_tensor = stego.clamp(0, 1)

        to_pil = transforms.ToPILImage()
        img = to_pil(im_tensor)
        img.save(save_path+ "stego" + str(count) + ".jpg")