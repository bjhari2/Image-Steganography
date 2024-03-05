import sys
import cv2
from PIL import Image
import torch
from torchvision import transforms
from models.fc_densenet import FC_DenseNet
from models.reveal_network import RevealNetwork

# Importing the saved trained model
encoder = FC_DenseNet()
decoder = RevealNetwork()
r1 = RevealNetwork()
r2 = RevealNetwork()

encoder.load_state_dict(torch.load('steganocnn_model.pth')['encoder_state_dict'])
decoder.load_state_dict(torch.load('steganocnn_model.pth')['decoder_state_dict'])
r1.load_state_dict(torch.load('steganocnn_model.pth')['reveal_net1_state_dict'])
r2.load_state_dict(torch.load('steganocnn_model.pth')['reveal_net2_state_dict'])

# Transform function
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# Getting image inputs for Steganography
carrier_image = transform(Image.open("cat1.jpg"))
secret_image1 = transform(Image.open("cat2.jpg"))
secret_image2 = transform(Image.open("cat3.jpg"))

# Concatenating the images
x = torch.cat((carrier_image, secret_image1, secret_image2)).unsqueeze(0)

intermediate = encoder(x)
output = decoder(intermediate)
rn1 = r1(output)
rn2 = r2(output)

print("\n\n\n")
print(intermediate.size())
print("\n\n\n")
intermediate = intermediate.squeeze(0)
im_tensor = intermediate.clamp(0, 1)

to_pil = transforms.ToPILImage()
img = to_pil(im_tensor)
img.save("intermediate.jpg")


print("\n\n\n")
print(output.size())
print("\n\n\n")
# r1 = r1.squeeze(0)
# im_tensor = r1.clamp(0, 1)

to_pil = transforms.ToPILImage()
img = to_pil(im_tensor)
img.save("r1.jpg")

# r2 = r2.squeeze(0)
# im_tensor = r2.clamp(0, 1)

to_pil = transforms.ToPILImage()
img = to_pil(im_tensor)
img.save("r2.jpg")