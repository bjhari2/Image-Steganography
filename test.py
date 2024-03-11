from PIL import Image
import torch
from torchvision import transforms
from models.reveal_network import RevealNetwork
from models.net import Net

# Importing the saved trained model
encoder = Net()
decoder = RevealNetwork()
r1 = RevealNetwork()
r2 = RevealNetwork()

encoder.load_state_dict(torch.load('steganocnn_model.pth')['encoder_state_dict'])
decoder.load_state_dict(torch.load('steganocnn_model.pth')['decoder_state_dict'])
r1.load_state_dict(torch.load('steganocnn_model.pth')['reveal_net1_state_dict'])
r2.load_state_dict(torch.load('steganocnn_model.pth')['reveal_net2_state_dict'])

# Transform function
transform = transforms.Compose([
    transforms.ToTensor()
])

# Getting image inputs for Steganography
carrier_image = transform(Image.open("dog11.jpg"))
secret_image1 = transform(Image.open("dog8.jpg"))
secret_image2 = transform(Image.open("dog7.jpg"))

# Concatenating the images
x = torch.cat((carrier_image, secret_image1, secret_image2)).unsqueeze(0)

intermediate = encoder(x)
rn1 = r1(intermediate)
rn2 = r2(intermediate)

print("\n\n\n")
print(intermediate.size())
print("\n\n\n")
intermediate = intermediate.squeeze(0)
im_tensor = intermediate.clamp(0, 1)

to_pil = transforms.ToPILImage()
img = to_pil(im_tensor)
img.save("intermediate.jpg")


print("\n\n\n")
print(intermediate.size())
print("\n\n\n")
r1 = rn1.squeeze(0)
im_tensor = r1.clamp(0, 1)

to_pil = transforms.ToPILImage()
img = to_pil(im_tensor)
img.save("r1.jpg")

r2 = rn2.squeeze(0)
im_tensor = r2.clamp(0, 1)

to_pil = transforms.ToPILImage()
img = to_pil(im_tensor)
img.save("r2.jpg")