import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.fc_densenet import FC_DenseNet
from models.reveal_network import RevealNetwork
from models.custom_loss import CustomLoss

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder('dataset', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize the models
encoder = FC_DenseNet().to(device)
decoder = RevealNetwork().to(device)
reveal_net1 = RevealNetwork().to(device)
reveal_net2 = RevealNetwork().to(device)

# Define the loss function and optimizer
custom_loss = CustomLoss(beta=0.5)
parameters = list(encoder.parameters()) + list(decoder.parameters())

# Define the optimizer for Encoder and Decoder
encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001)

# Initializing the variables
x = None
r1 = None
r2 = None
encoded_images = None
decoded_images = None
torch.autograd.set_detect_anomaly(True)
iter = 0
for images, _ in train_loader:
    iter += 1
    count = 0
    for i in range(0, 30, 3):
        count += 1
        print("Iter: ", iter, end=" ")
        print("Count: ", count)
        c, s1, s2 = images[i].clone(), images[i+1].clone(), images[i+2].clone()
        if i < 29:
            # Concatenating c, s1, s2
            x = torch.cat((c, s1, s2)).unsqueeze(0)
            print(x.shape)
            # exit()
        # Training loop
        for epoch in range(50):
            # Encoder forward pass
            c_prime = encoder(x)

            # Update the Encoder weights using the loss function ‖c - c'‖
            loss_c = torch.norm(c - c_prime)
            encoder_optimizer.zero_grad()
            loss_c.backward(retain_graph=True)
            encoder_optimizer.step()

            # Decoder forward pass, r1 and r2
            y = decoder(c_prime)
            s1_prime = reveal_net1(y)
            s2_prime = reveal_net2(y)

            # Create copies of the tensors before modifying them
            c_copy, c_prime_copy, s1_copy, s1_prime_copy, s2_copy, s2_prime_copy = c.clone(), c_prime.clone(), s1.clone(), s1_prime.clone(), s2.clone(), s2_prime.clone()
            
            loss_zeta = custom_loss(c_copy, c_prime_copy, s1_copy, s1_prime_copy, s2_copy, s2_prime_copy)
            
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss_zeta.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
    
    

#################### Converting tensors to images ####################################

# numpy_image = encoded_images.cpu().detach().numpy()

# # Rescale the values to be between 0 and 255
# rescaled_image = 255 * (numpy_image - numpy_image.min()) / (numpy_image.max() - numpy_image.min())
# rescaled_image = rescaled_image.astype(np.uint8)

# # Convert the NumPy array to a PIL Image
# pil_image = Image.fromarray(rescaled_image.transpose(1, 2, 0))

# # Save or display the PIL Image
# pil_image.save("output_image.jpg")
# pil_image.show()

#######################################################################################


# Testing loop
# Implement testing on a separate test dataset

# Save the trained model
torch.save({
    'encoder_state_dict': encoder.state_dict(),
    'decoder_state_dict': decoder.state_dict(),
    'reveal_net1_state_dict': reveal_net1.state_dict(),
    'reveal_net2_state_dict': reveal_net2.state_dict()
}, 'steganocnn_model.pth')