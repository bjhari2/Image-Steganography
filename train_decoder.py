import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.net import Net
from models.reveal_network import RevealNetwork
from models.custom_loss import CustomLoss

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder('dataset', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=15, shuffle=True)

# Initialize the models
encoder = Net().to(device)
decoder = RevealNetwork().to(device)
reveal_net1 = RevealNetwork().to(device)
reveal_net2 = RevealNetwork().to(device)

# Define the loss function and optimizer
custom_loss = CustomLoss(beta=0.75)
parameters = list(decoder.parameters())

# Define the optimizer for Decoder
decoder_optimizer = optim.Adam(decoder.parameters(), lr=1e-4)

torch.autograd.set_detect_anomaly(True)

iter = 0
for images, _ in train_loader:
    images = images.cuda()
    iter += 1
    count = 0
    for i in range(0, 15, 3):
        count += 1
        print("Iter: ", iter, end=" ")
        print("Count: ", count)
        c, s1, s2 = images[i].clone(), images[i+1].clone(), images[i+2].clone()
                
        # Concatenating c, s1, s2
        x = torch.cat((c, s1, s2)).unsqueeze(0)

        # c.requires_grad = True
        # s1.requires_grad = True
        # s2.requires_grad = True
        # x.requires_grad = True

        # Training loop
        for epoch in range(200):
            stego = encoder(x)

            # Decoder forward pass, extracted s1 and s2
            # y = decoder(c_prime)
            s1_prime = reveal_net1(stego)
            s2_prime = reveal_net2(stego)

            # Create copies of the tensors before modifying them
            c_copy, c_prime_copy, s1_copy, s1_prime_copy, s2_copy, s2_prime_copy = c.clone(), stego.clone(), s1.clone(), s1_prime.clone(), s2.clone(), s2_prime.clone()
            
            # Update the Encoder and Decoder weights through ζ.
            loss_zeta = custom_loss(c_copy, c_prime_copy, s1_copy, s1_prime_copy, s2_copy, s2_prime_copy)
            decoder_optimizer.zero_grad()
            loss_zeta.backward()
            decoder_optimizer.step()

# Save the trained model
torch.save({
    'decoder_state_dict': decoder.state_dict(),
    'reveal_net1_state_dict': reveal_net1.state_dict(),
    'reveal_net2_state_dict': reveal_net2.state_dict()
}, 'decoder_model.pth')