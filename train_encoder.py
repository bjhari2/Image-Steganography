import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.net import Net
# from models.custom_loss import CustomLoss

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder('dataset', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=15, shuffle=True)

# Initialize the models
encoder = Net().to(device)

# Define the optimizer for Encoder
encoder_optimizer = optim.Adam(encoder.parameters(), lr=1e-4)

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
            # Encoder forward pass
            stego = encoder(x)

            # Update the Encoder weights using the loss function ‖c - c'‖
            loss_c = torch.norm(c - stego)
            encoder_optimizer.zero_grad()
            loss_c.backward(retain_graph=True)
            encoder_optimizer.step()

# Save the trained model
torch.save({
    'encoder_state_dict': encoder.state_dict(),
}, 'encoder_model.pth')