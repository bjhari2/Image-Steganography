import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self, beta):
        super(CustomLoss, self).__init__()
        self.beta = beta

    def forward(self, c, c_prime, s1, s1_prime, s2, s2_prime):
        loss_c = torch.norm(c - c_prime)
        loss_s1 = torch.norm(s1 - s1_prime)
        loss_s2 = torch.norm(s2 - s2_prime)

        total_loss = loss_c + self.beta * (loss_s1 + loss_s2)
        print(total_loss)
        return total_loss