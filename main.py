import sys
import cv2
import torch
from torchvision import transforms
from models.fc_densenet import FC_DenseNet
from models.reveal_network import RevealNetwork

# Importing the saved trained model
encoder_model = None
decoder_model = None

# Getting image inputs for Steganography
carrier_image = cv2.imread(sys.argv[0])
secret_image1 = cv2.imread(sys.argv[1])
secret_image2 = cv2.imread(sys.argv[2])

