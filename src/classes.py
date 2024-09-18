import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn
import numpy as np

class MonetDataset(Dataset):
    def __init__(self, root, transform = None):
        self.root = root

        self.std = [0.5, 0.5, 0.5]
        self.mean = [0.5, 0.5, 0.5]

        self.transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images
        transforms.ToTensor(),          # Convert images to PyTorch tensors
        transforms.Normalize(mean=self.mean, std=self.std)  # Normalize images to [-1, 1]
        ])

        self.images = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        path = os.path.join(self.root, self.images[idx])
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        return image
    
    def visualise(self, idx):
        tensor = self[idx]
        tensor_np = tensor.numpy().transpose(1,2,0)
        tensor_np = (tensor_np * self.std + self.mean)  # Reverse normalization (assuming mean=0.5 and std=0.5)
        tensor_np = np.clip(tensor_np, 0, 1)
        return tensor_np

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.channel_size = 64
        self.main = nn.Sequential(
            # Input is Z, going into a convolution
            nn.ConvTranspose2d(100, self.channel_size * 32, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.channel_size * 32),
            nn.ReLU(True),
            
            # State size: (channel_size*16) x 4 x 4
            nn.ConvTranspose2d(self.channel_size * 32, self.channel_size * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_size * 16),
            nn.ReLU(True),

            # State size: (channel_size*8) x 8 x 8
            nn.ConvTranspose2d(self.channel_size * 16, self.channel_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_size * 8),
            nn.ReLU(True),

            # State size: (channel_size*4) x 16 x 16
            nn.ConvTranspose2d(self.channel_size * 8, self.channel_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_size * 4),
            nn.ReLU(True),

            # State size: (channel_size*2) x 32 x 32
            nn.ConvTranspose2d(self.channel_size * 4, self.channel_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_size*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.channel_size * 2, self.channel_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_size),
            nn.ReLU(True),

            # State size: (channel_size) x 64 x 64
            nn.ConvTranspose2d(self.channel_size, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output state size: (3) x 128 x 128 (with padding)
        )
    
    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.channel_size = 64
        self.main = nn.Sequential(
            # Input is (3) x 256 x 256
            nn.Conv2d(3, self.channel_size, 4, 2, 1, bias=False),  # -> (64) x 128 x 128
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.channel_size, self.channel_size * 2, 4, 2, 1, bias=False),  # -> (128) x 64 x 64
            nn.BatchNorm2d(self.channel_size * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.channel_size * 2, self.channel_size * 4, 4, 2, 1, bias=False),  # -> (256) x 32 x 32
            nn.BatchNorm2d(self.channel_size * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.channel_size * 4, self.channel_size * 8, 4, 2, 1, bias=False),  # -> (512) x 16 x 16
            nn.BatchNorm2d(self.channel_size * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.channel_size * 8, self.channel_size * 16, 4, 2, 1, bias=False),  # -> (1024) x 8 x 8
            nn.BatchNorm2d(self.channel_size * 16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.channel_size * 16, self.channel_size * 32, 4, 2, 1, bias=False),  # -> (2048) x 4 x 4
            nn.BatchNorm2d(self.channel_size * 32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.channel_size * 32, 1, 4, 1, 0, bias=False),  # Kernel size 4, stride 1, padding 0
            nn.Sigmoid()
            # Output size: 1
        )
    
    def forward(self, input):
        return self.main(input)