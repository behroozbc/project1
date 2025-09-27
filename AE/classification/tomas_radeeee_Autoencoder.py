import torch
import torch.nn as nn
from AutoEncoder import AutoEncoder
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from AutoEncoder import AutoEncoder
import sys
sys.path.append('e:\\Work\\University\\PR\\project1')
from BaseFunc import plotResults, plotit, test, train
from DatasetLoader import CustomDataLoader
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch
import torch.nn as nn

# Encoder class
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # First part: Three parallel CLs with 16 filters, two Conv2d layers each
        self.conv1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),  # (1, 257, 51) -> (16, 257, 51)
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.Conv2d(16, 16, kernel_size=3, padding=1), # (16, 257, 51) -> (16, 257, 51)
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.MaxPool2d(2)                              # (16, 257, 51) -> (16, 128, 25)
            ) for _ in range(3)
        ])
        
        # Second part: Three parallel CLs with 32 filters, two Conv2d layers each
        self.conv2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(48, 32, kernel_size=3, padding=1), # (48, 128, 25) -> (32, 128, 25)
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1), # (32, 128, 25) -> (32, 128, 25)
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.MaxPool2d(2)                              # (32, 128, 25) -> (32, 64, 12)
            ) for _ in range(3)
        ])
        
        # Third part: Linear layer for bottleneck
        self.fc = nn.Linear(96 * 64 * 12, 512)  # 73728 -> 256
        
    def forward(self, x):
        # First parallel convolutions
        outs1 = [branch(x) for branch in self.conv1]  # 3 x (batch_size, 16, 128, 25)
        out1 = torch.cat(outs1, dim=1)                # (batch_size, 48, 128, 25)
        
        # Second parallel convolutions
        outs2 = [branch(out1) for branch in self.conv2]  # 3 x (batch_size, 32, 64, 12)
        out2 = torch.cat(outs2, dim=1)                   # (batch_size, 96, 64, 12)
        
        # Flatten and apply linear layer
        out2_flat = out2.view(out2.size(0), -1)  # (batch_size, 73728)
        z = self.fc(out2_flat)                   # (batch_size, 256)
        return z

# Decoder class
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # First part: Linear layer to expand bottleneck
        self.fc = nn.Linear(512, 96 * 64 * 12)  # 256 -> 73728
        
        # Second part: Three parallel deconvolutional layers with 32 filters
        self.deconv1 = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(size=(128, 25)),           # (96, 64, 12) -> (96, 128, 25)
                nn.Conv2d(96, 32, kernel_size=3, padding=1), # (96, 128, 25) -> (32, 128, 25)
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1), # (32, 128, 25) -> (32, 128, 25)
                nn.BatchNorm2d(32),
                nn.ELU()
            ) for _ in range(3)
        ])
        
        # Third part: Three parallel deconvolutional layers to reconstruct input
        self.deconv2 = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(size=(257, 51)),           # (96, 128, 25) -> (96, 257, 51)
                nn.Conv2d(96, 16, kernel_size=3, padding=1), # (96, 257, 51) -> (16, 257, 51)
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.Conv2d(16, 1, kernel_size=3, padding=1),  # (16, 257, 51) -> (1, 257, 51)
                nn.BatchNorm2d(1),
                nn.ELU()
            ) for _ in range(3)
        ])
        
    def forward(self, z):
        # Expand bottleneck
        out = self.fc(z)                      # (batch_size, 73728)
        out = out.view(-1, 96, 64, 12)       # (batch_size, 96, 64, 12)
        
        # First parallel deconvolutions
        outs1 = [branch(out) for branch in self.deconv1]  # 3 x (batch_size, 32, 128, 25)
        out1 = torch.cat(outs1, dim=1)                    # (batch_size, 96, 128, 25)
        
        # Second parallel deconvolutions and averaging
        outs2 = [branch(out1) for branch in self.deconv2] # 3 x (batch_size, 1, 257, 51)
        out2 = torch.mean(torch.stack(outs2), dim=0)      # (batch_size, 1, 257, 51)
        return out2
# dataset= CustomDataLoader(rootDirs=['E:\\Work\\University\\PR\\datas\\voice_gender_detection\\females','E:\\Work\\University\\PR\\datas\\voice_gender_detection\\males'],sr=16000,duration=0.5,HaveSaveOutput=False,isNormalized=True)
# trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
# trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
# valloader = DataLoader(valset, batch_size=32, shuffle=False)
learning_rate = 1e-3
num_epochs = 150
enc=Encoder()
nowTime=datetime.now().strftime("%m-%d-%Y--%H-%M_")
writer = SummaryWriter(log_dir='./runs/'+nowTime+'rpDeee_Autoencoder')
model = AutoEncoder(enc, Decoder()).cuda()
# criterion = nn.MSELoss().cuda()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# loss=[]
# n_fft = 25  # 400 samples at 16kHz
    
# hop_length = 10
# for epoch in range(num_epochs):
#     loss.append(train(model, trainloader, optimizer, criterion, epoch,writer))
#     test(model, epoch, valloader, criterion,n_fft,hop_length,writer, "Validation")
# torch.save(enc.state_dict(), nowTime+'.pth')

# Count total parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

# Count trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params}")
