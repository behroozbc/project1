import torch
import torch.nn as nn

from AutoEncoder import AutoEncoder
from DatasetLoader import CustomDataLoader
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from AutoEncoder import AutoEncoder
from BaseFunc import plotResults, plotit, test, train
from DatasetLoader import CustomDataLoader
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


# Deeper Encoder class
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Stage 1: Three parallel CLs with 8 filters
        self.conv1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding=1),  # (1, 201, 6) -> (8, 201, 6)
                nn.MaxPool2d(2),                            # (8, 201, 6) -> (8, 100, 3)
                nn.BatchNorm2d(8),
                nn.ELU()
            ) for _ in range(3)
        ])
        
        # Stage 2: Three parallel CLs with 16 filters
        self.conv2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(24, 16, kernel_size=3, padding=1),  # (24, 100, 3) -> (16, 100, 3)
                nn.MaxPool2d(2),                              # (16, 100, 3) -> (16, 50, 1)
                nn.BatchNorm2d(16),
                nn.ELU()
            ) for _ in range(3)
        ])
        
        # Stage 3: Three parallel CLs with 32 filters
        self.conv3 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(48, 32, kernel_size=3, padding=1),  # (48, 50, 1) -> (32, 50, 1)
                nn.MaxPool2d(kernel_size=(2, 1)),             # (32, 50, 1) -> (32, 25, 1)
                nn.BatchNorm2d(32),
                nn.ELU()
            ) for _ in range(3)
        ])
        
        # Stage 4: Three parallel CLs with 64 filters
        self.conv4 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(96, 64, kernel_size=3, padding=1),  # (96, 25, 1) -> (64, 25, 1)
                nn.MaxPool2d(kernel_size=(2, 1)),             # (64, 25, 1) -> (64, 12, 1)
                nn.BatchNorm2d(64),
                nn.ELU()
            ) for _ in range(3)
        ])
        
        # Linear layer for bottleneck
        self.fc = nn.Linear(192 * 12 * 1, 128)  # (192 * 12 * 1 = 2304) -> 128
        
    def forward(self, x):
        # Stage 1
        outs1 = [branch(x) for branch in self.conv1]      # 3x (batch, 8, 100, 3)
        out1 = torch.cat(outs1, dim=1)                    # (batch, 24, 100, 3)
        
        # Stage 2
        outs2 = [branch(out1) for branch in self.conv2]   # 3x (batch, 16, 50, 1)
        out2 = torch.cat(outs2, dim=1)                    # (batch, 48, 50, 1)
        
        # Stage 3
        outs3 = [branch(out2) for branch in self.conv3]   # 3x (batch, 32, 25, 1)
        out3 = torch.cat(outs3, dim=1)                    # (batch, 96, 25, 1)
        
        # Stage 4
        outs4 = [branch(out3) for branch in self.conv4]   # 3x (batch, 64, 12, 1)
        out4 = torch.cat(outs4, dim=1)                    # (batch, 192, 12, 1)
        
        # Flatten and bottleneck
        out4_flat = out4.view(out4.size(0), -1)           # (batch, 2304)
        z = self.fc(out4_flat)                            # (batch, 128)
        return z

# Deeper Decoder class
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Linear layer to expand bottleneck
        self.fc = nn.Linear(128, 192 * 12 * 1)  # 128 -> 2304
        
        # Stage 1: Three parallel DCLs
        self.deconv1 = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(size=(25, 1)),            # (192, 12, 1) -> (192, 25, 1)
                nn.Conv2d(192, 32, kernel_size=3, padding=1),  # (192, 25, 1) -> (32, 25, 1)
                nn.BatchNorm2d(32),
                nn.ELU()
            ) for _ in range(3)
        ])
        
        # Stage 2: Three parallel DCLs
        self.deconv2 = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(size=(50, 1)),            # (96, 25, 1) -> (96, 50, 1)
                nn.Conv2d(96, 16, kernel_size=3, padding=1),  # (96, 50, 1) -> (16, 50, 1)
                nn.BatchNorm2d(16),
                nn.ELU()
            ) for _ in range(3)
        ])
        
        # Stage 3: Three parallel DCLs
        self.deconv3 = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(size=(100, 3)),           # (48, 50, 1) -> (48, 100, 3)
                nn.Conv2d(48, 8, kernel_size=3, padding=1),  # (48, 100, 3) -> (8, 100, 3)
                nn.BatchNorm2d(8),
                nn.ELU()
            ) for _ in range(3)
        ])
        
        # Stage 4: Three parallel DCLs to reconstruct input
        self.deconv4 = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(size=(201, 6)),           # (24, 100, 3) -> (24, 201, 6)
                nn.Conv2d(24, 1, kernel_size=3, padding=1),  # (24, 201, 6) -> (1, 201, 6)
                nn.BatchNorm2d(1),
                nn.ELU()
            ) for _ in range(3)
        ])
        
    def forward(self, z):
        # Expand bottleneck
        out = self.fc(z)                            # (batch, 2304)
        out = out.view(-1, 192, 12, 1)              # (batch, 192, 12, 1)
        
        # Stage 1
        outs1 = [branch(out) for branch in self.deconv1]  # 3x (batch, 32, 25, 1)
        out1 = torch.cat(outs1, dim=1)                    # (batch, 96, 25, 1)
        
        # Stage 2
        outs2 = [branch(out1) for branch in self.deconv2] # 3x (batch, 16, 50, 1)
        out2 = torch.cat(outs2, dim=1)                    # (batch, 48, 50, 1)
        
        # Stage 3
        outs3 = [branch(out2) for branch in self.deconv3] # 3x (batch, 8, 100, 3)
        out3 = torch.cat(outs3, dim=1)                    # (batch, 24, 100, 3)
        
        # Stage 4 and averaging
        outs4 = [branch(out3) for branch in self.deconv4] # 3x (batch, 1, 201, 6)
        out4 = torch.mean(torch.stack(outs4), dim=0)      # (batch, 1, 201, 6)
        return out4
dataset= CustomDataLoader(rootDirs=['E:\\Work\\University\\PR\\datas\\voice_gender_detection\\females','E:\\Work\\University\\PR\\datas\\voice_gender_detection\\males'],sr=16000,duration=0.07,HaveSaveOutput=False,isNormalized=True)
trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
valloader = DataLoader(valset, batch_size=32, shuffle=False)
learning_rate = 1e-3
num_epochs = 100
enc=Encoder()
writer = SummaryWriter(log_dir='./runs/'+datetime.now().strftime("%m-%d-%Y--%H-%M_")+'split-paraller-autoencoder-deeper')
model = AutoEncoder(enc, Decoder()).cuda()
criterion = nn.MSELoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss=[]
for epoch in range(num_epochs):
    loss.append(train(model, trainloader, optimizer, criterion, epoch,writer))
    test(model, epoch, valloader, criterion,writer, "Validation")
torch.save(enc.state_dict(), 'split-ckeck-sep_autoencoder.pth')