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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.elu1 = nn.ELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # Adjust residual connection if channel sizes differ
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = None
        self.elu2=nn.ELU()

    def forward(self, x):
        residual = x if self.residual is None else self.residual(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.elu2(out)
        return out
class DecoderResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_size):
        super(DecoderResidualBlock, self).__init__()
        self.upsample = nn.Upsample(size=upsample_size)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.elu1 = nn.ELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # Residual connection always needed due to channel reduction
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.elu2=nn.ELU()

    def forward(self, x):
        x_upsampled = self.upsample(x)
        residual = self.residual(x_upsampled)
        out = self.conv1(x_upsampled)
        out = self.bn1(out)
        out = self.elu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.elu2(out)
        return out
    
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Stage 1: Three parallel branches, 1 -> 16 filters
        self.stage1 = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(1, 16),           # (1, 201, 6) -> (16, 201, 6)
                nn.MaxPool2d(2)                 # (16, 201, 6) -> (16, 100, 3)
            ) for _ in range(3)
        ])
        
        # Stage 2: Three parallel branches, 48 -> 32 filters
        self.stage2 = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(48, 32),          # (48, 100, 3) -> (32, 100, 3)
                nn.MaxPool2d(2)                 # (32, 100, 3) -> (32, 50, 1)
            ) for _ in range(3)
        ])
        
        # Stage 3: Three parallel branches, 96 -> 64 filters
        self.stage3 = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(96, 64),          # (96, 50, 1) -> (64, 50, 1)
                nn.MaxPool2d(kernel_size=(2, 1))  # (64, 50, 1) -> (64, 25, 1)
            ) for _ in range(3)
        ])
        
        # Stage 4: Three parallel branches, 192 -> 128 filters
        self.stage4 = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(192, 128),        # (192, 25, 1) -> (128, 25, 1)
                nn.MaxPool2d(kernel_size=(2, 1))  # (128, 25, 1) -> (128, 12, 1)
            ) for _ in range(3)
        ])
        
        # Linear layer for bottleneck, 384 * 12 * 1 = 4608
        self.fc = nn.Linear(384 * 12 * 1, 512)  # 4608 -> 512
        
    def forward(self, x):
        # Stage 1
        outs1 = [branch(x) for branch in self.stage1]  # 3x (batch, 16, 100, 3)
        out1 = torch.cat(outs1, dim=1)                 # (batch, 48, 100, 3)
        
        # Stage 2
        outs2 = [branch(out1) for branch in self.stage2]  # 3x (batch, 32, 50, 1)
        out2 = torch.cat(outs2, dim=1)                    # (batch, 96, 50, 1)
        
        # Stage 3
        outs3 = [branch(out2) for branch in self.stage3]  # 3x (batch, 64, 25, 1)
        out3 = torch.cat(outs3, dim=1)                    # (batch, 192, 25, 1)
        
        # Stage 4
        outs4 = [branch(out3) for branch in self.stage4]  # 3x (batch, 128, 12, 1)
        out4 = torch.cat(outs4, dim=1)                    # (batch, 384, 12, 1)
        
        # Flatten and bottleneck
        out4_flat = out4.view(out4.size(0), -1)           # (batch, 4608)
        z = self.fc(out4_flat)                            # (batch, 512)
        return z
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Linear layer to expand bottleneck
        self.fc = nn.Linear(512, 384 * 12 * 1)  # 512 -> 4608
        
        # Stage 1: Three parallel branches, 384 -> 64
        self.stage1 = nn.ModuleList([
            DecoderResidualBlock(384, 64, upsample_size=(25, 1))  # (384, 12, 1) -> (64, 25, 1)
            for _ in range(3)
        ])
        
        # Stage 2: Three parallel branches, 192 -> 32
        self.stage2 = nn.ModuleList([
            DecoderResidualBlock(192, 32, upsample_size=(50, 1))  # (192, 25, 1) -> (32, 50, 1)
            for _ in range(3)
        ])
        
        # Stage 3: Three parallel branches, 96 -> 16
        self.stage3 = nn.ModuleList([
            DecoderResidualBlock(96, 16, upsample_size=(100, 3))  # (96, 50, 1) -> (16, 100, 3)
            for _ in range(3)
        ])
        
        # Stage 4: Three parallel branches, 48 -> 1
        self.stage4 = nn.ModuleList([
            DecoderResidualBlock(48, 1, upsample_size=(201, 6))  # (48, 100, 3) -> (1, 201, 6)
            for _ in range(3)
        ])
        
    def forward(self, z):
        # Expand bottleneck
        out = self.fc(z)                            # (batch, 4608)
        out = out.view(-1, 384, 12, 1)              # (batch, 384, 12, 1)
        
        # Stage 1
        outs1 = [branch(out) for branch in self.stage1]  # 3x (batch, 64, 25, 1)
        out1 = torch.cat(outs1, dim=1)                   # (batch, 192, 25, 1)
        
        # Stage 2
        outs2 = [branch(out1) for branch in self.stage2]  # 3x (batch, 32, 50, 1)
        out2 = torch.cat(outs2, dim=1)                    # (batch, 96, 50, 1)
        
        # Stage 3
        outs3 = [branch(out2) for branch in self.stage3]  # 3x (batch, 16, 100, 3)
        out3 = torch.cat(outs3, dim=1)                    # (batch, 48, 100, 3)
        
        # Stage 4 and averaging
        outs4 = [branch(out3) for branch in self.stage4]  # 3x (batch, 1, 201, 6)
        out4 = torch.mean(torch.stack(outs4), dim=0)      # (batch, 1, 201, 6)
        return out4

dataset= CustomDataLoader(rootDirs=['E:\\Work\\University\\PR\\datas\\voice_gender_detection\\females','E:\\Work\\University\\PR\\datas\\voice_gender_detection\\males'],sr=16000,duration=0.07,HaveSaveOutput=False,isNormalized=True)
trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
valloader = DataLoader(valset, batch_size=32, shuffle=False)
learning_rate = 1e-3
num_epochs = 150
enc=Encoder()
nowTime=datetime.now().strftime("%m-%d-%Y--%H-%M_")
writer = SummaryWriter(log_dir='./runs/'+nowTime+'radeee_Autoencoder_256')
model = AutoEncoder(enc, Decoder()).cuda()
criterion = nn.MSELoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss=[]
for epoch in range(num_epochs):
    loss.append(train(model, trainloader, optimizer, criterion, epoch,writer))
    test(model, epoch, valloader, criterion,writer, "Validation")
torch.save(enc.state_dict(), nowTime+',pth')