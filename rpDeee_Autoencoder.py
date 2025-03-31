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
        self.elu2 = nn.ELU()
        
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.elu3 = nn.ELU()
        
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.elu4 = nn.ELU()
        
        self.conv5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(out_channels)
        self.elu5 = nn.ELU()
        
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = None
            
    def forward(self, x):
        residual = x if self.residual is None else self.residual(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.elu2(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.elu3(out)
        
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.elu4(out)
        
        out = self.conv5(out)  
        out = self.bn5(out)    
        out = out + residual
        out = self.elu5(out)   
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
        self.elu2 = nn.ELU()
        
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.elu3 = nn.ELU()
        
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.elu4 = nn.ELU()
        
        self.conv5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(out_channels)
        self.elu5 = nn.ELU()
        
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x_upsampled = self.upsample(x)
        residual = self.residual(x_upsampled)
        
        out = self.conv1(x_upsampled)
        out = self.bn1(out)
        out = self.elu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.elu2(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.elu3(out)
        
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.elu4(out)
        
        out = self.conv5(out)
        out = self.bn5(out)
        out = out + residual
        out = self.elu5(out)
        return out
    
class EncoderPart(nn.Module):
    def __init__(self):
        super(EncoderPart, self).__init__()
        self.stage1 = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(1, 16),           # (1, 67, 6) -> (16, 67, 6)
                nn.MaxPool2d(2)                 # (16, 67, 6) -> (16, 33, 3)
            ) for _ in range(3)
        ])
        
        self.stage2 = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(48, 32),          # (48, 33, 3) -> (32, 33, 3)
                nn.MaxPool2d(2)                 # (32, 33, 3) -> (32, 16, 1)
            ) for _ in range(3)
        ])
        
        self.stage3 = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(96, 64),          # (96, 16, 1) -> (64, 16, 1)
                nn.MaxPool2d(kernel_size=(2, 1))  # (64, 16, 1) -> (64, 8, 1)
            ) for _ in range(3)
        ])
        
        self.stage4 = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(192, 128),        # (192, 8, 1) -> (128, 8, 1)
                nn.MaxPool2d(kernel_size=(2, 1))  # (128, 8, 1) -> (128, 4, 1)
            ) for _ in range(3)
        ])
        
    def forward(self, x):
        outs1 = [branch(x) for branch in self.stage1]  # 3x (batch, 16, 33, 3)
        out1 = torch.cat(outs1, dim=1)                 # (batch, 48, 33, 3)
        
        outs2 = [branch(out1) for branch in self.stage2]  # 3x (batch, 32, 16, 1)
        out2 = torch.cat(outs2, dim=1)                    # (batch, 96, 16, 1)
        
        outs3 = [branch(out2) for branch in self.stage3]  # 3x (batch, 64, 8, 1)
        out3 = torch.cat(outs3, dim=1)                    # (batch, 192, 8, 1)
        
        outs4 = [branch(out3) for branch in self.stage4]  # 3x (batch, 128, 4, 1)
        out4 = torch.cat(outs4, dim=1)                    # (batch, 384, 4, 1)
        
        return out4

class DecoderPart(nn.Module):
    def __init__(self):
        super(DecoderPart, self).__init__()
        
        self.stage1 = nn.ModuleList([
            DecoderResidualBlock(384, 64, upsample_size=(8, 1))  # Changed: (25,1) to (8,1)
            for _ in range(3)
        ])
        
        self.stage2 = nn.ModuleList([
            DecoderResidualBlock(192, 32, upsample_size=(16, 1))  # Changed: (50,1) to (16,1)
            for _ in range(3)
        ])
        
        self.stage3 = nn.ModuleList([
            DecoderResidualBlock(96, 16, upsample_size=(33, 3))  # Changed: (100,3) to (33,3)
            for _ in range(3)
        ])
        
        self.stage4 = nn.ModuleList([
            DecoderResidualBlock(48, 1, upsample_size=(67, 6))  # Changed: (201,6) to (67,6)
            for _ in range(3)
        ])
        
    def forward(self, z):
        outs1 = [branch(z) for branch in self.stage1]  # 3x (batch, 64, 8, 1)
        out1 = torch.cat(outs1, dim=1)                   # (batch, 192, 8, 1)
        
        outs2 = [branch(out1) for branch in self.stage2]  # 3x (batch, 32, 16, 1)
        out2 = torch.cat(outs2, dim=1)                    # (batch, 96, 16, 1)
        
        outs3 = [branch(out2) for branch in self.stage3]  # 3x (batch, 16, 33, 3)
        out3 = torch.cat(outs3, dim=1)                    # (batch, 48, 33, 3)
        
        outs4 = [branch(out3) for branch in self.stage4]  # 3x (batch, 1, 67, 6)
        out4 = torch.mean(torch.stack(outs4), dim=0)      # (batch, 1, 67, 6)
        return out4
class Encoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Three encoder parts
        self.part1=EncoderPart()
        self.part2=EncoderPart()
        self.part3=EncoderPart()
        # Linear layer for bottleneck, 384 * 12 * 1 = 4608
        self.fc = nn.Linear(384 * 12 * 1, 128)  # 4608 -> 128 tried with 512
    def forward(self,x): # [batch , 201,6]
        x1=x[:,:,:67,:]
        x2=x[:,:,67:134,:]
        x3=x[:,:,134:,:]
        x1=self.part1(x1)
        x2=self.part2(x2)
        x3=self.part3(x3)
        total=torch.cat((x1,x2,x3),dim=2)
        total=total.view(total.size(0),-1)
        return self.fc(total)
class Decoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Three decoder part
        self.part1=DecoderPart()
        self.part2=DecoderPart()
        self.part3=DecoderPart()
        # Linear layer to expand bottleneck
        self.fc = nn.Linear(128, 384 * 12 * 1)  # 128 -> 4608
    def forward(self,z):
        out = self.fc(z)                            # (batch, 4608)
        out = out.view(-1, 384, 12, 1)              # (batch, 384, 12, 1)
        x1=out[:,:,:4,:]
        x2=out[:,:,4:8,:]
        x3=out[:,:,8:,:]
        x1=self.part1(x1)
        x2=self.part2(x2)
        x3=self.part3(x3)
        out=torch.cat((x1,x2,x3),dim=2)
        return out
dataset= CustomDataLoader(rootDirs=['E:\\Work\\University\\PR\\datas\\voice_gender_detection\\females','E:\\Work\\University\\PR\\datas\\voice_gender_detection\\males'],sr=16000,duration=0.07,HaveSaveOutput=False,isNormalized=True)
trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
valloader = DataLoader(valset, batch_size=32, shuffle=False)
learning_rate = 1e-3
num_epochs = 150
enc=Encoder()
nowTime=datetime.now().strftime("%m-%d-%Y--%H-%M_")
writer = SummaryWriter(log_dir='./runs/'+nowTime+'rpDeee_Autoencoder')
model = AutoEncoder(enc, Decoder()).cuda()
criterion = nn.MSELoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss=[]
for epoch in range(num_epochs):
    loss.append(train(model, trainloader, optimizer, criterion, epoch,writer))
    test(model, epoch, valloader, criterion,writer, "Validation")
torch.save(enc.state_dict(), nowTime+'.pth')