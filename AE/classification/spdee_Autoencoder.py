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

class EncoderPart(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding=1),  # (1, 67, 6) -> (8, 67, 6)
                nn.BatchNorm2d(8),
                nn.ELU(),
                nn.Conv2d(8, 8, kernel_size=3, padding=1),  # (8, 67, 6) -> (8, 67, 6)
                nn.BatchNorm2d(8),
                nn.ELU(),
                nn.MaxPool2d(2)                             # (8, 67, 6) -> (8, 33, 3)
            ) for _ in range(3)
        ])
        
        # Stage 2: Three parallel CLs with 16 filters
        self.conv2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(24, 16, kernel_size=3, padding=1),  # (24, 33, 3) -> (16, 33, 3)
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.Conv2d(16, 16, kernel_size=3, padding=1),  # (16, 33, 3) -> (16, 33, 3)
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.MaxPool2d(2)                               # (16, 33, 3) -> (16, 16, 1)
            ) for _ in range(3)
        ])
        
        # Stage 3: Three parallel CLs with 32 filters
        self.conv3 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(48, 32, kernel_size=3, padding=1),  # (48, 16, 1) -> (32, 16, 1)
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),  # (32, 16, 1) -> (32, 16, 1)
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(2, 1))              # (32, 16, 1) -> (32, 8, 1)
            ) for _ in range(3)
        ])
        
        # Stage 4: Three parallel CLs with 64 filters
        self.conv4 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(96, 64, kernel_size=3, padding=1),  # (96, 8, 1) -> (64, 8, 1)
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),  # (64, 8, 1) -> (64, 8, 1)
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(2, 1))              # (64, 8, 1) -> (64, 4, 1)
            ) for _ in range(3)
        ])
    def forward(self,x):
        outs1 = [branch(x) for branch in self.conv1]      # 3x (batch, 8, 33, 3)
        out1 = torch.cat(outs1, dim=1)                    # (batch, 24, 33, 3)
        
        # Stage 2
        outs2 = [branch(out1) for branch in self.conv2]   # 3x (batch, 16, 16, 1)
        out2 = torch.cat(outs2, dim=1)                    # (batch, 48, 16, 1)
        
        # Stage 3
        outs3 = [branch(out2) for branch in self.conv3]   # 3x (batch, 32, 8, 1)
        out3 = torch.cat(outs3, dim=1)                    # (batch, 96, 8, 1)
        
        # Stage 4
        outs4 = [branch(out3) for branch in self.conv4]   # 3x (batch, 64, 4, 1)
        out4 = torch.cat(outs4, dim=1)
        return out4
class DecoderPart(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deconv1 = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(size=(8, 1)),            # (192, 4, 1) -> (192, 8, 1)
                nn.Conv2d(192, 32, kernel_size=3, padding=1),  # (192, 8, 1) -> (32, 8, 1)
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),  # (32, 8, 1) -> (32, 8, 1)
                nn.BatchNorm2d(32),
                nn.ELU()
            ) for _ in range(3)
        ])
        
        # Stage 2: Three parallel DCLs
        self.deconv2 = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(size=(16, 1)),           # (96, 8, 1) -> (96, 16, 1)
                nn.Conv2d(96, 16, kernel_size=3, padding=1),  # (96, 16, 1) -> (16, 16, 1)
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.Conv2d(16, 16, kernel_size=3, padding=1),  # (16, 16, 1) -> (16, 16, 1)
                nn.BatchNorm2d(16),
                nn.ELU()
            ) for _ in range(3)
        ])
        
        # Stage 3: Three parallel DCLs
        self.deconv3 = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(size=(33, 3)),           # (48, 16, 1) -> (48, 33, 3)
                nn.Conv2d(48, 8, kernel_size=3, padding=1),  # (48, 33, 3) -> (8, 33, 3)
                nn.BatchNorm2d(8),
                nn.ELU(),
                nn.Conv2d(8, 8, kernel_size=3, padding=1),  # (8, 33, 3) -> (8, 33, 3)
                nn.BatchNorm2d(8),
                nn.ELU()
            ) for _ in range(3)
        ])
        
        # Stage 4: Three parallel DCLs to reconstruct input
        self.deconv4 = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(size=(67, 6)),           # (24, 33, 3) -> (24, 67, 6)
                nn.Conv2d(24, 1, kernel_size=3, padding=1),  # (24, 67, 6) -> (1, 67, 6)
                nn.BatchNorm2d(1),
                nn.ELU(),
                nn.Conv2d(1, 1, kernel_size=3, padding=1),  # (1, 67, 6) -> (1, 67, 6)
                nn.BatchNorm2d(1),
                nn.ELU()
            ) for _ in range(3)
        ])
        
    def forward(self, z):
        # Stage 1
        outs1 = [branch(z) for branch in self.deconv1]  # 3x (batch, 32, 8, 1)
        out1 = torch.cat(outs1, dim=1)                    # (batch, 96, 8, 1)
        
        # Stage 2
        outs2 = [branch(out1) for branch in self.deconv2] # 3x (batch, 16, 16, 1)
        out2 = torch.cat(outs2, dim=1)                    # (batch, 48, 16, 1)
        
        # Stage 3
        outs3 = [branch(out2) for branch in self.deconv3] # 3x (batch, 8, 33, 3)
        out3 = torch.cat(outs3, dim=1)                    # (batch, 24, 33, 3)
        
        # Stage 4 and averaging
        outs4 = [branch(out3) for branch in self.deconv4] # 3x (batch, 1, 67, 6)
        out4 = torch.mean(torch.stack(outs4), dim=0)      # (batch, 1, 67, 6)
        return out4
class Encoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Three encoder parts
        self.part1=EncoderPart()
        self.part2=EncoderPart()
        self.part3=EncoderPart()
        # Linear layer for bottleneck
        self.fc = nn.Linear(192 * 12 * 1, 256)  # (192 * 12 * 1 = 2304) -> 128
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
        self.fc = nn.Linear(256, 192 * 12 * 1)  # 128 -> 2304
    def forward(self,z):
        out = self.fc(z)                            # (batch, 2304)
        out = out.view(-1, 192, 12, 1)              # (batch, 192, 12, 1)
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
writer = SummaryWriter(log_dir='./runs/'+nowTime+'spdee_Autoencoder')
model = AutoEncoder(enc, Decoder()).cuda()
criterion = nn.MSELoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss=[]
for epoch in range(num_epochs):
    loss.append(train(model, trainloader, optimizer, criterion, epoch,writer))
    test(model, epoch, valloader, criterion,writer, "Validation")
torch.save(enc.state_dict(), nowTime+',pth')