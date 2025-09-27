import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
from loss import VaeLoss
sys.path.append('e:\\Work\\University\\PR\\project1')
from BaseFunc import plot_latent_spaceVAE, plotit
from DatasetLoader import CustomDataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Parallel Convolution Block for Encoder (Deeper)
class ParallelConvBlock(nn.Module):
    def __init__(self, input_channels, C_hidden, C_out, num_branches=4):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_channels, C_hidden, kernel_size=3, stride=1, padding=1),
                nn.ELU(),
                nn.Conv2d(C_hidden, C_hidden, kernel_size=3, stride=1, padding=1),
                nn.ELU(),
                nn.Conv2d(C_hidden, C_hidden, kernel_size=3, stride=1, padding=1),  # Added layer
                nn.ELU(),
                nn.Conv2d(C_hidden, C_out, kernel_size=3, stride=2, padding=0)
            ) for _ in range(num_branches)
        ])
    
    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]
        return torch.cat(branch_outputs, dim=1)

# Parallel Transposed Convolution Block for Decoder (Deeper, intermediate blocks)
class ParallelConvTransposeBlock(nn.Module):
    def __init__(self, input_channels, C_hidden, C_out, num_branches=4, output_padding=(0,0)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(input_channels, C_hidden, kernel_size=3, stride=2, padding=0, output_padding=output_padding),
                nn.ELU(),
                nn.Conv2d(C_hidden, C_hidden, kernel_size=3, stride=1, padding=1),
                nn.ELU(),
                nn.Conv2d(C_hidden, C_hidden, kernel_size=3, stride=1, padding=1),  # Added layer
                nn.ELU(),
                nn.Conv2d(C_hidden, C_out, kernel_size=3, stride=1, padding=1)
            ) for _ in range(num_branches)
        ])
    
    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]
        return torch.cat(branch_outputs, dim=1)

# Final Parallel Transposed Convolution Block (Deeper, outputs 1 channel)
class FinalParallelConvTransposeBlock(nn.Module):
    def __init__(self, input_channels, C_hidden, num_branches=4, output_padding=(0,0)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(input_channels, C_hidden, kernel_size=3, stride=2, padding=0, output_padding=output_padding),
                nn.ELU(),
                nn.Conv2d(C_hidden, C_hidden, kernel_size=3, stride=1, padding=1),
                nn.ELU(),
                nn.Conv2d(C_hidden, C_hidden, kernel_size=3, stride=1, padding=1),  # Added layer
                nn.ELU(),
                nn.Conv2d(C_hidden, 1, kernel_size=3, stride=1, padding=1)
            ) for _ in range(num_branches)
        ])
    
    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]
        return sum(branch_outputs) / len(branch_outputs)

# VAE Model (Deeper)
class VAE(nn.Module):
    def __init__(self, Z_dim=256):
        super().__init__()
        
        # Encoder with deeper blocks
        self.encoder = nn.Sequential(
            ParallelConvBlock(1, 32, 4),    # 513x51x1 -> 256x25x16 (4 branches * 4 channels)
            ParallelConvBlock(16, 32, 8),   # 256x25x16 -> 127x12x32 (4 * 8)
            ParallelConvBlock(32, 32, 16),  # 127x12x32 -> 63x5x64 (4 * 16)
            ParallelConvBlock(64, 32, 32)   # 63x5x64 -> 31x2x128 (4 * 32)
        )
        self.fc_mean = nn.Linear(31 * 2 * 128, Z_dim)
        self.fc_logvar = nn.Linear(31 * 2 * 128, Z_dim)
        
        # Decoder with deeper blocks
        self.fc_decode = nn.Linear(Z_dim, 31 * 2 * 128)
        self.decoder_blocks = nn.ModuleList([
            ParallelConvTransposeBlock(128, 32, 16, output_padding=(0,0)),  # 31x2x128 -> 63x5x64
            ParallelConvTransposeBlock(64, 32, 8, output_padding=(0,1)),    # 63x5x64 -> 127x12x32
            ParallelConvTransposeBlock(32, 32, 4, output_padding=(1,0)),    # 127x12x32 -> 256x25x16
            FinalParallelConvTransposeBlock(16, 32, output_padding=(0,0))   # 256x25x16 -> 513x51x1
        ])
    
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Flatten to (batch_size, 31*2*128)
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(-1, 128, 31, 2)  # Reshape to (batch_size, 128, 31, 2)
        for block in self.decoder_blocks:
            h = block(h)
        return h
    
    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon_x = self.decode(z)
        return recon_x, mean, logvar
vae=VAE().cuda()
lr=1e-3
optimz=optim.Adam(vae.parameters(),lr)
epochNumber=100
nowTime=datetime.now().strftime("%m-%d-%Y--%H-%M_")
writer = SummaryWriter(log_dir='./runs/'+nowTime+'4')
dataset= CustomDataLoader(rootDirs=['E:\\Work\\University\\PR\\datas\\voice_gender_detection\\females','E:\\Work\\University\\PR\\datas\\voice_gender_detection\\males'],sr=16000,duration=0.5,HaveSaveOutput=False,isNormalized=True)
trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
valloader = DataLoader(valset, batch_size=32, shuffle=False)
allloader=DataLoader(dataset,batch_size=32)
for epoch in range(epochNumber):
    vae.train()
    loss_training=[]
    for batch,[spec,_] in  enumerate(trainloader):
        spec = spec.cuda()
        optimz.zero_grad()
        recon_x, mean, logvar = vae(spec)
        loss = VaeLoss(recon_x, spec, mean, logvar)
        loss.backward()
        loss_training.append(loss.item())
        optimz.step()
    print('Train Epoch: {} \tLoss: {:.6f}'.format(
                epoch, np.mean(loss_training)))
    testLoss=[]
    vae.eval()
    with torch.no_grad():
        for batch, [spec,_] in enumerate(valloader):
            spec=spec.cuda()
            recon_x, mean, logvar = vae(spec)
            loss = VaeLoss(recon_x, spec, mean, logvar)
            testLoss.append(loss.item())
            if batch==0:
                plotit(spec[0],epoch,"Real",writer)
                plotit(recon_x[0],epoch,"Perdict",writer)
    writer.add_scalar("Test Loss",np.mean(testLoss),epoch)
    writer.add_scalar("Training Loss",np.mean(loss_training),epoch)
    plot_latent_spaceVAE(vae,valloader,epoch,writer)
torch.save(vae.state_dict(), nowTime+'.pth')
writer.close()