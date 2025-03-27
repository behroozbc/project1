import torch
import torch.nn as nn

from AutoEncoder import AutoEncoder
from DatasetLoader import CustomDataLoader
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from AutoEncoder import AutoEncoder
from BaseFunc import plotResults, plotit
from DatasetLoader import CustomDataLoader
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


def train(model, trainloader, optimizer, criterion, epoch,writer:SummaryWriter):
    model.train()
    train_loss = []
    for batch_idx, [_,spec,_,_] in enumerate(trainloader):
        spec = spec.cuda()
        optimizer.zero_grad()
        output = model(spec)
        loss = criterion(output, spec)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        if batch_idx % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(spec), len(trainloader.dataset),
                       100. * batch_idx / len(trainloader), loss.item()))
    mean=np.mean(train_loss)
    writer.add_scalar("Train Loss",mean,epoch)
    return np.mean(train_loss)
def test(model, device, test_loader, criterion,writer:SummaryWriter, set="Test"):
    model.eval()
    test_loss = []
    latent_vectors = []
    labels = []
    
    
    with torch.no_grad():
        for batch_idx, [_,spec,_,label] in enumerate(test_loader):
            spec= spec.cuda()
            output = model(spec)
            if batch_idx == 0:
                plotit(spec[0],epoch=epoch,kind="Real",writer=writer)
                plotit(output[0],epoch=epoch,kind="Reconstructed",writer=writer)
            test_loss.append( criterion(output, spec).item()) 
            latent_vector = model.encoder(spec)
            latent_vectors.append(latent_vector)
            labels+=label
    latent_vectors = torch.cat(latent_vectors, dim=0)
    
    plotResults(epoch, latent_vectors, labels,writer)
    writer.add_scalar("Test Loss",np.mean(test_loss),epoch)
# Encoder class
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # First part: Three parallel CLs with 8 filters
        self.conv1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding=1),  # Input: (1, 201, 6), Output: (8, 201, 4)
                nn.MaxPool2d(2),                            # Output: (8, 100, 3)
                nn.BatchNorm2d(8),
                nn.ELU()
            ) for _ in range(3)
        ])
        
        # Second part: Three parallel CLs with 16 filters
        self.conv2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(24, 16, kernel_size=3, padding=1),  # Input: (24, 100, 3), Output: (16, 100, 2)
                nn.MaxPool2d(2),                             # Output: (16, 50, 1)
                nn.BatchNorm2d(16),
                nn.ELU()
            ) for _ in range(3)
        ])
        
        # Third part: Linear layer for bottleneck
        self.fc = nn.Linear(48 * 50 * 1, 128)  # Input: 48*50*1=2400, Output: 128
        
    def forward(self, x):
        # First parallel convolutions
        outs1 = [branch(x) for branch in self.conv1]      # List of 3 tensors, each (batch_size, 8, 100, 2)
        out1 = torch.cat(outs1, dim=1)                    # (batch_size, 24, 100, 2)
        
        # Second parallel convolutions
        outs2 = [branch(out1) for branch in self.conv2]   # List of 3 tensors, each (batch_size, 16, 50, 1)
        out2 = torch.cat(outs2, dim=1)                    # (batch_size, 48, 50, 1)
        
        # Flatten and apply linear layer
        out2_flat = out2.view(out2.size(0), -1)           # (batch_size, 2400)
        z = self.fc(out2_flat)                            # (batch_size, 128)
        return z

# Decoder class
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # First part: Linear layer to expand bottleneck
        self.fc = nn.Linear(128, 48 * 50 * 1)  # Input: 128, Output: 2400
        
        # Second part: Three parallel deconvolutional layers
        self.deconv1 = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(size=(100, 2)),            # Input: (48, 50, 1), Output: (48, 100, 2)
                nn.Conv2d(48, 8, kernel_size=3, padding=1),  # Output: (8, 100, 2)
                nn.BatchNorm2d(8),
                nn.ELU()
            ) for _ in range(3)
        ])
        
        # Third part: Three parallel deconvolutional layers to reconstruct input
        self.deconv2 = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(size=(201, 4)),            # Input: (24, 100, 3), Output: (24, 201, 4)
                nn.Conv2d(24, 1, kernel_size=3, padding=1),  # Output: (1, 201, 4)
                nn.BatchNorm2d(1),
                nn.ELU()
            ) for _ in range(3)
        ])
        
    def forward(self, z):
        # Expand bottleneck
        out = self.fc(z)                            # (batch_size, 2400)
        out = out.view(-1, 48, 50, 1)               # (batch_size, 48, 50, 1)
        
        # First parallel deconvolutions
        outs1 = [branch(out) for branch in self.deconv1]  # List of 3 tensors, each (batch_size, 8, 100, 2)
        out1 = torch.cat(outs1, dim=1)                    
        
        # Second parallel deconvolutions and averaging
        outs2 = [branch(out1) for branch in self.deconv2] # List of 3 tensors, each (batch_size, 1, 201, 4)
        out2 = torch.mean(torch.stack(outs2), dim=0)      # (batch_size, 1, 201, 4)
        return out2

        
dataset= CustomDataLoader(rootDirs=['E:\\Work\\University\\PR\\datas\\voice_gender_detection\\females','E:\\Work\\University\\PR\\datas\\voice_gender_detection\\males'],sr=16000,duration=0.04,HaveSaveOutput=False,isNormalized=True)
trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
valloader = DataLoader(valset, batch_size=32, shuffle=False)
learning_rate = 1e-3
num_epochs = 100
enc=Encoder()
writer = SummaryWriter(log_dir='./runs/'+datetime.now().strftime("%m-%d-%Y--%H-%M_")+'part-2-spec-autoencoder',comment="64 bottleneck")
model = AutoEncoder(enc, Decoder()).cuda()
criterion = nn.MSELoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss=[]
for epoch in range(num_epochs):
    loss.append(train(model, trainloader, optimizer, criterion, epoch,writer))
    test(model, 'cuda', valloader, criterion,writer, "Validation")
torch.save(enc.state_dict(), 'split-ckeck-sep_autoencoder.pth')
