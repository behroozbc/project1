import torch
import torch.nn as nn

from datetime import datetime
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
        # First stage: Three parallel branches with 16 filters, five Conv2d layers each
        self.conv1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),  # (1, 201, 4) -> (16, 201, 4)
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.Conv2d(16, 16, kernel_size=3, padding=1), # (16, 201, 4) -> (16, 201, 4)
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.Conv2d(16, 16, kernel_size=3, padding=1), # (16, 201, 4) -> (16, 201, 4)
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.Conv2d(16, 16, kernel_size=3, padding=1), # (16, 201, 4) -> (16, 201, 4)
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.Conv2d(16, 16, kernel_size=3, padding=1), # (16, 201, 4) -> (16, 201, 4)
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.MaxPool2d((2, 1))                         # (16, 201, 4) -> (16, 100, 4)
            ) for _ in range(3)
        ])
        
        # Second stage: Three parallel branches with 32 filters, five Conv2d layers each
        self.conv2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(48, 32, kernel_size=3, padding=1), # (48, 100, 4) -> (32, 100, 4)
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1), # (32, 100, 4) -> (32, 100, 4)
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1), # (32, 100, 4) -> (32, 100, 4)
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1), # (32, 100, 4) -> (32, 100, 4)
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1), # (32, 100, 4) -> (32, 100, 4)
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.MaxPool2d(2)                              # (32, 100, 4) -> (32, 50, 2)
            ) for _ in range(3)
        ])
        
        # Third stage: Three parallel branches with 64 filters, five Conv2d layers each
        self.conv3 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(96, 64, kernel_size=3, padding=1), # (96, 50, 2) -> (64, 50, 2)
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1), # (64, 50, 2) -> (64, 50, 2)
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1), # (64, 50, 2) -> (64, 50, 2)
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1), # (64, 50, 2) -> (64, 50, 2)
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1), # (64, 50, 2) -> (64, 50, 2)
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.MaxPool2d(2)                              # (64, 50, 2) -> (64, 25, 1)
            ) for _ in range(3)
        ])
        
        # Bottleneck: Linear layer
        self.fc = nn.Linear(192 * 25 * 1, 512)  # 4800 -> 512
        
    def forward(self, x):
        # First stage
        outs1 = [branch(x) for branch in self.conv1]  # 3 x (batch_size, 16, 100, 4)
        out1 = torch.cat(outs1, dim=1)                # (batch_size, 48, 100, 4)
        
        # Second stage
        outs2 = [branch(out1) for branch in self.conv2]  # 3 x (batch_size, 32, 50, 2)
        out2 = torch.cat(outs2, dim=1)                   # (batch_size, 96, 50, 2)
        
        # Third stage
        outs3 = [branch(out2) for branch in self.conv3]  # 3 x (batch_size, 64, 25, 1)
        out3 = torch.cat(outs3, dim=1)                   # (batch_size, 192, 25, 1)
        
        # Flatten and apply bottleneck
        out3_flat = out3.view(out3.size(0), -1)  # (batch_size, 4800)
        z = self.fc(out3_flat)                   # (batch_size, 512)
        return z

# Decoder class
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Expand bottleneck
        self.fc = nn.Linear(512, 192 * 25 * 1)  # 512 -> 4800
        
        # First stage: Three parallel branches with 64 filters, five Conv2d layers each
        self.deconv1 = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(size=(50, 2)),            # (192, 25, 1) -> (192, 50, 2)
                nn.Conv2d(192, 64, kernel_size=3, padding=1), # (192, 50, 2) -> (64, 50, 2)
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1), # (64, 50, 2) -> (64, 50, 2)
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1), # (64, 50, 2) -> (64, 50, 2)
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1), # (64, 50, 2) -> (64, 50, 2)
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1), # (64, 50, 2) -> (64, 50, 2)
                nn.BatchNorm2d(64),
                nn.ELU()
            ) for _ in range(3)
        ])
        
        # Second stage: Three parallel branches with 32 filters, five Conv2d layers each
        self.deconv2 = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(size=(100, 4)),           # (192, 50, 2) -> (192, 100, 4)
                nn.Conv2d(192, 32, kernel_size=3, padding=1), # (192, 100, 4) -> (32, 100, 4)
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1), # (32, 100, 4) -> (32, 100, 4)
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1), # (32, 100, 4) -> (32, 100, 4)
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1), # (32, 100, 4) -> (32, 100, 4)
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1), # (32, 100, 4) -> (32, 100, 4)
                nn.BatchNorm2d(32),
                nn.ELU()
            ) for _ in range(3)
        ])
        
        # Third stage: Three parallel branches to reconstruct input
        self.deconv3 = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(size=(201, 4)),           # (96, 100, 4) -> (96, 201, 4)
                nn.Conv2d(96, 16, kernel_size=3, padding=1), # (96, 201, 4) -> (16, 201, 4)
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.Conv2d(16, 16, kernel_size=3, padding=1), # (16, 201, 4) -> (16, 201, 4)
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.Conv2d(16, 16, kernel_size=3, padding=1), # (16, 201, 4) -> (16, 201, 4)
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.Conv2d(16, 16, kernel_size=3, padding=1), # (16, 201, 4) -> (16, 201, 4)
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.Conv2d(16, 1, kernel_size=3, padding=1),  # (16, 201, 4) -> (1, 201, 4)
                nn.BatchNorm2d(1),
                nn.ELU()
            ) for _ in range(3)
        ])
        
    def forward(self, z):
        # Expand bottleneck
        out = self.fc(z)                      # (batch_size, 4800)
        out = out.view(-1, 192, 25, 1)        # (batch_size, 192, 25, 1)
        
        # First stage
        outs1 = [branch(out) for branch in self.deconv1]  # 3 x (batch_size, 64, 50, 2)
        out1 = torch.cat(outs1, dim=1)                    # (batch_size, 192, 50, 2)
        
        # Second stage
        outs2 = [branch(out1) for branch in self.deconv2] # 3 x (batch_size, 32, 100, 4)
        out2 = torch.cat(outs2, dim=1)                    # (batch_size, 96, 100, 4)
        
        # Third stage and averaging
        outs3 = [branch(out2) for branch in self.deconv3] # 3 x (batch_size, 1, 201, 4)
        out3 = torch.mean(torch.stack(outs3), dim=0)      # (batch_size, 1, 201, 4)
        return out3

dataset= CustomDataLoader(rootDirs=['E:\\Work\\University\\PR\\datas\\voice_gender_detection\\females','E:\\Work\\University\\PR\\datas\\voice_gender_detection\\males'],sr=16000,duration=0.04,HaveSaveOutput=False,isNormalized=True)
trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
valloader = DataLoader(valset, batch_size=32, shuffle=False)
learning_rate = 1e-3
num_epochs = 100
enc=Encoder()
writer = SummaryWriter(log_dir='./runs/'+datetime.now().strftime("%m-%d-%Y--%H-%M_")+'sep_autoencoder-deeeeeeeeeeeeeper')
model = AutoEncoder(enc, Decoder()).cuda()
criterion = nn.MSELoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss=[]
for epoch in range(num_epochs):
    loss.append(train(model, trainloader, optimizer, criterion, epoch,writer))
    test(model, 'cuda', valloader, criterion,writer, "Validation")
torch.save(enc.state_dict(), 'sep_autoencoder-deeeeeeeeeeeeeper.pth')