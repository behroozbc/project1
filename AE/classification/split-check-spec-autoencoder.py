import torch
import torch.nn as nn
import numpy as np
import torchaudio
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from AutoEncoder import AutoEncoder
from BaseFunc import plotResults, plotit
from DatasetLoader import CustomDataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn import metrics 
from sklearn.model_selection import train_test_split
from multiprocessing import Process,freeze_support



def train(model, trainloader, optimizer, criterion, epoch):
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
    return np.mean(train_loss)
def test(model, device, test_loader, criterion, set="Test"):
    model.eval()
    test_loss = 0
    correct = 0
    latent_vectors = []
    labels = []
    
    
    with torch.no_grad():
        for batch_idx, [_,spec,_,label] in enumerate(test_loader):
            spec= spec.cuda()
            output = model(spec)
            if batch_idx == 0:
                plotit('split-spec/data/',spec[0],epoch=epoch,kind="Real")
                plotit('split-spec/data/',output[0],epoch=epoch,kind="Reconstructed")
            test_loss += criterion(output, spec).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            latent_vector = model.encoder(spec)
            latent_vectors.append(latent_vector)
            labels+=label
    latent_vectors = torch.cat(latent_vectors, dim=0)
    plotResults('split-spec',epoch, latent_vectors, labels)
    
    test_loss /= len(test_loader.dataset)
    accuarcy = 100. * (1 - test_loss)
    # accuarcy=100. * correct / len(test_loader.dataset)
    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        set, test_loss, correct, len(test_loader.dataset),
        accuarcy))
    return accuarcy
# Encoder module for one part (top or bottom)
class PartEncoder(nn.Module):
    def __init__(self, C1, C2, C3):
        super().__init__()
        # First conv layer: 1 channel -> C1
        self.conv1 = nn.Conv2d(1, C1, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.elu1 = nn.ELU()
        # Second conv layer: C1 -> C2
        self.conv2 = nn.Conv2d(C1, C2, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.elu2 = nn.ELU()
        # Third conv layer: C2 -> C3
        self.conv3 = nn.Conv2d(C2, C3, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d((2, 1))  # Adjusted to (2,1) due to width=1
        self.elu3 = nn.ELU()
        self.batchnorm1 = nn.BatchNorm2d(C1)
        self.batchnorm2 = nn.BatchNorm2d(C2)
        self.batchnorm3 = nn.BatchNorm2d(C3)
    def forward(self, x):
        x = self.elu1(self.batchnorm1( self.pool1(self.conv1(x))))  # (batch, C1, 25, 2)
        x = self.elu2(self.batchnorm2(self.pool2(self.conv2(x))))  # (batch, C2, 12, 1)
        x = self.elu3(self.batchnorm3(self.pool3(self.conv3(x))))  # (batch, C3, 6, 1)
        return x

# Decoder module for one part (top or bottom)
class PartDecoder(nn.Module):
    def __init__(self, C1, C2, C3):
        super().__init__()
        # Upsample and conv: C3 -> C2
        self.upsample1 = nn.Upsample(size=(12, 1), mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(C3, C2, kernel_size=3, padding=1)
        self.elu1 = nn.ELU()
        # Upsample and conv: C2 -> C1
        self.upsample2 = nn.Upsample(size=(25, 2), mode='bilinear', align_corners=False)
        self.conv2 = nn.Conv2d(C2, C1, kernel_size=3, padding=1)
        self.elu2 = nn.ELU()
        # Upsample and conv: C1 -> 1
        self.upsample3 = nn.Upsample(size=(50, 4), mode='bilinear', align_corners=False)
        self.conv3 = nn.Conv2d(C1, 1, kernel_size=3, padding=1)  # No activation on output
        self.batchnorm1 = nn.BatchNorm2d(1)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.batchnorm3 = nn.BatchNorm2d(32)
    def forward(self, x):
        x = self.elu1(self.batchnorm3(self.conv1(self.upsample1(x))))  # (batch, C2, 12, 1)
        x = self.elu2(self.batchnorm2(self.conv2(self.upsample2(x))))  # (batch, C1, 25, 2)
        x = self.batchnorm1(self.conv3(self.upsample3(x)))             # (batch, 1, 50, 4)
        return F.relu(x)
class encoder(nn.Module):
    def __init__(self, C1, C2, C3, Z):
        super().__init__()
        self.top_encoder = PartEncoder(C1, C2, C3)
        self.bottom_encoder = PartEncoder(C1, C2, C3)
        # Flattened size after encoding one part
        self.flatten_size = C3 * 6
        # Linear layers for latent space
        self.linear_enc = nn.Linear(2 * self.flatten_size, Z)  # 2 parts combined
        
    def forward(self, x):
        batch_size = x.size(0)
        # Split spectrogram into top and bottom
        top = x[:, :, :50, :]    # (batch, 1, 50, 4)
        bottom = x[:, :, 50:, :] # (batch, 1, 50, 4)
        # Encode each part separately
        top_enc = self.top_encoder(top)       # (batch, C3, 6, 1)
        bottom_enc = self.bottom_encoder(bottom)  # (batch, C3, 6, 1)
        # Flatten outputs
        C3 = self.top_encoder.conv3.out_channels
        top_flat = top_enc.view(batch_size, -1)    # (batch, C3*6)
        bottom_flat = bottom_enc.view(batch_size, -1)  # (batch, C3*6)
        # Concatenate and map to latent space
        combined = torch.cat((top_flat, bottom_flat), dim=1)  # (batch, 2*C3*6)
        z = self.linear_enc(combined)  
        return z
class decoder(nn.Module):
    def __init__(self, C1, C2, C3, Z):
        super().__init__()
        self.flatten_size = C3 * 6
        self.linear_dec = nn.Linear(Z, 2 * self.flatten_size)
        # Separate decoders for top and bottom parts
        self.top_decoder = PartDecoder(C1, C2, C3)
        self.bottom_decoder = PartDecoder(C1, C2, C3)
        self.C3=C3
    
    def forward(self, z):
        batch_size = z.size(0)
        # Decode from latent space
        decoded = self.linear_dec(z)                         # (batch, 2*C3*6)
        # Split decoded vector for each part
        top_dec = decoded[:, :self.flatten_size].view(batch_size,self. C3, 6, 1)    # (batch, C3, 6, 1)
        bottom_dec = decoded[:, self.flatten_size:].view(batch_size,self. C3, 6, 1) # (batch, C3, 6, 1)
        # Reconstruct each part
        top_recon = self.top_decoder(top_dec)       # (batch, 1, 50, 4)
        bottom_recon = self.bottom_decoder(bottom_dec)  # (batch, 1, 50, 4)
        # Concatenate along height to match input size
        recon = torch.cat((top_recon, bottom_recon), dim=2)  # (batch, 1, 100, 4)
        return F.relu(recon)
        
dataset= CustomDataLoader(rootDirs=['E:\\Work\\University\\PR\\datas\\voice_gender_detection\\females','E:\\Work\\University\\PR\\datas\\voice_gender_detection\\males'],duration=0.05,HaveSaveOutput=False,isNormalized=True)
trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
valloader = DataLoader(valset, batch_size=32, shuffle=False)
learning_rate = 1e-3
num_epochs = 100
C1, C2, C3, Z = 16, 32, 64, 128
enc=encoder(C1, C2, C3, Z)
freeze_support()
model = AutoEncoder(enc, decoder(C1, C2, C3, Z)).cuda()
criterion = nn.MSELoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss=[]
for epoch in range(num_epochs):
    loss.append(train(model, trainloader, optimizer, criterion, epoch))
    test(model, 'cuda', valloader, criterion, "Validation")
torch.save(enc.state_dict(), 'split-sep_autoencoder.pth')
