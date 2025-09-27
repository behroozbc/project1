import torch
import torch.nn as nn
import numpy as np
import torchaudio
from datetime import datetime
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from AutoEncoder import AutoEncoder
from BaseFunc import plotResults, plotit
from DatasetLoader import CustomDataLoader
import torch.nn.functional as F
from sklearn import svm
from sklearn import metrics 
from sklearn.model_selection import train_test_split
from multiprocessing import Process,freeze_support
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
def test(model, device, test_loader, criterion, writer:SummaryWriter, set="Test"):
    model.eval()
    test_loss = []
    correct = 0
    latent_vectors = []
    labels = []
    model.encoder.eval()
    
    
    with torch.no_grad():
        for batch_idx, [_,spec,_,label] in enumerate(test_loader):
            spec= spec.cuda()
            output = model(spec)
            if batch_idx == 0:
                plotit(spec[0],epoch=epoch,kind="Real",writer=writer)
                plotit(output[0],epoch=epoch,kind="Reconstructed",writer=writer)
            test_loss.append(criterion(output, spec).item())  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            latent_vector = model.encoder(spec)
            latent_vectors.append(latent_vector)
            labels+=label
    latent_vectors = torch.cat(latent_vectors, dim=0)
    plotResults(epoch, latent_vectors, labels,writer=writer)
    
    writer.add_scalar("Test Loss",np.mean(test_loss),epoch)
    accuarcy = 100. * (1 - test_loss)
    # accuarcy=100. * correct / len(test_loader.dataset)
    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        set, test_loss, correct, len(test_loader.dataset),
        accuarcy))
    return accuarcy
class encoder(nn.Module):
    def __init__(self,latent_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.flatten = nn.Flatten()
        self.activ1= nn.ELU()
        self.activ2= nn.ELU()
        self.activ3= nn.ELU()
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.batchnorm3 = nn.BatchNorm2d(64)
        # After three pooling layers: 100 -> 50 -> 25 -> 13, 4 -> 2 -> 1 -> 1
        self.fc_encode = nn.Linear(64 * 13 * 1, latent_dim)
    def forward(self,x):
        # print(x.shape)
        x = self.activ1(self.batchnorm1(self.pool1(self.conv1(x))))

        x = self.activ2(self.batchnorm2(self.pool2(self.conv2(x))))
        x = self.activ3(self.batchnorm3(self.pool3(self.conv3(x))))
        # x = self.elu(self.conv3(x))
        # x = self.pool3(x)  # (batch, 64, 25, 3)
        
        x = self.flatten(x)
        z = self.fc_encode(x)
        return z
    
class decoder(nn.Module):
    def __init__(self,latent_dim=128):
        super().__init__()
        self.fc_decode = nn.Linear(latent_dim, 64 * 13 * 1)
        self.upsample1 = nn.Upsample(size=(25, 1))
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.upsample2 = nn.Upsample(size=(50, 2))
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.upsample3 = nn.Upsample(size=(100, 4))
        self.conv6 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1)
        self.activ1= nn.ELU()
        self.activ2= nn.ELU()
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.batchnorm3 = nn.BatchNorm2d(1)
    def forward(self, x):
        x = self.fc_decode(x)
        x = x.view(-1, 64, 13, 1)
        # Upsample and conv blocks with ELU, except for the final output
        x = self.activ1(self.batchnorm1(self.conv4(self.upsample1(x))))
        x = self.activ2(self.batchnorm2(self.conv5(self.upsample2(x))))
        x = self.batchnorm3(self.conv6(self.upsample3(x)))
        x=F.relu(x)
        return x
dataset= CustomDataLoader(rootDirs=['E:\\Work\\University\\PR\\datas\\voice_gender_detection\\females','E:\\Work\\University\\PR\\datas\\voice_gender_detection\\males'],duration=0.05,HaveSaveOutput=False,isNormalized=True)
trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
valloader = DataLoader(valset, batch_size=64, shuffle=False)
learning_rate = 1e-3
num_epochs = 100
writer = SummaryWriter(log_dir='./runs/'+datetime.now().strftime("%m-%d-%Y--%H-%M_")+'spec-autoencoder')
enc=encoder()
freeze_support()
model = AutoEncoder(enc, decoder()).cuda()
criterion = nn.MSELoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss=[]
for epoch in range(num_epochs):
    loss.append(train(model, trainloader, optimizer, criterion, epoch,writer))
    test(model, 'cuda', valloader, criterion, "Validation",writer)
torch.save(enc.state_dict(), 'sep_autoencoder.pth')
writer.close()