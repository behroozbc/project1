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
    correct = 0
    latent_vectors = []
    labels = []
    
    
    with torch.no_grad():
        for batch_idx, [_,spec,_,label] in enumerate(test_loader):
            spec= spec.cuda()
            output = model(spec)
            if batch_idx == 0:
                plotit(spectrogram=spec[0],epoch=epoch,kind="Real",writer= writer)
                plotit(spectrogram=output[0],epoch=epoch,kind="Reconstructed",writer=writer)
            test_loss .append( criterion(output, spec).item())  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            latent_vector = model.encoder(spec)
            latent_vectors.append(latent_vector)
            labels+=label
    latent_vectors = torch.cat(latent_vectors, dim=0)
    plotResults(epoch, latent_vectors, labels,writer)
    writer.add_scalar("Test Loss",np.mean(test_loss),epoch)


class EncoderPart(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,16,kernel_size=3,padding=1)
        self.pool1=nn.MaxPool2d(2)
        self.batch1=nn.BatchNorm2d(16)
        self.elu1=nn.ELU()
        
        self.conv2= nn.Conv2d(16,32,3,padding=1)
        self.pool2=nn.MaxPool2d(2)
        self.batch2=nn.BatchNorm2d(32)
        self.elu2=nn.ELU()
        
        # self.conv3= nn.Conv2d(16,32,3,padding=1)
        # self.pool3=nn.MaxPool2d(2)
        # self.batch3=nn.BatchNorm2d(32) 17*2
        # self.elu3=nn.ELU()
    def forward(self,x):
        # x [67,4]
        x=self.conv1(x)
        x=self.pool1(x)
        x=self.batch1(x)
        x=self.elu1(x)
        
        
        x=self.conv2(x)
        x=self.pool2(x)
        x=self.batch2(x)
        x=self.elu2(x)
        
        # x=self.conv3(x)
        # x=self.pool3(x)
        # x=self.batch3(x)
        # x=self.elu3(x)
        return x
class DecoderPart(nn.Module):
    def __init__(self):
        super().__init__()
        # Block 1:  [32, 16, 1] -> [16, 32,2]
        self.upsample1 = nn.Upsample(size=(32, 2), mode='bilinear', align_corners=False)
        self.conv1 = nn.ConvTranspose2d(in_channels=32,out_channels= 16, kernel_size=3, padding=1)
        self.batch1 = nn.BatchNorm2d(16)
        self.elu1 = nn.ELU()
        
        # Block 2: [16, 32,2] -> [1, 67,4]
        self.upsample2 = nn.Upsample(size=(67, 4), mode='bilinear', align_corners=False)
        self.conv2 = nn.ConvTranspose2d(in_channels= 16,out_channels= 1, kernel_size=3, padding=1)
        self.batch2 = nn.BatchNorm2d(1)
        self.elu2 = nn.ELU()
        
        # # Block 3: [8, 33, 3] -> [1, 67, 6]
        # self.upsample3 = nn.Upsample(size=(67, 6), mode='bilinear', align_corners=False)
        # self.conv3 = nn.Conv2d(8, 1, kernel_size=3, padding=1)
        # self.batch3 = nn.BatchNorm2d(1)  # Optional, included for symmetry
        # self.elu3 = nn.ELU()  # Could use sigmoid if output range [0,1] is desired

    def forward(self, x):
        x = self.upsample1(x)
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.elu1(x)
        
        x = self.upsample2(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.elu2(x)
        
        # x = self.upsample3(x)
        # x = self.conv3(x)
        # x = self.batch3(x)
        # x = self.elu3(x)
        return x
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.part1=EncoderPart()
        self.part2=EncoderPart()
        self.part3=EncoderPart()
        self.fc=nn.Linear(32*48*1,128)
    def forward(self,x):
        x1=x[:,:,:67,:]
        x2=x[:,:,67:134,:]
        x3=x[:,:,134:,:]
        x1=self.part1(x1)
        x2=self.part2(x2)
        x3=self.part3(x3)
        cated=torch.cat((x1,x2,x3),dim=2)
        # (batch, 32,48,1)
        cated = cated.view(cated.size(0), -1)
        return self.fc(cated)
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc=nn.Linear(128,32*48*1)
        self.part1=DecoderPart()
        self.part2=DecoderPart()
        self.part3=DecoderPart()
    def forward(self,x):
        x=self.fc(x) #[32, 8, 1]
        x = x.view(x.size(0), 32, 48, 1)
        x1=x[:,:,:16,:]
        x2=x[:,:,16:32,:]
        x3=x[:,:,32:,:]
        x1=self.part1(x1)
        x2=self.part2(x2)
        x3=self.part3(x3)
        out=torch.cat((x1,x2,x3),dim=2)
        return out
dataset= CustomDataLoader(rootDirs=['E:\\Work\\University\\PR\\datas\\voice_gender_detection\\females','E:\\Work\\University\\PR\\datas\\voice_gender_detection\\males'],sr=16000,duration=0.04,HaveSaveOutput=False,isNormalized=True)
trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
valloader = DataLoader(valset, batch_size=32, shuffle=False)
learning_rate = 1e-3
num_epochs = 100
enc=Encoder()
writer = SummaryWriter(log_dir='./runs/'+datetime.now().strftime("%m-%d-%Y--%H-%M_")+'3part-spec-autoencoder')
model = AutoEncoder(enc, Decoder()).cuda()
criterion = nn.MSELoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss=[]
for epoch in range(num_epochs):
    loss.append(train(model, trainloader, optimizer, criterion, epoch,writer))
    test(model, 'cuda', valloader, criterion, writer,"Validation")
torch.save(enc.state_dict(), 'split-3-ckeck-sep_autoencoder.pth')
