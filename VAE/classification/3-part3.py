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
class EncoderPart(nn.Module):
    def __init__(self,inChannel:int,outChannel:int):
        super().__init__()
        # self.conv = nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=2, padding=1)    # (1, 257, 51) -> (16, 129, 26)
        # self.conv = nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=2, padding=1)    # (1, 257, 51) -> (16, 129, 26)
        # self.batch=nn.BatchNorm2d(outChannel)
        # self.activ=nn.ELU()
        self.seq1=nn.Sequential(
             nn.Conv2d(inChannel, outChannel, kernel_size=3, padding=1),
             nn.Conv2d(outChannel, outChannel, kernel_size=3, stride=2, padding=1),
             nn.BatchNorm2d(outChannel),
             nn.ELU(),
             
        )
        self.seq2=nn.Sequential(
             nn.Conv2d(inChannel, outChannel, kernel_size=3, padding=1),
             nn.Conv2d(outChannel, outChannel, kernel_size=3, stride=2, padding=1),
             nn.BatchNorm2d(outChannel),
             nn.ELU(),
        )
        self.seq3=nn.Sequential(
            nn.Conv2d(inChannel, outChannel, kernel_size=5, padding=2),
            nn.Conv2d(outChannel, outChannel, kernel_size=5, padding=2),
             nn.Conv2d(outChannel, outChannel, kernel_size=3, stride=2, padding=1),
             nn.BatchNorm2d(outChannel),
             nn.ELU(),
        )
        self.seq4=nn.Sequential(
             nn.Conv2d(inChannel, outChannel, kernel_size=5, padding=2),
             nn.Conv2d(outChannel, outChannel, kernel_size=5, padding=2),
             nn.Conv2d(outChannel, outChannel, kernel_size=3, stride=2, padding=1),
             nn.BatchNorm2d(outChannel),
             nn.ELU(),
        )
        self.seq5=nn.Sequential(
            nn.Conv2d(inChannel, outChannel, kernel_size=3, padding=1),
             nn.Conv2d(outChannel, outChannel, kernel_size=3, stride=2, padding=1),
             nn.BatchNorm2d(outChannel),
             nn.ELU(),
        )
        self.elu=nn.ELU()
        self.reduce = nn.Conv2d(5 * outChannel, outChannel, kernel_size=1)
        self.batch=nn.BatchNorm2d(outChannel)
    def forward(self,x):
        # return self.activ(self.batch(self.conv(x)))
        # return self.seq1(x)+self.seq2(x)+self.seq3(x)+self.seq4(x)
        out1 = self.seq1(x)
        out2 = self.seq2(x)
        out3 = self.seq3(x)
        out4 = self.seq4(x)
        out5 = self.seq5(x)
        # Concatenate along the channel dimension (dim=1)
        concatenated = torch.cat([out1, out2, out3,out4,out5], dim=1)
        reduced = self.reduce(concatenated)
        reduced= self.batch(reduced)
        reduced=self.elu(reduced)
        return reduced
class DecoderPart(nn.Module):
    def __init__(self, inChannel:int,outChannel:int,outputPadding=0,isHaveActive=False):
        super().__init__()
        self.seq1=nn.Sequential(
            nn.ConvTranspose2d(inChannel,outChannel, kernel_size=5,  padding=2),
            nn.ConvTranspose2d(outChannel,outChannel, kernel_size=3, stride=2, padding=1, output_padding=outputPadding),
            nn.BatchNorm2d(outChannel),
            nn.ELU())
        self.seq2=nn.Sequential(
            nn.ConvTranspose2d(inChannel,outChannel, kernel_size=3,  padding=1),
            nn.ConvTranspose2d(outChannel,outChannel, kernel_size=3, stride=2, padding=1, output_padding=outputPadding),
            nn.BatchNorm2d(outChannel),
            nn.ELU())
        self.seq3=nn.Sequential(
            nn.ConvTranspose2d(inChannel,outChannel, kernel_size=5,  padding=2),
            nn.ConvTranspose2d(outChannel,outChannel, kernel_size=5,  padding=2),
            nn.ConvTranspose2d(outChannel,outChannel, kernel_size=3, stride=2, padding=1, output_padding=outputPadding),
            nn.BatchNorm2d(outChannel),
            nn.ELU())
        self.seq4=nn.Sequential(
            nn.ConvTranspose2d(inChannel,outChannel, kernel_size=5,  padding=2),
            nn.ConvTranspose2d(outChannel,outChannel, kernel_size=5,  padding=2),
            nn.ConvTranspose2d(outChannel,outChannel, kernel_size=3, stride=2, padding=1, output_padding=outputPadding),
            nn.BatchNorm2d(outChannel),
            nn.ELU())
        self.seq5=nn.Sequential(
            nn.ConvTranspose2d(inChannel, outChannel, kernel_size=3, padding=1),
            nn.ConvTranspose2d(outChannel,outChannel, kernel_size=3, stride=2, padding=1, output_padding=outputPadding),
            nn.BatchNorm2d(outChannel),
            nn.ELU())
        self.reduce = nn.Conv2d(5 * outChannel, outChannel, kernel_size=1)
        # self.deconv=nn.ConvTranspose2d(inChannel,outChannel, kernel_size=3, stride=2, padding=1, output_padding=outputPadding)
        # self.batch=nn.BatchNorm2d(outChannel)
        self.activ=nn.ELU()
        self.isHaveActive=isHaveActive
    def forward(self,x):
        # return self.activ(self.batch(self.deconv(x)))
        # x=self.seq1(x)+self.seq2(x)+self.seq3(x)+self.seq4(x)
        # if self.isHaveActive:
        #     x= self.activ(x)
        # return x
        out1 = self.seq1(x)
        out2 = self.seq2(x)
        out3 = self.seq3(x)
        out4 = self.seq4(x)
        out5 = self.seq5(x)
        concatenated = torch.cat([out1, out2, out3,out4,out5], dim=1)
        reduced = self.reduce(concatenated)
        return reduced
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        # Conv2d layers with kernel_size=3, stride=2, padding=1
        self.seq1=nn.Sequential( EncoderPart(1,16),   
         EncoderPart(16,32),   
         EncoderPart(32,64)  , 
         EncoderPart(64,128) , 
         EncoderPart(128,256), 
         EncoderPart(256,512), 
        )
        self.seq2=nn.Sequential( EncoderPart(1,16),   
         EncoderPart(16,32),   
         EncoderPart(32,64)  , 
         EncoderPart(64,128) , 
         EncoderPart(128,256), 
         EncoderPart(256,512), 
        )
        self.seq3=nn.Sequential( EncoderPart(1,16),   
         EncoderPart(16,32),   
         EncoderPart(32,64)  , 
         EncoderPart(64,128) , 
         EncoderPart(128,256), 
         EncoderPart(256,512), 
        )
        self.flatten = nn.Flatten()
        
        # Linear layers for mean and log-variance
        self.fc_mu = nn.Linear(512 * 9 * 1, latent_dim)
        self.fc_logvar = nn.Linear(512 * 9 * 1, latent_dim)
    def forward(self,x):
        x1=x[:,:,:171,:]
        x2=x[:,:,171:342,:]
        x3=x[:,:,342:,:]
        x1=self.seq1(x1)
        x2=self.seq2(x2)
        x3=self.seq3(x3)
        x=torch.cat((x1,x2,x3),dim=2)
        x = self.flatten(x)  # Shape: (batch_size, 256*9*2)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        # Linear layer to map latent vector to the shape before flattening
        self.fc = nn.Linear(latent_dim, 512 * 9 * 1)
        self.reshape = lambda x: x.view(-1, 512,9, 1)  # Reshape to (256, 9, 2)
        
        # Conv2dTranspose layers with appropriate output_padding
        self.seq1=nn.Sequential(
         DecoderPart(512,256,(1, 1)),
         DecoderPart(256,128,(0, 1))  ,
         DecoderPart(128,64,(1,0)), 
         DecoderPart(64,32),  
         DecoderPart(32,16,(1,1)),
       DecoderPart(16,1,0,False))
        self.seq2=nn.Sequential(
         DecoderPart(512,256,(1, 1)),
         DecoderPart(256,128,(0, 1))  ,
         DecoderPart(128,64,(1,0)), 
         DecoderPart(64,32),  
         DecoderPart(32,16,(1,1)),
       DecoderPart(16,1,0,False))
        self.seq3=nn.Sequential(
         DecoderPart(512,256,(1, 1)),
         DecoderPart(256,128,(0, 1))  ,
         DecoderPart(128,64,(1,0)), 
         DecoderPart(64,32),  
         DecoderPart(32,16,(1,1)),
       DecoderPart(16,1,0,False))

    def forward(self, z):
        x = self.fc(z)
        x = self.reshape(x)
        x1=x[:,:,:3,:]
        x2=x[:,:,3:6,:]
        x3=x[:,:,6:,:]
        x1=self.seq1(x1)
        x2=self.seq2(x2)
        x3=self.seq3(x3)
        x=torch.cat((x1,x2,x3),dim=2)
        return x
class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar
vae=VAE().cuda()
lr=1e-3
optimz=optim.Adam(vae.parameters(),lr)
epochNumber=100
nowTime=datetime.now().strftime("%m-%d-%Y--%H-%M_")
writer = SummaryWriter(log_dir='./runs/'+nowTime+'3')
dataset= CustomDataLoader(rootDirs=['E:\\Work\\University\\PR\\datas\\voice_gender_detection\\females','E:\\Work\\University\\PR\\datas\\voice_gender_detection\\males'],sr=16000,duration=0.5,HaveSaveOutput=False,isNormalized=True)
trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.9), len(dataset)-int(len(dataset)*0.9)])
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
    # plot_latent_spaceVAE(vae,valloader,epoch,writer)
    plot_latent_spaceVAE(vae,valloader,epoch,writer)
    # plot_latent_spaceVAE(vae,allloader,epoch,writer,kind="all")
writer.close()