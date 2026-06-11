import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
sys.path.append('e:\\Work\\University\\PR\\project1')
from RegDataLoader import RegDataLoader
from BaseFunc import claculateSVCRegrestin, createTensorboard, plotit
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# ====================== ENCODER ======================
class EncoderPart(nn.Module):
    def __init__(self, inChannel: int, outChannel: int, stride=2):
        super().__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(outChannel),
            nn.ELU(),
        )
        self.seq2 = nn.Sequential(
            nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(outChannel),
            nn.ELU(),
        )
        self.seq3 = nn.Sequential(
            nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(outChannel),
            nn.ELU(),
        )
        self.seq4 = nn.Sequential(
            nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(outChannel),
            nn.ELU(),
        )
        self.reduce = nn.Conv2d(4 * outChannel, outChannel, kernel_size=1)

    def forward(self, x):
        out1 = self.seq1(x)
        out2 = self.seq2(x)
        out3 = self.seq3(x)
        out4 = self.seq4(x)
        concatenated = torch.cat([out1, out2, out3, out4], dim=1)
        return self.reduce(concatenated)


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = EncoderPart(1, 16)
        self.conv2 = EncoderPart(16, 32)
        self.conv3 = EncoderPart(32, 64)
        self.conv4 = EncoderPart(64, 128)
        self.conv5 = EncoderPart(128, 256)
        self.conv6 = EncoderPart(256, 512)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * 9 * 1, latent_dim)   # deterministic latent

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.flatten(x)
        z = self.fc(x)
        return z


# ====================== DECODER ======================
class DecoderPart(nn.Module):
    def __init__(self, inChannel: int, outChannel: int, outputPadding=0, isHaveActive=False):
        super().__init__()
        self.seq1 = nn.Sequential(
            nn.ConvTranspose2d(inChannel, outChannel, kernel_size=3, stride=2, padding=1, output_padding=outputPadding),
            nn.BatchNorm2d(outChannel),
            nn.ELU())
        self.seq2 = nn.Sequential(
            nn.ConvTranspose2d(inChannel, outChannel, kernel_size=3, stride=2, padding=1, output_padding=outputPadding),
            nn.BatchNorm2d(outChannel),
            nn.ELU())
        self.seq3 = nn.Sequential(
            nn.ConvTranspose2d(inChannel, outChannel, kernel_size=3, stride=2, padding=1, output_padding=outputPadding),
            nn.BatchNorm2d(outChannel),
            nn.ELU())
        self.seq4 = nn.Sequential(
            nn.ConvTranspose2d(inChannel, outChannel, kernel_size=3, stride=2, padding=1, output_padding=outputPadding),
            nn.BatchNorm2d(outChannel),
            nn.ELU())
        self.reduce = nn.Conv2d(4 * outChannel, outChannel, kernel_size=1)
        self.isHaveActive = isHaveActive

    def forward(self, x):
        out1 = self.seq1(x)
        out2 = self.seq2(x)
        out3 = self.seq3(x)
        out4 = self.seq4(x)
        concatenated = torch.cat([out1, out2, out3, out4], dim=1)
        return self.reduce(concatenated)


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 512 * 9 * 1)
        self.reshape = lambda x: x.view(-1, 512, 9, 1)

        self.deconv1 = DecoderPart(512, 256, (0, 1))
        self.deconv2 = DecoderPart(256, 128, (0, 1))
        self.deconv3 = DecoderPart(128, 64)
        self.deconv4 = DecoderPart(64, 32)
        self.deconv5 = DecoderPart(32, 16, (0, 1))
        self.deconv6 = DecoderPart(16, 1, 0, False)

    def forward(self, z):
        x = self.fc(z)
        x = self.reshape(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.deconv6(x)
        return x


# ====================== AUTOENCODER ======================
class AE(nn.Module):
    def __init__(self, latent_dim=512):
        super(AE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x, z   # return both for downstream tasks (e.g. SVC)


# ====================== TRAINING ======================
writer = createTensorboard("reg_ae_1")   # changed name for clarity

model = AE(512).cuda()
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr)

# Use MSE for reconstruction
reconstruction_loss = nn.MSELoss()

dataset = RegDataLoader(
    rootDir="E:/Work/University/PR/datas/voice-icar-federico-ii-database-1.0.0/",
    ids=list(range(1, 209)),
    metadataPath="E:\\Work\\University\\PR\\project1\\base-reg\\out.json",
    sr=8000,
    isNormalized=True
)

trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
valloader = DataLoader(valset, batch_size=32, shuffle=False)

epochNumber = 40

for epoch in range(epochNumber):
    loss_training = []
    model.train()
    
    for batch, [spec, _, _] in enumerate(trainloader):
        spec = spec.cuda()
        optimizer.zero_grad()
        
        recon_x, _ = model(spec)
        loss = reconstruction_loss(recon_x, spec)
        
        loss.backward()
        optimizer.step()
        loss_training.append(loss.item())
    
    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, np.mean(loss_training)))

    # Validation
    model.eval()
    testLoss = []
    val_latents = []
    val_rsis = []
    val_vhis = []
    
    with torch.no_grad():
        for batch, [spec, rsi, vhi] in enumerate(valloader):
            spec = spec.cuda()
            recon_x, latent = model(spec)
            
            val_latents.append(latent.cpu().numpy())
            loss = reconstruction_loss(recon_x, spec)
            testLoss.append(loss.item())
            
            val_rsis.append(rsi.numpy())
            val_vhis.append(vhi.numpy())
            
            if batch == 0:
                plotit(spec[0], epoch, "Real", writer)
                plotit(recon_x[0], epoch, "Predict", writer)

    writer.add_scalar("Test Loss", np.mean(testLoss), epoch)
    writer.add_scalar("Training Loss", np.mean(loss_training), epoch)

    # Concatenate for downstream tasks
    val_vhis = np.concatenate(val_vhis, axis=0)
    val_rsis = np.concatenate(val_rsis, axis=0)
    val_latents = np.concatenate(val_latents, axis=0)

    claculateSVCRegrestin(val_latents, val_rsis, epoch, writer, 'VAL', "RSI")

writer.close()