import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
sys.path.append('e:\\Work\\University\\PR\\project1')
from BaseFunc import plotit,claculateSVCRegrestin  # Assuming plot_latent_spaceVAE is not directly applicable
from RegDataLoader import RegDataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime




# EncoderPart (unchanged)
class EncoderPart(nn.Module):
    def __init__(self, inChannel: int, outChannel: int, kernel=(7,3), padding=(3,1)):
        super().__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(inChannel, outChannel, kernel_size=kernel, stride=2, padding=padding),
            nn.BatchNorm2d(outChannel),
            nn.ELU(),
        )
        self.seq2 = nn.Sequential(
            nn.Conv2d(inChannel, outChannel, kernel_size=kernel, stride=2, padding=padding),
            nn.BatchNorm2d(outChannel),
            nn.ELU(),
        )
        self.seq3 = nn.Sequential(
            nn.Conv2d(inChannel, outChannel, kernel_size=kernel, stride=2, padding=padding),
            nn.BatchNorm2d(outChannel),
            nn.ELU(),
        )
        self.seq4 = nn.Sequential(
            nn.Conv2d(inChannel, outChannel, kernel_size=kernel, stride=2, padding=padding),
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
        reduced = self.reduce(concatenated)
        return reduced

# Modified Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = EncoderPart(1, 16)
        self.conv2 = EncoderPart(16, 32)
        self.conv3 = EncoderPart(32, 64, 3, 1)
        self.conv4 = EncoderPart(64, 128, 3, 1)
        self.conv5 = EncoderPart(128, 256, 3, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x  # (batch_size, 256, 17, 2)

# DecoderPart (unchanged)
class DecoderPart(nn.Module):
    def __init__(self, inChannel: int, outChannel: int, outputPadding=0, kernel=3):
        super().__init__()
        self.seq1 = nn.Sequential(
            nn.ConvTranspose2d(inChannel, outChannel, kernel_size=kernel, stride=2, padding=1, output_padding=outputPadding),
            nn.BatchNorm2d(outChannel),
            nn.ELU()
        )
        self.seq2 = nn.Sequential(
            nn.ConvTranspose2d(inChannel, outChannel, kernel_size=kernel, stride=2, padding=1, output_padding=outputPadding),
            nn.BatchNorm2d(outChannel),
            nn.ELU()
        )
        self.seq3 = nn.Sequential(
            nn.ConvTranspose2d(inChannel, outChannel, kernel_size=kernel, stride=2, padding=1, output_padding=outputPadding),
            nn.BatchNorm2d(outChannel),
            nn.ELU()
        )
        self.seq4 = nn.Sequential(
            nn.ConvTranspose2d(inChannel, outChannel, kernel_size=kernel, stride=2, padding=1, output_padding=outputPadding),
            nn.BatchNorm2d(outChannel),
            nn.ELU()
        )
        self.reduce = nn.Conv2d(4 * outChannel, outChannel, kernel_size=1)

    def forward(self, x):
        out1 = self.seq1(x)
        out2 = self.seq2(x)
        out3 = self.seq3(x)
        out4 = self.seq4(x)
        concatenated = torch.cat([out1, out2, out3, out4], dim=1)
        reduced = self.reduce(concatenated)
        return reduced

# Modified Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv1 = DecoderPart(256, 128, (0, 1))
        self.deconv2 = DecoderPart(128, 64)
        self.deconv3 = DecoderPart(64, 32)
        self.deconv4 = DecoderPart(32, 16, (0, 1))
        self.deconv5 = DecoderPart(16, 1, 0)

    def forward(self, z):
        x = self.deconv1(z)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        return x

# VQ-VAE Class
class VQVAE(nn.Module):
    def __init__(self, K=512, embedding_dim=256, beta=0.25):
        super(VQVAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.codebook = nn.Embedding(K, embedding_dim)
        self.beta = beta

    def quantize(self, z_e):
        batch_size, _, height, width = z_e.shape
        z_e_permuted = z_e.permute(0, 2, 3, 1)  # (batch_size, 17, 2, 256)
        z_e_flat = z_e_permuted.reshape(batch_size, -1, 256)  # (batch_size, 34, 256)
        
        codebook = self.codebook.weight  # (K, 256)
        z_e_norms = torch.sum(z_e_flat ** 2, dim=2, keepdim=True)  # (batch_size, 34, 1)
        codebook_norms = torch.sum(codebook ** 2, dim=1, keepdim=True)  # (K, 1)
        inner_product = torch.bmm(z_e_flat, codebook.t().unsqueeze(0).repeat(batch_size, 1, 1))  # (batch_size, 34, K)
        distances = z_e_norms + codebook_norms.t() - 2 * inner_product  # (batch_size, 34, K)
        
        indices = torch.argmin(distances, dim=2)  # (batch_size, 34)
        z_q_flat = self.codebook(indices)  # (batch_size, 34, 256)
        z_q_permuted = z_q_flat.view(batch_size, height, width, 256)
        z_q = z_q_permuted.permute(0, 3, 1, 2)  # (batch_size, 256, 17, 2)
        
        return z_q, indices

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, indices = self.quantize(z_e)
        recon_x = self.decoder(z_q)
        return recon_x, z_e, z_q, indices


# Training and Post-Processing
vqvae = VQVAE(K=512).cuda()
lr = 1e-3
optimz = optim.Adam(vqvae.parameters(), lr)
epochNumber = 100
nowTime = datetime.now().strftime("%m-%d-%Y--%H-%M_")
writer = SummaryWriter(log_dir='./runs/' + nowTime + 'VQVAE')
dataset= RegDataLoader(rootDir="E:/Work/University/PR/datas/voice-icar-federico-ii-database-1.0.0/",ids=list(range(1, 209)),metadataPath="E:\\Work\\University\\PR\\project1\\base-reg\\out.json",sr=8000,isNormalized=True)
trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
valloader = DataLoader(valset, batch_size=32, shuffle=False)

# Training Loop
for epoch in range(epochNumber):
    vqvae.train()
    loss_training = []
    for batch, [spec, _,_] in enumerate(trainloader):
        spec = spec.cuda()
        optimz.zero_grad()
        recon_x, z_e, z_q, indices = vqvae(spec)
        recon_loss = F.mse_loss(recon_x, spec)
        codebook_loss = F.mse_loss(z_e.detach(), z_q)
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + vqvae.beta * commitment_loss
        total_loss = recon_loss + vq_loss
        total_loss.backward()
        loss_training.append(total_loss.item())
        optimz.step()
    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, np.mean(loss_training)))
    
    
    # Validation
    vqvae.eval()
    test_loss = []
            
    val_latents = []
    val_rsis = []
    val_vhis = []
    
    with torch.no_grad():
        for batch, [spec, rsi,vhi] in enumerate(valloader):
            spec = spec.cuda()
            recon_x, z_e, z_q, indices = vqvae(spec)
            recon_loss = F.mse_loss(recon_x, spec)
            codebook_loss = F.mse_loss(z_e.detach(), z_q)
            commitment_loss = F.mse_loss(z_e, z_q.detach())
            z_q_flat = z_q.reshape(z_q.size(0), -1)
            val_latents.append(z_q_flat.cpu().numpy())
            val_rsis.append(rsi.numpy())
            val_vhis.append(vhi.numpy())
            vq_loss = codebook_loss + vqvae.beta * commitment_loss
            total_loss = recon_loss + vq_loss
            test_loss.append(total_loss.item())
            if batch == 0:
                plotit(spec[0], epoch, "Real", writer)
                plotit(recon_x[0], epoch, "Predict", writer)
    writer.add_scalar("Test Loss", np.mean(test_loss), epoch)
    writer.add_scalar("Training Loss", np.mean(loss_training), epoch)
    val_latents = np.concatenate(val_latents, axis=0)
    val_vhis=np.concatenate(val_vhis,axis=0)
    val_rsis=np.concatenate(val_rsis,axis=0)
    claculateSVCRegrestin(val_latents,val_rsis,epoch,writer,'Val','RSI')

writer.close()
