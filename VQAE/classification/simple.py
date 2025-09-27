import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
sys.path.append('e:\\Work\\University\\PR\\project1')
from BaseFunc import plot_codebook_tsne, plot_latent_spaceVAE, plot_latent_spaceVQAE, plotit
import torch_directml
from DatasetLoader import CustomDataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
class EncoderPart(nn.Module):
    def __init__(self,inChannel:int,outChannel:int,kernel=(7,3),pardding=(3,1)):
        super().__init__()
        # self.conv = nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=2, padding=1)    # (1, 257, 51) -> (16, 129, 26)
        # self.conv = nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=2, padding=1)    # (1, 257, 51) -> (16, 129, 26)
        # self.batch=nn.BatchNorm2d(outChannel)
        # self.activ=nn.ELU()
        self.seq1=nn.Sequential(
             nn.Conv2d(inChannel, outChannel, kernel_size=kernel, stride=2, padding=pardding),
             nn.BatchNorm2d(outChannel),
             nn.ELU(),
             
        )
        self.seq2=nn.Sequential(
             nn.Conv2d(inChannel, outChannel, kernel_size=kernel, stride=2, padding=pardding),
             nn.BatchNorm2d(outChannel),
             nn.ELU(),
        )
        self.seq3=nn.Sequential(
             nn.Conv2d(inChannel, outChannel, kernel_size=kernel, stride=2, padding=pardding),
             nn.BatchNorm2d(outChannel),
             nn.ELU(),
        )
        self.seq4=nn.Sequential(
             nn.Conv2d(inChannel, outChannel, kernel_size=kernel, stride=2, padding=pardding),
             nn.BatchNorm2d(outChannel),
             nn.ELU(),
        )
        self.reduce = nn.Conv2d(4 * outChannel, outChannel, kernel_size=1)
    def forward(self,x):
        out1 = self.seq1(x)
        out2 = self.seq2(x)
        out3 = self.seq3(x)
        out4 = self.seq4(x)
        concatenated = torch.cat([out1, out2, out3,out4], dim=1)
        reduced = self.reduce(concatenated)
        return reduced
class DecoderPart(nn.Module):
    def __init__(self, inChannel:int,outChannel:int,outputPadding=0,kernel=3):
        super().__init__()
        self.seq1=nn.Sequential(
            nn.ConvTranspose2d(inChannel,outChannel, kernel_size=kernel, stride=2, padding=1, output_padding=outputPadding),
            nn.BatchNorm2d(outChannel),
            nn.ELU())
        self.seq2=nn.Sequential(
            nn.ConvTranspose2d(inChannel,outChannel, kernel_size=kernel, stride=2, padding=1, output_padding=outputPadding),
            nn.BatchNorm2d(outChannel),
            nn.ELU())
        self.seq3=nn.Sequential(
            nn.ConvTranspose2d(inChannel,outChannel, kernel_size=kernel, stride=2, padding=1, output_padding=outputPadding),
            nn.BatchNorm2d(outChannel),
            nn.ELU())
        self.seq4=nn.Sequential(
            nn.ConvTranspose2d(inChannel,outChannel, kernel_size=kernel, stride=2, padding=1, output_padding=outputPadding),
            nn.BatchNorm2d(outChannel),
            nn.ELU())
        self.reduce = nn.Conv2d(4 * outChannel, outChannel, kernel_size=1)
    def forward(self,x):
        out1 = self.seq1(x)
        out2 = self.seq2(x)
        out3 = self.seq3(x)
        out4 = self.seq4(x)
        concatenated = torch.cat([out1, out2, out3,out4], dim=1)
        reduced = self.reduce(concatenated)
        return reduced
    

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
        return x

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(VectorQuantizer, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z_e):
        z_e_flat = z_e.permute(0, 2, 3, 1).reshape(-1, z_e.size(1))
        distances = (z_e_flat ** 2).sum(dim=1, keepdim=True) + \
                    (self.embedding.weight ** 2).sum(dim=1) - \
                    2 * torch.matmul(z_e_flat, self.embedding.weight.t())
        encoding_indices = torch.argmin(distances, dim=1)
        z_q_flat = self.embedding(encoding_indices)
        z_q = z_q_flat.view(z_e.shape)
        return z_q, encoding_indices

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv1 = DecoderPart(256, 128, (0, 1))
        self.deconv2 = DecoderPart(128, 64)
        self.deconv3 = DecoderPart(64, 32)
        self.deconv4 = DecoderPart(32, 16, (0, 1))
        self.deconv5 = DecoderPart(16, 1, 0)

    def forward(self, z_q):
        x = self.deconv1(z_q)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        return x

class VQAE(nn.Module):
    def __init__(self, num_embeddings=32, embedding_dim=256, beta=0.25):
        super(VQAE, self).__init__()
        self.encoder = Encoder()
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = Decoder()
        self.beta = beta

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, indices = self.quantizer(z_e)
        z_q_st = z_e + (z_q - z_e).detach()
        recon_x = self.decoder(z_q_st)
        return recon_x, z_e, z_q

def VQAE_loss(recon_x, x, z_e, z_q, beta=0.25):
    recon_loss = F.mse_loss(recon_x, x)
    codebook_loss = F.mse_loss(z_e.detach(), z_q)
    commitment_loss = F.mse_loss(z_e, z_q.detach())
    total_loss = recon_loss + codebook_loss + beta * commitment_loss
    return total_loss
dml = torch_directml.device()
vqae = VQAE().to(dml)
lr = 1e-3
optimz = optim.Adam(vqae.parameters(), lr)
epochNumber = 100
nowTime = datetime.now().strftime("%m-%d-%Y--%H-%M_")
writer = SummaryWriter(log_dir='./runs/' + nowTime + '2')

dataset = CustomDataLoader(
    rootDirs=['E:\\Work\\University\\PR\\datas\\voice_gender_detection\\females',
              'E:\\Work\\University\\PR\\datas\\voice_gender_detection\\males'],
    sr=16000, duration=0.5, HaveSaveOutput=False, isNormalized=True, randomSelection=False
)
trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
valloader = DataLoader(valset, batch_size=32, shuffle=False)

for epoch in range(epochNumber):
    vqae.train()
    loss_training = []
    for batch, [spec, _] in enumerate(trainloader):
        spec = spec.to(dml)
        optimz.zero_grad()
        recon_x, x_enc, z_q = vqae(spec)
        loss = VQAE_loss(recon_x, spec, x_enc,z_q)
        loss.backward()
        loss_training.append(loss.item())
        optimz.step()

    vqae.eval()
    testLoss = []
    with torch.no_grad():
        for batch, [spec, _] in enumerate(valloader):
            spec = spec.to(dml)
            recon_x, x_enc, z_q = vqae(spec)
            loss = VQAE_loss(recon_x, spec, x_enc,z_q)
            testLoss.append(loss.item())
            if batch == 0:
                plotit(spec[0], epoch, "Real", writer)
                plotit(recon_x[0], epoch, "Predict", writer)
    lossMean=np.mean(testLoss)
    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, lossMean))
    writer.add_scalar("Test Loss", lossMean, epoch)
    writer.add_scalar("Training Loss", np.mean(loss_training), epoch)
    # Optional: Remove or adapt plot_latent_spaceVAE if needed
    # plot_codebook_tsne(vqae,valloader,epoch,writer)
torch.save(vqae.state_dict(), nowTime + '.pth')
writer.close()