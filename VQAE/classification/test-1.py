from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
sys.path.append('e:\\Work\\University\\PR\\project1')
from BaseFunc import plot_codebook_tsne, plot_latent_spaceVAE, plot_latent_spaceVQAE, plotit
from DatasetLoader import CustomDataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
class VectorQuantizer(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss  # [B x D x H x W]
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
class VQVAE(nn.Module):
    def __init__(self,embedding_dim,num_embeddings,beta: float = 0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta
        self.vq_layer = VectorQuantizer(num_embeddings,
                                        embedding_dim,
                                        self.beta)
        # Conv2d layers with kernel_size=3, stride=2, padding=1
        # self.conv1 = EncoderPart(1,16)    # (1, 257, 51) -> (16, 129, 26)
        # self.conv2 = EncoderPart(16,32)   # (16, 129, 26) -> (32, 65, 13)
        # self.conv3 = EncoderPart(32,64,3,1)   # (32, 65, 13) -> (64, 33, 7)
        # self.conv4 = EncoderPart(64,128,3,1)  # (64, 33, 7) -> (128, 17, 4)
        # self.conv5 = EncoderPart(128,256,3,1) # (128, 17, 4) -> (256, 9, 2)
        self.encoderSeq=nn.Sequential(EncoderPart(1,16),EncoderPart(16,32),EncoderPart(32,64,3,1),EncoderPart(64,128,3,1),EncoderPart(128,self.num_embeddings,3,1))
        #  # Conv2dTranspose layers with appropriate output_padding
        # self.deconv1 = DecoderPart(256,128,(0, 1))  # (256, 9, 2) -> (128, 17, 4)
        # self.deconv2 = DecoderPart(128,64) # (128, 17, 4) -> (64, 33, 7)
        # self.deconv3 = DecoderPart(64,32)                           # (64, 33, 7) -> (32, 65, 13)
        # self.deconv4 = DecoderPart(32,16,(0,1))    # (32, 65, 13) -> (16, 129, 26)
        # self.deconv5 = DecoderPart(16,1,0)                            # (16, 129, 26) -> (1, 257, 51)
        
        self.decoderSeq=nn.Sequential(DecoderPart(self.num_embeddings,128,(0, 1)),DecoderPart(128,64),DecoderPart(64,32),DecoderPart(32,16,(0,1)),DecoderPart(16,1,0))
    def encoder(self,x)-> torch.Tensor:
        return self.encoderSeq(x)
    def decoder(self,z):
        return self.decoderSeq(z)
    def forward(self,input):
        encoding = self.encoder(input)
        quantized_inputs, vq_loss = self.vq_layer(encoding)
        return [self.decoder(quantized_inputs), input, vq_loss]
    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        vq_loss = args[2]

        recons_loss = F.mse_loss(recons, input)

        loss = recons_loss + vq_loss
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'VQ_Loss':vq_loss}
        
vqae = VQVAE(256,64).to('cuda')
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
        spec = spec.to('cuda')
        optimz.zero_grad()
        all= vqae(spec)
        loss = vqae.loss_function(*all )
        loss['loss'].backward()
        loss_training.append(loss['loss'].item())
        optimz.step()

    vqae.eval()
    testLoss = []
    with torch.no_grad():
        for batch, [spec, _] in enumerate(valloader):
            spec = spec.to('cuda')
            all= vqae(spec)
            loss = vqae.loss_function(*all )
            testLoss.append(loss['loss'].item())
            if batch == 0:
                plotit(spec[0], epoch, "Real", writer)
                plotit(all[0][0], epoch, "Predict", writer)
    lossMean=np.mean(testLoss)
    
    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, lossMean))
    writer.add_scalar("Test Loss", lossMean, epoch)
    writer.add_scalar("Training Loss", np.mean(loss_training), epoch)
    # Optional: Remove or adapt plot_latent_spaceVAE if needed
    # plot_codebook_tsne(vqae,valloader,epoch,writer)
torch.save(vqae.state_dict(), nowTime + '.pth')
writer.close()