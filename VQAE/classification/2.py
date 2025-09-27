import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
sys.path.append('e:\\Work\\University\\PR\\project1')
from BaseFunc import  claculateSVC, plot_latent_spaceVAE, plotit,plot_latent_space
from VQAEBase import plot_codebook_usage,plot_codebook_tsne
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
        self.elu=nn.ELU()
        self.reduce = nn.Conv2d(4 * outChannel, outChannel, kernel_size=1)
    def forward(self,x):
        # return self.activ(self.batch(self.conv(x)))
        # return self.seq1(x)+self.seq2(x)+self.seq3(x)+self.seq4(x)
        out1 = self.seq1(x)
        out2 = self.seq2(x)
        out3 = self.seq3(x)
        out4 = self.seq4(x)
        # Concatenate along the channel dimension (dim=1)
        concatenated = torch.cat([out1, out2, out3,out4], dim=1)
        reduced = self.reduce(concatenated)
        return reduced
class Encoder(nn.Module):
    def __init__(self, embedding_dim):
        super(Encoder, self).__init__()
        # Conv2d layers remain similar, but the output will be a feature map
        self.conv1 = EncoderPart(1, 16)  # Input: (1, 513, 51)
        self.conv2 = EncoderPart(16, 32)
        self.conv3 = EncoderPart(32, 64, 3, 1)
        self.conv4 = EncoderPart(64, 128, 3, 1)
        self.conv5 = EncoderPart(128, embedding_dim, 3, 1)  # Output channels = embedding_dim

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        z_e = self.conv5(x)  # Output shape: (batch_size, embedding_dim, H, W)
        return z_e
    
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings  # Size of the codebook
        self.embedding_dim = embedding_dim    # Dimension of each embedding
        self.commitment_cost = commitment_cost  # Weight for commitment loss

        # Initialize the codebook
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z_e):
        # z_e shape: (batch_size, embedding_dim, H, W)
        # Flatten spatial dimensions for quantization
        z_e_flattened = z_e.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)

        # Compute distances to all embeddings
        distances = (torch.sum(z_e_flattened ** 2, dim=1, keepdim=True) +
                     torch.sum(self.embeddings.weight ** 2, dim=1) -
                     2 * torch.matmul(z_e_flattened, self.embeddings.weight.t()))

        # Find the nearest embedding indices
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=z_e.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize by mapping to the nearest embedding
        z_q = torch.matmul(encodings, self.embeddings.weight).view(z_e.shape)
        # Compute the VQ loss
        vq_loss = torch.mean((z_q.detach() - z_e) ** 2)  # Codebook loss
        commitment_loss = self.commitment_cost * torch.mean((z_q - z_e.detach()) ** 2)
        total_vq_loss = vq_loss + commitment_loss

        # Straight-through estimator for gradients
        z_q = z_e + (z_q - z_e).detach()

        return z_q, total_vq_loss
class Decoder(nn.Module):
    def __init__(self, embedding_dim):
        super(Decoder, self).__init__()
        # Input is (embedding_dim, 17, 2)
        self.deconv1 = DecoderPart(embedding_dim, 128, (0, 1))
        self.deconv2 = DecoderPart(128, 64)
        self.deconv3 = DecoderPart(64, 32)
        self.deconv4 = DecoderPart(32, 16, (0, 1))
        self.deconv5 = DecoderPart(16, 1, 0)  # Output: (1, 513, 51)

    def forward(self, z_q):
        x = self.deconv1(z_q)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        return x
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
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(embedding_dim)
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss = self.quantizer(z_e)
        recon_x = self.decoder(z_q)
        return recon_x,z_e,z_q ,vq_loss
    
def vqvae_loss(recon_x, x, vq_loss):
    recon_loss = F.mse_loss(recon_x, x)
    total_loss = recon_loss + vq_loss
    return total_loss

num_embeddings = 256  # Size of the codebook
embedding_dim = 64    # Dimension of each embedding
commitment_cost = 0.25  # Weight for commitment loss
lr = 1e-3
epochNumber = 100

# Initialize model and optimizer
vqvae = VQVAE(num_embeddings, embedding_dim, commitment_cost).cuda()
total_params = sum(p.numel() for p in vqvae.parameters())
print(f"Total parameters: {total_params}")

# Count trainable parameters
trainable_params = sum(p.numel() for p in vqvae.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params}")

# optimizer = optim.Adam(vqvae.parameters(), lr=lr)

# # Dataset and DataLoader (unchanged)
# nowTime = datetime.now().strftime("%m-%d-%Y--%H-%M_")
# writer = SummaryWriter(log_dir='./runs/' + nowTime + '2')
# dataset = CustomDataLoader(
#     rootDirs=['E:\\Work\\University\\PR\\datas\\voice_gender_detection\\females',
#               'E:\\Work\\University\\PR\\datas\\voice_gender_detection\\males'],
#     sr=16000, duration=0.5, HaveSaveOutput=False, isNormalized=True, randomSelection=False
# )
# trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
# trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
# valloader = DataLoader(valset, batch_size=32, shuffle=False)

# # Training loop
# for epoch in range(epochNumber):
#     vqvae.train()
#     loss_training = []
#     for batch, [spec, _] in enumerate(trainloader):
#         spec = spec.cuda()  # Shape: (batch_size, 1, 513, 51)
#         optimizer.zero_grad()
#         recon_x,_,_, vq_loss = vqvae(spec)
#         loss = vqvae_loss(recon_x, spec, vq_loss)
#         loss.backward()
#         loss_training.append(loss.item())
#         optimizer.step()
#     print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, np.mean(loss_training)))

#     # Validation
#     vqvae.eval()
#     testLoss = []
#     val_latents = []
#     val_labels = []
#     with torch.no_grad():
#         for batch, [spec, label] in enumerate(valloader):
#             spec = spec.cuda()
#             recon_x, _,z_q,vq_loss = vqvae(spec)
#             loss = vqvae_loss(recon_x, spec, vq_loss)
#             z_q_flat = z_q.view(z_q.size(0), -1)
#             val_latents.append(z_q_flat.cpu().numpy())
#             val_labels.append(label.numpy())
#             testLoss.append(loss.item())
#             if batch == 0:
#                 plotit(spec[0], epoch, "Real", writer)
#                 plotit(recon_x[0], epoch, "Predict", writer)
#     # In the validation loop, after plot_codebook_tsne
#     val_latents = np.concatenate(val_latents, axis=0)  # Shape: (num_val_samples, 2176)
#     val_labels = np.concatenate(val_labels, axis=0)
#     claculateSVC(val_latents,val_labels,epoch,writer,'Val')
#     plot_latent_space(val_latents,val_labels,epoch,writer,'Val')
#     writer.add_scalar("Test Loss", np.mean(testLoss), epoch)
#     writer.add_scalar("Training Loss", np.mean(loss_training), epoch)


# torch.save(vqvae.state_dict(), nowTime + '.pth')
# writer.close()