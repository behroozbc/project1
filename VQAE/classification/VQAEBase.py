import io
from sklearn.manifold import TSNE
import torch
import PIL.Image
import torchaudio
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn import metrics 
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
def plot_codebook_usage(model, dataloader, writer, epoch, num_embeddings):
    model.eval()
    embedding_counts = torch.zeros(num_embeddings, device='cuda')
    
    with torch.no_grad():
        for spec, _ in dataloader:
            spec = spec.cuda()
            z_e = model.encoder(spec)  # Shape: (batch_size, embedding_dim, H, W)
            z_e_flattened = z_e.permute(0, 2, 3, 1).contiguous().view(-1, model.quantizer.embedding_dim)
            
            # Compute distances and find nearest embedding indices
            distances = (torch.sum(z_e_flattened ** 2, dim=1, keepdim=True) +
                         torch.sum(model.quantizer.embeddings.weight ** 2, dim=1) -
                         2 * torch.matmul(z_e_flattened, model.quantizer.embeddings.weight.t()))
            encoding_indices = torch.argmin(distances, dim=1)
            
            # Count occurrences of each index
            counts = torch.bincount(encoding_indices, minlength=num_embeddings)
            embedding_counts += counts
    
    # Convert to numpy for plotting
    embedding_counts = embedding_counts.cpu().numpy()
    
    # Plot histogram
    plt.figure(figsize=(10, 4))
    plt.bar(range(num_embeddings), embedding_counts)
    plt.title(f"Codebook Embedding Usage (Epoch {epoch})")
    plt.xlabel("Embedding Index")
    plt.ylabel("Usage Count")
    
    # Log to TensorBoard
    writer.add_figure("Codebook/Usage", plt.gcf(), global_step=epoch)
    plt.close()
    
def plot_codebook_tsne(model, writer, epoch):
    # Extract codebook
    codebook = get_codebook(model)  # Shape: (num_embeddings, embedding_dim)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    codebook_2d = tsne.fit_transform(codebook)  # Shape: (num_embeddings, 2)
    
    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(codebook_2d[:, 0], codebook_2d[:, 1], s=10, c='blue', alpha=0.6)
    plt.title(f"Codebook t-SNE Projection (Epoch {epoch})")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    
    # Log to TensorBoard
    writer.add_figure("Codebook/t-SNE", plt.gcf(), global_step=epoch)
    plt.close()
def get_codebook(model):
    # model: VQVAE instance
    codebook = model.quantizer.embeddings.weight.detach().cpu().numpy()  # Shape: (num_embeddings, embedding_dim)
    return codebook

