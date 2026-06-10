import io
import librosa
from sklearn.manifold import TSNE
import torch
import PIL.Image
import torchaudio
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn import metrics 
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
import wfdb
from scipy.io import wavfile
from scipy.stats import skew, kurtosis
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import os
def plotit(spectrogram,epoch:int,kind:str,writer:SummaryWriter,n_fft=512,hopSize=160):
    
    spectrogram_np = spectrogram.squeeze(0).detach().cpu().numpy()
    #compute_and_inverse_fourier_with_audio(spectrogram,16000,n_fft,hopSize,writer,epoch,kind)
    fig=plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram_np, cmap='viridis', origin='lower', aspect='auto')
    plt.colorbar(label='Decibels (dB)')
    plt.xlabel('Time Frames')
    plt.ylabel('Frequency Bins')
    plt.title('Spectrogram')
    writer.add_figure('Spectrogram '+kind,fig,epoch)
    plt.close(fig)
def plotResults(epoch:int, latent_vectors, labels,writer:SummaryWriter):
    labels=list(map(lambda x: "#0f0" if x==1 else "#f00", labels))
    svns= [svm.SVC(kernel='poly',degree=3),svm.SVC(kernel='rbf')]
    pca = PCA(n_components=2)
    
    latent_np = latent_vectors.cpu().numpy()
    latent_pca = pca.fit_transform(latent_np[:,:])
    X_train,X_test,Y_train,Y_test=train_test_split(latent_np,labels,test_size=0.2,random_state=30)
    for svn in svns:
        svn.fit(X_train,Y_train)
        message=svn.kernel+" Accuracy"
        writer.add_scalar(message,svn.score(X_test,Y_test),epoch)
    fig=plt.figure(figsize=(12, 5))
    scatter = plt.scatter(latent_pca[:, 0], latent_pca[:, 1], c=labels, cmap='tab10', alpha=0.6)
    
    fig.colorbar(scatter)
    plt.title("PCA of Latent Space")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    writer.add_figure('PCA',fig,epoch)
    plt.close(fig)
def compute_and_inverse_fourier_with_audio(spectrogram, sample_rate,n_fft,hopSize, writer:SummaryWriter, epoch, tag="Inverse_Fourier"):
    """
    Compute inverse Fourier transform, log visualization and audio to TensorBoard
    
    Args:
        tensor_data: Input tensor in frequency domain
        sample_rate: Audio sample rate in Hz (e.g., 44100 for CD quality)
        writer: TensorBoard SummaryWriter object
        epoch: Current epoch/step number
        tag: Name for the TensorBoard log
    """
    # Ensure tensor is on CPU and in proper format
    spectrogram=spectrogram.cpu()

    if torch.min(spectrogram) < 0:  # Assuming dB scale if negative values present
        spectrogram = torchaudio.functional.DB_to_amplitude(spectrogram, ref=1.0, power=0.5)
    
    # Reconstruct audio using Griffin-Lim
    inverse_transform = torchaudio.transforms.GriffinLim(
        n_fft=n_fft,
        hop_length=hopSize,
    )
    waveform = inverse_transform(spectrogram)

    audio_data = waveform / torch.max(torch.abs(waveform))
    # Log audio to TensorBoard
    # Audio needs to be in range [-1, 1] and 1D
    writer.add_audio(f"{tag}_audio", audio_data, epoch, sample_rate=sample_rate)
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
def test(model,epoch:int, test_loader, criterion,n_fft,hopSize,writer:SummaryWriter, set="Test"):
    model.eval()
    test_loss = []
    latent_vectors = []
    labels = []
    with torch.no_grad():
        for batch_idx, [_,spec,_,label] in enumerate(test_loader):
            spec= spec.cuda()
            output = model(spec)
            if batch_idx == 0:
                plotit(spec[0],epoch=epoch,n_fft=n_fft,hopSize=hopSize,kind="Real",writer=writer)
                plotit(output[0],epoch=epoch,n_fft=n_fft,hopSize=hopSize,kind="Reconstructed",writer=writer)
            test_loss.append( criterion(output, spec).item()) 
            latent_vector = model.encoder(spec)
            latent_vectors.append(latent_vector)
            labels+=label
    latent_vectors = torch.cat(latent_vectors, dim=0)
    
    plotResults(epoch, latent_vectors, labels,writer)
    writer.add_scalar("Test Loss",np.mean(test_loss),epoch)
def getSummaryWriter(perfix: str) -> tuple[SummaryWriter, str]:
    nowTime=datetime.now().strftime("%m-%d-%Y--%H-%M_")
    return SummaryWriter(log_dir="E:\\Work\\University\\PR\\project1\\runs\\"+nowTime+perfix),nowTime

def plot_latent_spaceVQA(model, test_loader,writer:SummaryWriter, epoch:int,device='cuda' if torch.cuda.is_available() else 'cpu', plot_codebook=True):
    """
    Plot the latent space of the VQ Autoencoder using t-SNE.

    Args:
        model: Trained VQAutoencoder model.
        dataset: PyTorch Dataset with input data of shape [batch_size, 1, 257, 50].
        num_samples: Number of samples to visualize (default: 1000).
        device: Device to run inference on ('cuda' or 'cpu').
        plot_codebook: Whether to plot the codebook embeddings (default: True).
    """
    
    latent_vectors = []
    indices_list = []
    
    with torch.no_grad():
        for batch_idx, [_,spec,_,label] in enumerate(test_loader):
            inputs = spec.to(device)
            _, z_e, z_q, indices = model(inputs)
            # Flatten z_q: [batch_size, 64, 32, 6] -> [batch_size, 64*32*6]
            z_q_flat = z_q.permute(0, 2, 3, 1).reshape(z_q.size(0), -1)
            latent_vectors.append(z_q_flat.cpu().numpy())
            indices_list.append(indices.cpu().numpy())
    
    # Concatenate all latent vectors and indices
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    indices_list = np.concatenate(indices_list, axis=0)
    
    # Apply t-SNE to reduce dimensionality to 2D
    tsne = TSNE(n_components=2, perplexity=30)
    latent_2d = tsne.fit_transform(latent_vectors)
    
    # Plot latent space
    fig= plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=indices_list, cmap='viridis', s=50, alpha=0.6)
    plt.colorbar(scatter, label='Codebook Index')
    plt.title('Latent Space Visualization (t-SNE)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    # Optionally plot codebook embeddings
    if plot_codebook:
        codebook = model.codebook.weight.cpu().detach().numpy()  # Shape: [K, D]
        codebook_2d = tsne.fit_transform(codebook)
        plt.figure(figsize=(10, 8))
        plt.scatter(codebook_2d[:, 0], codebook_2d[:, 1], c='red', marker='x', s=100, label='Codebook Embeddings')
        plt.title('Codebook Embeddings Visualization (t-SNE)')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend()
    
    writer.add_figure('PCA',fig,epoch)
    
def plot_latent_spaceVAE(vae, dataloader,epoch:int,writer:SummaryWriter, kind='val'):
    """
    Plots the 2D projection of the latent space using t-SNE.
    
    Args:
        vae: Trained VAE model
        dataloader: DataLoader with the dataset
        device: Device to run the model on (e.g., 'cuda' or 'cpu')
    """
    vae.eval()  # Set to evaluation mode
    means = []
    labels=[]
    with torch.no_grad():
        for batch,[spec,label] in enumerate(dataloader):
            x = spec.to("cuda")  # Assuming batch is (data, label) or just data
            _ , mean, _ = vae(x)
            means.append(mean.cpu().numpy())
            labels+=label
    # Concatenate all mean vectors
    means = np.concatenate(means, axis=0)  # Shape: (n_samples, 512)
    claculateSVC(means,labels,epoch,writer,kind)
    # Reduce to 2D using t-SNE
    tns_means_2d = TSNE(n_components=2, random_state=42).fit_transform(means)
    pca_means_2d = PCA(n_components=2).fit_transform(means)
    labels=list(map(lambda x: "#0f0" if x==1 else "#f00", labels))
    # Plot
    fig=plt.figure(figsize=(8, 6))
    plt.scatter(tns_means_2d[:, 0], tns_means_2d[:, 1],c=labels, s=5)
    plt.title("Latent Space Visualization")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    writer.add_figure('t-SNE-'+kind,fig,epoch)
    plt.close(fig)
    fig=plt.figure(figsize=(8, 6))
    plt.scatter(pca_means_2d[:, 0], pca_means_2d[:, 1],c=labels, s=5)
    plt.title("Latent Space Visualization")
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    writer.add_figure('PCA-'+kind,fig,epoch)
    plt.close(fig)
def claculateSVC(means,labels,epoch:int,writer:SummaryWriter,kind:str):
    X_train,X_test,Y_train,Y_test=train_test_split(means,labels,test_size=0.2,random_state=30)
    svns= [svm.SVC(kernel='poly',degree=3),svm.SVC(kernel='rbf')]
    for svn in svns:
        svn.fit(X_train,Y_train)

        y_perdict=svn.predict(X_test)
        writer.add_scalar(svn.kernel+" Accuracy"+kind,metrics.accuracy_score(Y_test,y_perdict),epoch)
        writer.add_scalar(svn.kernel+" F1"+kind,metrics.f1_score(Y_test,y_perdict),epoch)
        cm= metrics.confusion_matrix(Y_test,y_perdict)
        tn, fp, fn, tp=cm.ravel()
        writer.add_scalar(svn.kernel+"tn"+kind,tn,epoch)
        writer.add_scalar(svn.kernel+"fp"+kind,fp,epoch)
        writer.add_scalar(svn.kernel+"fn"+kind,fn,epoch)
        writer.add_scalar(svn.kernel+"tp"+kind,tp,epoch)
def plot_latent_spaceVQAE(model, dataloader, epoch, writer, num_samples=4):
    """
    Plot the latent space of VQAE as heatmaps of codebook indices.
    
    Args:
        model: VQAE model
        dataloader: Validation DataLoader
        epoch: Current epoch
        writer: TensorBoard SummaryWriter
        num_samples: Number of samples to visualize
    """
    model.eval()
    with torch.no_grad():
        for batch, [spec, _] in enumerate(dataloader):
            spec = spec.cuda()
            # Forward pass to get encoding indices
            recon_x, z_e, z_q = model(spec)
            _, encoding_indices = model.quantizer(z_e)  # Shape: (batch*17*2,)
            
            # Reshape indices to (batch, 17, 2)
            batch_size = spec.size(0)
            indices = encoding_indices.view(batch_size, 17, 2).cpu().numpy()
            
            # Plot heatmaps for the first `num_samples` samples
            fig, axes = plt.subplots(1, min(num_samples, batch_size), figsize=(min(num_samples, batch_size) * 4, 3))
            if num_samples == 1:
                axes = [axes]
            for i in range(min(num_samples, batch_size)):
                ax = axes[i] if num_samples > 1 else axes[0]
                # Plot heatmap of indices
                cax = ax.imshow(indices[i], cmap='viridis', interpolation='nearest', aspect='auto')
                ax.set_title(f'Sample {i+1}')
                ax.set_xlabel('Time (2 bins)')
                ax.set_ylabel('Frequency (17 bins)')
                fig.colorbar(cax, ax=ax, label='Codebook Index')
            
            plt.tight_layout()
            
            # Log to TensorBoard
            writer.add_figure(f'Latent_Space', fig, global_step=epoch)
            plt.close(fig)
def plot_codebook_tsne(model, dataloader, epoch, writer):
    model.eval()
    # Count usage of each codebook vector
    counts = torch.zeros(model.quantizer.embeddings.num_embeddings).cuda()
    with torch.no_grad():
        for spec, _ in dataloader:
            spec = spec.cuda()
            _, z_e, _ = model(spec)
            _, indices = model.quantizer(z_e)
            counts += torch.bincount(indices, minlength=model.quantizer.embeddings.num_embeddings)
    counts = counts.cpu().numpy()
    
    # t-SNE on codebook vectors
    embeddings = model.quantizer.embeddings.weight.cpu().detach().numpy()
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(6, 6))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=counts, cmap='viridis', s=50)
    plt.colorbar(label='Usage Count')
    plt.title('t-SNE of Codebook Vectors')
    writer.add_figure(f'Codebook_tSNE', plt.gcf(), global_step=epoch)
    plt.close()
def plot_latent_space(means,labels,epoch:int,writer:SummaryWriter,kind:str):
    tns_means_2d = TSNE(n_components=2).fit_transform(means)
    pca_means_2d = PCA(n_components=2).fit_transform(means)
    labels=list(map(lambda x: "#0f0" if x==1 else "#f00", labels))
    # Plot
    fig=plt.figure(figsize=(8, 6))
    plt.scatter(tns_means_2d[:, 0], tns_means_2d[:, 1],c=labels, s=5)
    plt.title("Latent Space Visualization")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    writer.add_figure('t-SNE-'+kind,fig,epoch)
    plt.close(fig)
    fig=plt.figure(figsize=(8, 6))
    plt.scatter(pca_means_2d[:, 0], pca_means_2d[:, 1],c=labels, s=5)
    plt.title("Latent Space Visualization")
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    writer.add_figure('PCA-'+kind,fig,epoch)
    plt.close(fig)
    
def read_wfdb(address:str):
    record = wfdb.rdrecord(address)
    signal = record.p_signal
    fs = record.fs
    signal_normalized = np.int16(signal / np.max(np.abs(signal)) * 32767)
    return fs,signal_normalized
def extract_features(y, sr=16000, n_mfcc=13, n_mels=40):
    # Step 2: Normalize amplitude to [-1, 1]
    y = y / np.max(np.abs(y))
    num_frames=int(0.5 * sr)
    y=y[500:num_frames]
    # Step 3 & 4: Compute STFT and transform into Mel scale
    # We'll compute MFCCs directly, which includes STFT and Mel transformation internally
    # n_fft = 25ms window, hop_length = 10ms
    windowsize = int(0.025 * sr)  # 400 samples at 16kHz
    
    hop_length = int(0.01 * sr)  # 160 samples at 16kHz
    # print(n_fft,hop_length)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels,
                                 n_fft=1024,win_length=windowsize, hop_length=hop_length)
    
    # Step 5: MFCCs already include the cosine transform step
    
    # Step 6: Compute statistics across time frames for each MFCC coefficient
    mean = np.mean(mfccs, axis=1)  # Mean over time
    std = np.std(mfccs, axis=1)    # Standard deviation over time
    skewness = skew(mfccs, axis=1) # Skewness over time
    kurt = kurtosis(mfccs, axis=1) # Kurtosis over time
    
    # Combine into a 1D feature vector (4 statistics * n_mfcc)
    features = np.concatenate([mean, std, skewness, kurt])
    return features

def get_duration( wave, sr, duration):
    """
    Extract a fixed duration from the start of the waveform.
        
    Args:
        wave (Tensor): Input waveform tensor.
        sr (int): Sample rate.
        duration (float): Desired duration in seconds.
        
    Returns:
        Tensor: Waveform segment.
    """
    num_frames = int(duration * sr)
    if wave.size(1) >= num_frames:
        return wave[:, 1000:num_frames+1000],0,num_frames
    return wave,0, wave.size(1) # Return full wave if shorter than duration

def normalize_spectrogram(spec):
        """
        
        Args:
            spec (Tensor): Input spectrogram tensor.

        Returns:
            Tensor: Normalized spectrogram.
        """
        # min_val = spec.min()
        # max_val = spec.max()
        std= spec.std()
        mean=spec.mean()
        spec = (spec - mean) / std
        # return spec
        # Avoid division by zero
        # spec = (spec - min_val) / (max_val - min_val + 1e-6)
        return spec
    
def claculateSVCRegrestin(means,outs,epoch:int,writer:SummaryWriter,kind:str,name:str):
    X_train, X_test, y_train, y_test = train_test_split(means, outs, test_size=0.3, random_state=42)
    models=[SVR(kernel="linear"),SVR(kernel="poly"),SVR(kernel="rbf"),SVR(kernel="sigmoid")]
    for i in models:
        model=i
        scores_mse = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        
        writer.add_scalar("Cross val mean squared Mean MSE/train",-scores_mse.mean(),epoch)
        writer.add_scalar("Cross val Std MSE/train",scores_mse.std(),epoch)
        writer.add_scalar("Cross val Mean RMSE/train",np.sqrt(-scores_mse.mean()),epoch)
        # محاسبه معیارها برای کراس ولیدیشن
        # print('معیارهای Cross-Validation (روی داده train):')
        # print('Mean R2:', scores_r2.mean())
        # print('Std R2:', scores_r2.std())
        # آموزش مدل روی کل داده train
        model.fit(X_train, y_train)
        
        # پیش‌بینی و ارزیابی روی test
        y_test_pred = model.predict(X_test)
         
        mse_test = mean_squared_error(y_test, y_test_pred)
        writer.add_scalar('MSE/test:', mse_test,epoch)
        writer.add_scalar('RMSE/test:', np.sqrt(mse_test),epoch)
        writer.add_scalar('MAE/test:', mean_absolute_error(y_test, y_test_pred),epoch)
        writer.add_scalar('R2/test:', r2_score(y_test, y_test_pred),epoch)
        writer.add_scalar('Explained Variance/test:', explained_variance_score(y_test, y_test_pred),epoch)

def createTensorboard(passwand:str):
    nowTime=datetime.now().strftime("%m-%d-%Y--%H-%M_")
    os.makedirs('./runs',exist_ok=True)
    writer = SummaryWriter(log_dir='./runs/'+nowTime+str(passwand))
    return writer