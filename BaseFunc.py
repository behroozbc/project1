import io
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
def plotit(spectrogram,epoch:int,kind:str,writer:SummaryWriter):
    
    spectrogram_np = spectrogram.squeeze(0).detach().cpu().numpy()
    compute_and_inverse_fourier_with_audio(spectrogram,16000,writer,epoch,kind)
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
    buf = io.BytesIO()
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
def compute_and_inverse_fourier_with_audio(spectrogram, sample_rate, writer:SummaryWriter, epoch, tag="Inverse_Fourier"):
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
        n_fft=400,
        hop_length=200,
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
def test(model,epoch:int, test_loader, criterion,writer:SummaryWriter, set="Test"):
    model.eval()
    test_loss = []
    latent_vectors = []
    labels = []
    with torch.no_grad():
        for batch_idx, [_,spec,_,label] in enumerate(test_loader):
            spec= spec.cuda()
            output = model(spec)
            if batch_idx == 0:
                plotit(spec[0],epoch=epoch,kind="Real",writer=writer)
                plotit(output[0],epoch=epoch,kind="Reconstructed",writer=writer)
            test_loss.append( criterion(output, spec).item()) 
            latent_vector = model.encoder(spec)
            latent_vectors.append(latent_vector)
            labels+=label
    latent_vectors = torch.cat(latent_vectors, dim=0)
    
    plotResults(epoch, latent_vectors, labels,writer)
    writer.add_scalar("Test Loss",np.mean(test_loss),epoch)