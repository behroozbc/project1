import torchaudio
import torch
import matplotlib.pyplot as plt
def _normalize_spectrogram(spec):
        min_val = spec.min()
        max_val = spec.max()
        # Avoid division by zero
        spec = (spec - min_val) / (max_val - min_val + 1e-6)
        return spec
windowSize = 0.025 
    
hop_length = 0.01
N=256
duration=0.5
fullpath="E:\\Work\\University\\PR\\datas\\voice_gender_detection\\females\\3.wav"
wave, samp_rate = torchaudio.load(fullpath)
num_frames = int(duration * samp_rate)
n_fft = 512  # Frequency resolution
hop_length = int(hop_length*samp_rate)  # 10 ms hop at 16 kHz
win_length = int(windowSize * samp_rate)  # Window length matches n_fft
window = torch.hann_window(win_length)  # Hann window
wave=wave[:,:num_frames]
spec= torchaudio.transforms.Spectrogram(n_fft=n_fft,win_length=win_length,hop_length=hop_length)(wave)
log_power_spectrogram = 10 * torch.log10(spec + 1e-10)  # Log scale, avoid log(0)
def plotit(spectrogram):
    spectrogram_np = spectrogram.squeeze(0).detach().cpu().numpy()
    fig=plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram_np, cmap='viridis', origin='lower', aspect='auto')
    plt.colorbar(label='Decibels (dB)')
    plt.xlabel('Time Frames')
    plt.ylabel('Frequency Bins')
    plt.title('Spectrogram')
    plt.show()
    plt.close(fig)
print(log_power_spectrogram.shape)
plotit(log_power_spectrogram)