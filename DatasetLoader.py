import torch
import torchaudio
from torch.utils.data import Dataset
import os
import hashlib
import pickle
class CustomDataLoader(Dataset):
    def __init__(self, rootDirs, sr=8192, duration=1,savingPath="./data-storage",isNormalized=False, HaveSaveOutput=True, randomSelection=True):
        """
        Initialize the CustomDataLoader with directories and parameters.
        
        Args:
            rootDirs (list): List of directory paths containing audio files.
            sr (int): Target sample rate for audio.
            duration (float): Duration of audio segments in seconds (default: 1).
            saveOutput (bool): Flag for saving output (not used, default: True).
            randomSelection (bool): Whether to select random segments (default: True).
        """
        self.files = []
        # Iterate over each directory in rootDirs
        for rootdir in rootDirs:
            # Extract label from the last part of the directory path
            label = os.path.basename(rootdir)
            # Add all .wav files in the directory with their labels
            for file in os.listdir(rootdir):
                if file.endswith('.wav'):  # Assuming audio files are .wav
                    fullpath = os.path.join(rootdir, file)
                    self.files.append([fullpath, label])
        self.savingPath=savingPath
        self.sr = sr
        self.duration = duration
        self.HaveSaveOutput = HaveSaveOutput  # Not used in this implementation
        self.randomSelection = randomSelection
        # Create a mapping from string labels to integer IDs
        unique_labels = set(file[1] for file in self.files)
        self.label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
        # MFCC configuration (assuming n=13 means n_mfcc=13; adjust other params as needed)
        self.allmfccConfig = {'n_mfcc': 13}  # Add more parameters like 'n_mels' if specified
        self.isNormalized=isNormalized
        # Create savingPath directory if it doesn't exist and saving is enabled
        if self.HaveSaveOutput:
            os.makedirs(self.savingPath, exist_ok=True)

    def __len__(self):
        """
        Return the total number of audio files.
        
        Returns:
            int: Length of the files list.
        """
        return len(self.files)

    def __getitem__(self, id):
        """
        Retrieve and process an audio item by index.
        
        Args:
            id (int): Index of the item to retrieve.
        
        Returns:
            tuple: (waveform, spectrogram, MFCC, integer label)
        """
        # Get file path and label from the files list
        fullpath, label = self.files[id]
        # Convert string label to integer ID
        intId = self.label_to_int[label]
        
        # Load audio file
        wave, samp_rate = torchaudio.load(fullpath)
        
        # Resample if sample rate doesn't match target sr
        if samp_rate != self.sr:
            wave = torchaudio.transforms.Resample(orig_freq=samp_rate, new_freq=self.sr)(wave)
        
        # Extract audio segment based on randomSelection flag
        if not self.randomSelection:
            wave,start,end = self._get_duration(wave, self.sr, self.duration)
        else:
            wave,start,end = self._get_random_part(wave, self.sr, self.duration)
        config = {**self.allmfccConfig, 'start': start, 'end': end, 'sr': self.sr}
        filename = os.path.basename(fullpath)
        outputfilename = self._genratemd5hash('mfcc', config, filename)
        # Compute MFCC with 13 coefficients
        # Try to load precomputed MFCC if it exists
        outputfilename = self._genratemd5hash('mfcc', config, filename)
        mfcc = self._loadfileIfexist(outputfilename)
        
        # Compute MFCC if not loaded from disk
        if mfcc is None:
            mfcc_transform = torchaudio.transforms.MFCC(sample_rate=self.sr, n_mfcc=13)
            mfcc = mfcc_transform(wave)
            # Save MFCC to disk if enabled
            if self.HaveSaveOutput:
                self._saveOutput(outputfilename, mfcc)
        
        
        # Compute and normalize spectrogram, then slice to 82x82
        spec = torchaudio.transforms.Spectrogram(n_fft=200, hop_length=100)(wave)[:,:100,:4]
        # print(spec.shape)
        if(self.isNormalized):
            spec = self._normalize_spectrogram(spec)
        # spec = self._normalize_spectrogram(spec)
        # spec = spec[:, :, :164]  # Take first 82 frequency bins and time frames
        
        return wave, spec, mfcc, intId

    def _get_duration(self, wave, sr, duration):
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
            return wave[:, :num_frames],0,num_frames
        return wave,0, wave.size(1) # Return full wave if shorter than duration

    def _get_random_part(self, wave, sr, duration):
        """
        Extract a random segment of the specified duration from the waveform.
        
        Args:
            wave (Tensor): Input waveform tensor.
            sr (int): Sample rate.
            duration (float): Desired duration in seconds.
        
        Returns:
            Tensor: Waveform segment.
        """
        num_frames = int(duration * sr)
        if wave.size(1) <= num_frames:
            return wave
        start = torch.randint(0, wave.size(1) - num_frames + 1, (1,)).item()
        return wave[:, start:start + num_frames],start,num_frames

    def _normalize_spectrogram(self, spec):
        """
        Normalize the spectrogram to [0, 1] range.
        
        Args:
            spec (Tensor): Input spectrogram tensor.
        
        Returns:
            Tensor: Normalized spectrogram.
        """
        min_val = spec.min()
        max_val = spec.max()
        # Avoid division by zero
        spec = (spec - min_val) / (max_val - min_val + 1e-6)
        return spec
    def _genratemd5hash(self, name, config, filename):
        """
        Generate an MD5 hash from name, config, and filename.

        Args:
            name (str): Identifier (e.g., 'mfcc').
            config (dict): Configuration dictionary.
            filename (str): Base filename.

        Returns:
            str: MD5 hash string.
        """
        config_str = str(sorted(config.items()))  # Ensure consistent string representation
        hash_input = name + config_str + filename
        return hashlib.md5(hash_input.encode()).hexdigest()

    def _loadfileIfexist(self, outputfilename):
        """
        Load a file from disk if it exists.

        Args:
            outputfilename (str): Filename without extension.

        Returns:
            object: Loaded object if file exists, None otherwise.
        """
        filepath = os.path.join(self.savingPath, outputfilename + '.pkl')
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        return None

    def _saveOutput(self, outputfilename, file):
        """
        Save an object to disk.

        Args:
            filename (str): Original filename.
            name (str): Identifier (e.g., 'mfcc').
            config (dict): Configuration dictionary.
            file (object): Object to save (e.g., MFCC tensor).
        """
        # outputfilename = self._genratemd5hash(name, config, filename)
        filepath = os.path.join(self.savingPath, outputfilename + '.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(file, f)  # Synchronous save (background saving not implemented)