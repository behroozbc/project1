# import librosa
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
import os
# from sklearn.decomposition import PCA
# # Function to extract mean MFCC features from an audio file
# def extract_mfcc_mean(file_path, n_mfcc=13):
#     """
#     Extract MFCC features from an audio file and compute their mean over time.
    
#     Parameters:
#     - file_path (str): Path to the audio file.
#     - n_mfcc (int): Number of MFCC coefficients to extract (default is 13).
    
#     Returns:
#     - np.ndarray: Mean MFCC feature vector of size n_mfcc.
#     """
#     # Load audio file
#     y, sr = librosa.load(file_path,sr=8192)
    
#     y = y - np.mean(y) # make the sound balance and fix if not zero centered.
#     y = y / np.max(np.absolute(y)) # Normalize the signal, because help use to avoid distortion during playback and condistenty across diffrent audio files.
#     # Extract MFCC features
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
#     # Compute mean over time (axis=1)
#     mfcc_mean = np.mean(mfcc, axis=1)
    
#     return [mfcc_mean,np.std(mfcc, axis=1)]

# # Function to load data from multiple audio files
# def load_data(file_paths, labels, n_mfcc=13):
#     """
#     Load audio files, extract MFCC features, and pair them with labels.
    
#     Parameters:
#     - file_paths (list): List of paths to audio files.
#     - labels (list): List of corresponding gender labels (e.g., 0 for male, 1 for female).
#     - n_mfcc (int): Number of MFCC coefficients.
    
#     Returns:
#     - X (np.ndarray): Feature matrix of shape (n_samples, n_mfcc).
#     - y (np.ndarray): Label array of shape (n_samples,).
#     """
#     X = []
#     y = []
#     for file_path, label in zip(file_paths, labels):
#         mfcc_mean = extract_mfcc_mean(file_path, n_mfcc)
#         X.append(mfcc_mean)
#         y.append(label)
#     return np.array(X), np.array(y)

# # Main execution
# if __name__ == "__main__":
#     file_paths = []
#     labels = []
#     for rootdir in ['E:\\Work\\University\\PR\\datas\\voice_gender_detection\\females','E:\\Work\\University\\PR\\datas\\voice_gender_detection\\males']:
#             # Extract label from the last part of the directory path
#             label = os.path.basename(rootdir)
#             # Add all .wav files in the directory with their labels
#             for file in os.listdir(rootdir):
#                 if file.endswith('.wav'):  # Assuming audio files are .wav
#                     fullpath = os.path.join(rootdir, file)
#                     file_paths.append(fullpath)
#                     labels.append(label=='males')
#     # Example file paths and labels (replace with your actual data)
    
#       # 0 for male, 1 for female

#     # Step 1 & 2: Load data and extract MFCC features
#     print("Loading data and extracting MFCC features...")
#     X, y = load_data(file_paths, labels, n_mfcc=13)

#     # Step 3: Split data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     # Step 4: Scale the features
#     # scaler = StandardScaler()
#     # X_train_scaled = scaler.fit_transform(X_train)
#     # X_test_scaled = scaler.transform(X_test)

#     # Step 5: Train the SVM classifier
#     print("Training SVM classifier...")
#     # pca = PCA(n_components=2)
#     # X_train_scaled = pca.fit_transform(X_train)
#     # X_test_scaled = pca.transform(X_test)
#     clf = SVC(kernel='poly', degree=3)  
#     clf.fit(X_train, y_train)

#     # Step 6: Predict and evaluate
#     y_pred = clf.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"Accuracy: {accuracy:.2f}")

import librosa
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,f1_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, precision_score, classification_report
# Function to extract MFCC features and compute statistics
def extract_features(file_path, sr=16000, n_mfcc=13, n_mels=40):
    # Step 1: Load the WAV file with a fixed sampling rate
    y, sr = librosa.load(file_path, sr=sr, mono=True)
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

# Example dataset (replace with your actual data)
# Assume file_paths is a list of paths to WAV files, labels is a list of gender labels (e.g., 0 for male, 1 for female)
file_paths = []  # Replace with your file paths
labels = []  # Replace with your labels

# Extract features for all audio files
X = []
for rootdir in ['E:\\Work\\University\\PR\\datas\\voice_gender_detection\\females','E:\\Work\\University\\PR\\datas\\voice_gender_detection\\males']:
    label = os.path.basename(rootdir)
    for file in os.listdir(rootdir):
        if file.endswith('.wav'):  # Assuming audio files are .wav
            fullpath = os.path.join(rootdir, file)
            file_paths.append(fullpath)
            labels.append(label=='males')
for file_path in file_paths:
    features = extract_features(file_path)
    X.append(features)
X = np.array(X)  # Shape: (n_samples, 4 * n_mfcc)
y = np.array(labels)  # Shape: (n_samples,)

# Step 7: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train, X_test,y_train,y_test=train_test_split(X_test,y_test,test_size=0.2)
# Standardize features (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 8: Train SVM classifier
svm = SVC(kernel='rbf')  # Linear kernel; you can try 'rbf' too
svm.fit(X_train_scaled, y_train)

# Predict on test set
y_pred = svm.predict(X_test_scaled)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(f"F1: {f1:.2f}")
print(f"recall: {recall:.2f}")
print(f"precision: {precision:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print(classification_report(y_test, y_pred, target_names=['Female', 'Male']))