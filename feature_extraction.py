import librosa
import numpy as np

def extract_features(file_path):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=None)
    
    # 1. MFCC (Mean of MFCC coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extract 13 MFCC coefficients
    mfcc_mean = np.mean(mfcc, axis=1)  # Mean of the MFCCs
    
    # 2. Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = np.mean(spectral_centroid)  # Mean of Spectral Centroid
    
    # 3. Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_bandwidth_mean = np.mean(spectral_bandwidth)  # Mean of Spectral Bandwidth
    
    # 4. Zero Crossing Rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
    zero_crossing_rate_mean = np.mean(zero_crossing_rate)  # Mean of Zero Crossing Rate
    
    # 5. Root Mean Squared Error (RMSE)
    rmse = librosa.feature.rms(y=y)
    rmse_mean = np.mean(rmse)  # Mean of RMSE
    
    # Combine all features into a single array
    features = np.array([
        mfcc_mean[0],   # MFCC Mean
        spectral_centroid_mean,  # Spectral Centroid Mean
        spectral_bandwidth_mean,  # Spectral Bandwidth Mean
        zero_crossing_rate_mean,  # Zero Crossing Rate Mean
        rmse_mean  # RMSE Mean
    ])
    
    return features
