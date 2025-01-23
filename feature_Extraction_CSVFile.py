# -*- coding: utf-8 -*-  
"""  
Feature Extraction Script  
Created on Thu Nov 28 21:55:40 2024  

@author: Abc  
"""  

import os  
import librosa  
import pandas as pd  

# Step 1: Extract Features  
def extract_features(audio_dir, output_csv):  
    data = []  
    mantra_labels = {  
        "Om": "Improved focus",  
        "Sohum": "Peace and calm",  
        "Kohum": "Reduced stress"  
    }  
    
    for mantra, label in mantra_labels.items():  
        mantra_path = os.path.join(audio_dir, mantra)  
        if os.path.exists(mantra_path):  
            for file in os.listdir(mantra_path):  
                if file.endswith(".mp3") or file.endswith(".mpeg3"):  
                    file_path = os.path.join(mantra_path, file)  
                    try:  
                        y, sr = librosa.load(file_path, sr=None)  
                        
                        # Extract features  
                        mfcc_mean = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean()  
                        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()  
                        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()  
                        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y).mean()  
                        rmse = librosa.feature.rms(y=y).mean()  
                        
                        # Append data  
                        data.append([file, label, mfcc_mean, spectral_centroid, spectral_bandwidth, zero_crossing_rate, rmse])  
                    except Exception as e:  
                        print(f"Error processing {file}: {e}")  
    
    # Save data to CSV  
    columns = ["File Name", "Label", "MFCC_Mean", "Spectral_Centroid", "Spectral_Bandwidth", "Zero_Crossing_Rate", "RMSE"]  
    df = pd.DataFrame(data, columns=columns)  
    df.to_csv(output_csv, index=False)  
    print(f"Feature extraction complete. Dataset saved to {output_csv}")  

# Main Execution  
if __name__ == "__main__":  
    audio_dir = r"C:\Users\Abc\Desktop\all mantras audio"  # Path to mantra audio directory  
    output_csv = "mantra_dataset_improved.csv"  
    
    # Step 1: Feature Extraction  
    extract_features(audio_dir, output_csv)