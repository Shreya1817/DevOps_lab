# Import necessary modules
from flask import Flask, render_template, request
import librosa
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the pre-trained SVM model, scaler, and LabelEncoder
model = joblib.load('model/svm_model.pkl')
scaler = joblib.load('model/scaler.pkl')  # Load the scaler
label_encoder = joblib.load('model/label_encoder.pkl')  # Load the LabelEncoder

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        file = request.files['mantra_file']
        if file:
            file_path = 'uploads/' + file.filename
            file.save(file_path)
            y, sr = librosa.load(file_path, sr=None)
            features = extract_features(y, sr)

            # Scale the features using the scaler
            features_scaled = scaler.transform(features.reshape(1, -1))

            # Predict the encoded class using the SVM model
            prediction_encoded = model.predict(features_scaled)
            
            # Decode the encoded class to the actual label
            prediction_label = label_encoder.inverse_transform(prediction_encoded)

            # Render the result template with the decoded label
            return render_template('result.html', prediction=prediction_label[0])
    return render_template('classify.html')

def extract_features(audio, sr):
    # Extracting MFCC Mean
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc)

    # Extracting Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spectral_centroid_mean = np.mean(spectral_centroid)

    # Extracting Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    spectral_bandwidth_mean = np.mean(spectral_bandwidth)

    # Extracting Zero Crossing Rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)
    zero_crossing_rate_mean = np.mean(zero_crossing_rate)

    # Extracting RMSE
    rmse = librosa.feature.rms(y=audio)
    rmse_mean = np.mean(rmse)

    # Combine all features into a single array
    return np.array([
        mfcc_mean,
        spectral_centroid_mean,
        spectral_bandwidth_mean,
        zero_crossing_rate_mean,
        rmse_mean
    ])

if __name__ == "__main__":
    app.run(debug=True)
