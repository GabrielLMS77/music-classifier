from pathlib import Path
import librosa
import numpy as np
import matplotlib.pyplot as plt


def extract_features(audio_path):
    audio, sample_rate = librosa.load(audio_path)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=13
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    features =  np.concatenate([mfcc_mean, mfcc_std])

    return features

data_path = Path("data/genres_original")

audio_files = list(data_path.rglob("*.wav"))

print("Number of audio files:", len(audio_files))


