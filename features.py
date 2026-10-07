from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

audio_path = Path("data/genres_original/blues/blues.00000.wav")

features = extract_features(audio_path)

print("Features shape:", features.shape)
print("Features:", features)


audio_path = Path("data/genres_original/blues/blues.00000.wav")

audio, sample_rate = librosa.load(audio_path)

print("Audio shape: ", audio.shape)
print("Sample rate: ", sample_rate)

mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sample_rate,
    n_mfcc=13
)

print("MFCC Shape: ", mfcc.shape)



plt.figure(figsize=(10, 5))

librosa.display.specshow(
    mfcc,
    sr=sample_rate,
    x_axis="time"
)



plt.colorbar()
plt.title("MFCCs")
plt.ylabel("MFCC Coefficient")
plt.show()

mfcc_mean = np.mean(mfcc, axis=1)

print("MFCC mean shape: ", mfcc_mean.shape)
print("MFCC means: ", mfcc_mean)

mfcc_std = np.std(mfcc, axis=1)

print("MFCC std shape:", mfcc_std.shape)
print("MFCC std:", mfcc_std)

features = np.concatenate([mfcc_mean, mfcc_std])

print("Features shape: ", features.shape)
print("Features: ", features)

data_path = Path("data/genres_original")

audio_files = list(data_path.rglob("*.wav"))

print("Number of audio files:", len(audio_files))

X = []
y = []
failed_files = []

for audio_file in audio_files:
    try:
        features = extract_features(audio_file)
        X.append(features)

        genre = audio_file.parent.name
        y.append(genre)

    except Exception as e:
        print("Could not process:", audio_file)
        failed_files.append(audio_file)

print("Number of feature vectors:", len(X))
print("Number of failed files:", len(failed_files))
print(audio_file.parent.name)
print("Number of feature vectors:", len(X))
print("Number of labels:", len(y))
print("First 10 labels:", y[:10])

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

genre_counts = Counter(y)

print("Genre counts:")

for genre, count in genre_counts.items():
    print(genre, ":", count)


X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

np.save("X.npy", X)
np.save("y.npy", y)

print("Saved features and labels!")


model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("First 10 predictions:", predictions[:10])
print("First 10 actual:", y_test[:10])











