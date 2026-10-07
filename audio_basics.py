from pathlib import Path
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

audio_path = Path("data/genres_original/blues/blues.00000.wav")

print(audio_path.exists())

audio, sample_rate = librosa.load(audio_path)

print("Sample rate:", sample_rate)
print("Number of samples", len(audio))

print(audio[:10])
print(type(audio))


spectrum = np.fft.rfft(audio)

frequencies = np.fft.rfftfreq(
    len(audio),
    1 / sample_rate
)

magnitude = np.abs(spectrum)

plt.figure()

plt.plot(frequencies, magnitude)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Frequency Spectrum")

plt.show()

stft = librosa.stft(audio)

print("STFT shape:", stft.shape)


frequency_bins = stft.shape[0]
time_frames = stft.shape[1]

print("Frequency bins:", frequency_bins)
print("Time frames:", time_frames)

plt.figure(figsize=(10, 5))

librosa.display.specshow(
    np.abs(stft),
    sr=sample_rate,
    x_axis="time",
    y_axis="hz"
)

plt.colorbar()

plt.title("Spectrogram")

plt.show()

