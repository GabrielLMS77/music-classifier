from pathlib import Path
import librosa


audio_path = Path("data/genres_original/blues/blues.00000.wav")

print(audio_path.exists())


audio, sample_rate = librosa.load(audio_path)

print("Sample rate:", sample_rate)
print("Number of samples:", len(audio))




print(audio[:10])
print(type(audio))