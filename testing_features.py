from pathlib import Path
import librosa
import numpy as np
import matplotlib.pyplot as plt



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