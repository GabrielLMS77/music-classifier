# Music Classifier

A machine learning project for classifying music genres using audio features.

## Current Progress

- [x] Download and organize dataset
- [x] Explore dataset structure
- [x] Count songs by genre
- [x] Load audio files with Librosa
- [x] Understand sample rate and audio samples
- [x] Visualize waveform
- [x] Analyze frequency spectrum
- [x] Generate STFT spectrogram
- [x] Extract MFCC audio features
- [x] Create feature vectors
- [x] Split data into training and test sets
- [x] Train a Random Forest classifier
- [x] Evaluate initial model accuracy
- [ ] Analyze confusion matrix
- [ ] Improve the classifier
- [ ] Train a neural network
- [ ] Compare different approaches

## Technologies

- Python
- NumPy
- Librosa
- Matplotlib
- Scikit-learn

## Initial Results

The first Random Forest classifier was trained using 26 MFCC-based features
(13 MFCC means and 13 MFCC standard deviations).

The dataset contained 999 usable audio files after skipping one corrupted file.

Using an 80/20 train-test split:

- Training samples: 799
- Test samples: 200
- Features per song: 26
- Initial accuracy: 62%


## Dataset

This project uses the GTZAN music genre dataset.

The dataset itself is not included in this repository.