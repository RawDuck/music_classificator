import matplotlib.pyplot as plt
import librosa
import librosa.display
import sklearn
import scipy
import numpy as np
from scipy.fftpack import fft


def spectralCentroidFeature(x, sr, draw=False):
    spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
    if draw:
        t = librosa.frames_to_time(range(len(spectral_centroids)))
        librosa.display.waveplot(x, sr=sr, alpha=0.4)
        plt.xlabel("Czas")
        plt.plot(t, normalize(spectral_centroids), color='r')
        plt.show()
    return spectral_centroids


def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)


def mfccFeature(x, sr, draw=False):
    mfccs = librosa.feature.mfcc(x, sr=sr)
    if draw:
        librosa.display.specshow(mfccs, sr=sr, x_axis='time')
        plt.xlabel("Czas")
        plt.show()
    return mfccs

def fftFeature(x, sr, draw=False):
    yf = fft(x)
    if draw:
        n = len(x)
        T = 1 / sr
        xf = np.linspace(0.0, 1.0 / (2.0 * T), int(n / 2))
        fig, ax = plt.subplots()
        ax.plot(xf, 2.0 / n * np.abs(yf[:n // 2]))
        plt.grid()
        plt.xlabel("Częstotliwość")
        plt.ylabel("Amplituda")
        plt.show()
    return yf


audio_path = 'dataset/rock/rock.00000.wav'
x, sr = librosa.load(audio_path)
sc = spectralCentroidFeature(x, sr, draw=False)
mfcc = mfccFeature(x, sr, False)
fft_val = fftFeature(x, sr, draw=False)
