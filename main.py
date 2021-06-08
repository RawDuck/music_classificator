import matplotlib.pyplot as plt
import librosa
import librosa.display
import sklearn
import numpy as np
from scipy.fftpack import fft
import os
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import math


def spectralCentroidFeature(x, sr, draw=False):
    spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
    if draw:
        t = librosa.frames_to_time(range(len(spectral_centroids)))
        librosa.display.waveplot(x, sr=sr, alpha=0.4)
        plt.xlabel("Czas")
        plt.grid()
        plt.plot(t, normalize(spectral_centroids), color='r')
        plt.show()
    return spectral_centroids


def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)


def mfccFeature(x, sr, draw=False):
    mfccs = librosa.feature.mfcc(x, sr=sr)
    if draw:
        librosa.display.specshow(mfccs, sr=sr, x_axis='time')
        plt.grid()
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
        plt.ylabel("Amplituda")
        plt.xlabel("Częstotliwość")
        plt.show()
    return yf


def prepareData(method='mfcc'):
    x_data = []
    y_data = []

    for root, dirs, files in os.walk("./dataset/", topdown=False):
        for name in files:
            path = os.path.join(root, name)

            x, sr = librosa.load(path)
            if method == 'mfcc':
                x_val = mfccFeature(x, sr, False)
                x_val = x_val.flatten()
                x_val = x_val.tolist()
                x_val = x_val[:25800]
            elif method == 'fft':
                x_val = fftFeature(x, sr, draw=False)
                x_val = x_val.flatten()
                x_val = x_val.tolist()
                x_val = x_val[:660000]
                for i in range(len(x_val)):
                    R = x_val[i].real
                    C = x_val[i].imag
                    x_val[i] = math.sqrt(R * R + C * C)
            elif method == 'spectralCentroid':
                x_val = spectralCentroidFeature(x, sr, draw=False)
                x_val = x_val.flatten()
                x_val = x_val.tolist()
                x_val = x_val[:1290]

            x_data.append(x_val)
            if 'mix' in path:
                y_data.append(-1)
            else:
                y_data.append(1)
    return x_data, y_data


def runTests():
    methods = ['mfcc', 'fft', 'spectralCentroid']
    for m in methods:
        X, y = prepareData(m)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        clf = svm.SVC(kernel='linear')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        res = metrics.accuracy_score(y_test, y_pred) * 100
        res = "{:.2f}".format(res)
        print(f"SVM Accuracy with {m} feature: {res}%.")


audio_path = 'dataset/rock/rock.00000.wav'
x, sr = librosa.load(audio_path)
sc = spectralCentroidFeature(x, sr, draw=False)
# print(f'sc = {sc}')
mfcc = mfccFeature(x, sr, False)
# print(f'mfcc = {mfcc}')
fft_val = fftFeature(x, sr, draw=False)
# print(f'fft_val = {fft_val}')
runTests()
