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
import pickle
from pathlib import Path
import time


def spectral_centroid_feature(x, sr, draw=False):
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


def mfcc_feature(x, sr, draw=False):
    mfccs = librosa.feature.mfcc(x, sr=sr, n_mfcc=100000)
    if draw:
        librosa.display.specshow(mfccs, sr=sr, x_axis='time')
        plt.grid()
        plt.xlabel("Czas")
        plt.show()
    return mfccs


def fft_feature(x, sr, draw=False):
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


def prepare_data(method='mfcc'):
    x_data = []
    y_data = []

    for root, dirs, files in os.walk("./dataset_multiclass/", topdown=False):
        for name in files:
            path = os.path.join(root, name)

            x, sr = librosa.load(path)
            if method == 'mfcc':
                x_val = mfcc_feature(x, sr, False)
                x_val = x_val.flatten()
                x_val = x_val.tolist()
                x_val = x_val[:100000]
            elif method == 'fft':
                x_val = fft_feature(x, sr, draw=False)
                x_val = x_val.flatten()
                x_val = x_val.tolist()
                x_val = x_val[:660000]
                for i in range(len(x_val)):
                    R = x_val[i].real
                    C = x_val[i].imag
                    x_val[i] = math.sqrt(R * R + C * C)
            elif method == 'spectralCentroid':
                x_val = spectral_centroid_feature(x, sr, draw=False)
                x_val = x_val.flatten()
                x_val = x_val.tolist()
                x_val = x_val[:1290]
            x_data.append(x_val)
            if 'rock' in path:
                y_data.append(1)
            if 'hiphop' in path:
                y_data.append(2)
            if 'pop' in path:
                y_data.append(3)
            if 'metal' in path:
                y_data.append(4)
    return x_data, y_data


def run_tests(methods = [], save = False, pickles_filepaths = []):
    if len(methods) == 0:
        print('No methods to run!')
        return

    for i, m in enumerate(methods):
        start = time.time()
        X, y = prepare_data(m)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        if i < len(pickles_filepaths) - 1:
            p = Path(pickles_filepaths[i])
            clf = pickle.load(p.open(mode='rb'))
            print(f'Use loaded pickle: {p}')
        else:
            clf = svm.SVC(kernel='linear')
            clf.fit(X_train, y_train)

        if save:
            p = Path(f'pickle_{m}')
            print(f'Saving pickle: {p}')
            p.touch(exist_ok=True)
            opened_file = p.open(mode='wb')
            pickle.dump(clf, opened_file)

        y_pred = clf.predict(X_test)
        stop = time.time()
        f = open(m + "_feature_acc.txt", "w")
        f.write("y_test | y_pred\n")
        for j in range(len(y_test)):
            f.write(str(y_test[j]) + " | " + str(y_pred[j]) + "\n")
        f.close()

        res = metrics.accuracy_score(y_test, y_pred) * 100
        res = "{:.2f}".format(res)
        print("------------------------------------------------")
        print(f"SVM Accuracy with {m} feature: {res}%.")
        multiclass_accuracy(y_test, y_pred)
        print("Elapsed time: " + str(stop - start) + "s")
        print("------------------------------------------------")


def multiclass_accuracy(y_test, y_pred):
    sum1 = 0
    sum2 = 0
    sum3 = 0
    sum4 = 0
    for i in range(len(y_test)):
        if y_pred[i] == y_test[i] and y_test[i] == 1:
            sum1 += 1
        if y_pred[i] == y_test[i] and y_test[i] == 2:
            sum2 += 1
        if y_pred[i] == y_test[i] and y_test[i] == 3:
            sum3 += 1
        if y_pred[i] == y_test[i] and y_test[i] == 4:
            sum4 += 1
    acc1 = (sum1 / y_test.count(1)) * 100
    acc2 = (sum2 / y_test.count(2)) * 100
    acc3 = (sum3 / y_test.count(3)) * 100
    acc4 = (sum4 / y_test.count(4)) * 100
    acc1 = "{:.2f}".format(acc1)
    acc2 = "{:.2f}".format(acc2)
    acc3 = "{:.2f}".format(acc3)
    acc4 = "{:.2f}".format(acc4)

    print("Accuracy for Rock: " + str(acc1) + "%")
    print("Accuracy for Hiphop: " + str(acc2) + "%")
    print("Accuracy for Pop: " + str(acc3) + "%")
    print("Accuracy for Metal: " + str(acc4) + "%")



if __name__ == '__main__':
    methods = ['mfcc', 'fft', 'spectralCentroid']

    # Run clearly
    # run_tests(methods, True)

    # Run saved pickles
    pickle_filepaths = [f'pickle_{m}' for m in methods]
    run_tests(methods, False, pickle_filepaths)