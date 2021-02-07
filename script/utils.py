#----------------------------------------------------------------------------
#   Utils
#----------------------------------------------------------------------------

import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import librosa
from scipy.stats import kurtosis
from scipy.stats import skew
import matplotlib.pyplot as plt


#----------------------------------------------------------------------------
#   Preprocessing utils
#----------------------------------------------------------------------------

def get_feature_stats(features):
    """
    Get the summary statistics from the mfcc vectors in the extract features function
    """
    result = {}
    for k, v in features.items():
        result['{}_max'.format(k)] = np.max(v)
        result['{}_min'.format(k)] = np.min(v)
        result['{}_mean'.format(k)] = np.mean(v)
        result['{}_std'.format(k)] = np.std(v)
        result['{}_kurtosis'.format(k)] = kurtosis(v)
        result['{}_skew'.format(k)] = skew(v)
    return result


def extract_features(y,sr=22050,n_fft=1024,hop_length=512, n_mfcc=20):
    """
    Get the features from a specific track
    """
    features = {'centroid': librosa.feature.spectral_centroid(y, sr=sr, n_fft=n_fft, hop_length=hop_length).ravel(),
                'flux': librosa.onset.onset_strength(y=y, sr=sr).ravel(),
                'rmse': librosa.feature.rms(y, frame_length=n_fft, hop_length=hop_length).ravel(),
                'zcr': librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length).ravel(),
                'contrast': librosa.feature.spectral_contrast(y, sr=sr).ravel(),
                'bandwidth': librosa.feature.spectral_bandwidth(y, sr=sr, n_fft=n_fft, hop_length=hop_length).ravel(),
                'flatness': librosa.feature.spectral_flatness(y, n_fft=n_fft, hop_length=hop_length).ravel(),
                'rolloff': librosa.feature.spectral_rolloff(y, sr=sr, n_fft=n_fft, hop_length=hop_length).ravel(),
                'tempo':librosa.beat.tempo(y=y,sr=sr,hop_length=hop_length)[0]}

    # MFCC treatment
    mfcc = librosa.feature.mfcc(y, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)
    for idx, v_mfcc in enumerate(mfcc):
        features['mfcc_{}'.format(idx)] = v_mfcc.ravel()
    dict_agg_features = get_feature_stats(features)

    return dict_agg_features


def envelope(y, sr, threshold):
    """
    Data cleaning using an enveloppe
    """
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(sr/10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask

#----------------------------------------------------------------------------
#   Data Viz utils
#----------------------------------------------------------------------------

def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs
        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()
