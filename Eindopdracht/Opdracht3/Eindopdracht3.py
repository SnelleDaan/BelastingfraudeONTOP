import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from scipy.stats import kurtosis
from sklearn.model_selection import GridSearchCV
from scipy.signal import welch
import seaborn as sns
import os

#Opdracht 3a)
fs = 20480
def spectral_flatness(signal):
    """
    Bereken spectral flatness via Welch's methode.
    """
    freqs, power = welch(signal, fs=fs)
    geometric_mean = np.exp(np.mean(np.log(power + 1e-12)))  # voorkom log(0)
    arithmetic_mean = np.mean(power)
    return geometric_mean / arithmetic_mean

def crest_factor(signal):
    peak = np.max(np.abs(signal))
    rms = np.sqrt(np.mean(signal**2))
    return peak / rms

def load_bearing1_features(index):
    """
    Laadt acceleratie-data van sample met gegeven index,
    en berekent statistische en spectrale features voor bearing 1 (b1x en b1y).
    """
    filepath = f'Eindopdracht/train/{index}.csv'
    df = pd.read_csv(filepath, sep=';')
    
    features = {}
    for col in ['b1x', 'b1y']:
        signal = df[col]
        features[f'mean_{col}'] = np.mean(signal)
        features[f'std_{col}'] = np.std(signal)
        features[f'rms_{col}'] = np.sqrt(np.mean(signal**2))
        features[f'kurtosis_{col}'] = kurtosis(signal)
        features[f'flatness_{col}'] = spectral_flatness(signal)
        features[f'crest_{col}'] = crest_factor(signal)
    
    return features