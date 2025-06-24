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
from sklearn.cluster import SpectralClustering
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.mixture import GaussianMixture
from scipy.stats import kurtosis
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from scipy.signal import welch
from sklearn.decomposition import KernelPCA
import seaborn as sns
from sklearn.decomposition import PCA
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

def load_bearing1_features(index, highestStd):
    """
    Laadt acceleratie-data van sample met gegeven index,
    en berekent statistische en spectrale features voor bearing 1 (b1x en b1y).
    """
    filepath = f'Eindopdracht/train/{index}.csv'
    df = pd.read_csv(filepath, sep=';')
    
    features = {}
    newstd = highestStd
    for col in ['b1x', 'b1y']:
        signal = df[col]
        features[f'mean_{col}'] = np.mean(signal)
        features[f'std_{col}'] = np.std(signal)
        if np.std(signal) > highestStd:
            newstd = np.std(signal)
            features[f'max_std_{col}'] = np.std(signal)
        else:
            features[f'max_std_{col}'] = highestStd
        features[f'rms_{col}'] = np.sqrt(np.mean(signal**2))
        features[f'kurtosis_{col}'] = kurtosis(signal)
        features[f'flatness_{col}'] = spectral_flatness(signal)
        features[f'crest_{col}'] = crest_factor(signal)
    
    return features, newstd

#Opdracht 3b)
def getData():
    all_features_b1 = []
    highestStd = 0
    for i in range(1724):  # Er zijn 1724 samples (0.csv t/m 1723.csv)
        features, highestStd = load_bearing1_features(i, highestStd)
        features['time'] = i  # sample-index als tijd
        all_features_b1.append(features)
    df_b1_features = pd.DataFrame(all_features_b1)
    df_b1_features.to_csv('Eindopdracht/Opdracht3/bearing_features.csv', index=False)
getData()

def showGraph(df_b1_features):
    for col in df_b1_features.columns:
        if col != 'time':
            plt.figure(figsize=(10, 3))
            plt.plot(df_b1_features['time'], df_b1_features[col])
            plt.title(f'{col} over tijd')
            plt.xlabel('Tijd (sample index)')
            plt.ylabel(col)
            plt.grid(True)
            plt.tight_layout()
            plt.show()

def corrMatrix(df_b1_features):
    corr = df_b1_features.drop(columns='time').corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title("Correlatiematrix van bearing 1 features")
    plt.show()

def removeMultiColl(df_b1_features):
    X = df_b1_features.drop(columns=['time'])
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    te_verwijderen = [col for col in upper.columns if any(upper[col] > 0.8)]

    X_reduced = X.drop(columns=te_verwijderen)
    X_reduced["time"] = df_b1_features["time"]  # Voeg stage weer toe voor de PCA-plot
    X_reduced.to_csv('Eindopdracht/Opdracht3/bearing_features_reduced.csv', index=False)



df_b1_features = pd.read_csv('Eindopdracht/Opdracht3/bearing_features.csv')
# showGraph(df_b1_features)
corrMatrix(df_b1_features)
removeMultiColl(df_b1_features)
df_b1_reduced = pd.read_csv('Eindopdracht/Opdracht3/bearing_features_reduced.csv')

#Opdracht 3c)
def Clustering(df_b1_clean, df_b1_features):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_b1_clean)

    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    gmm = GaussianMixture(n_components=5, random_state=42)
    labels = gmm.fit_predict(X_scaled)

    df_b1_clean['cluster'] = labels
    df_b1_features['cluster'] = labels

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=50)
    plt.title('PCA-visualisatie van clusters')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.colorbar(label='Cluster')
    plt.grid(True)
    plt.show()
    return df_b1_features

df_b1_clusters = Clustering(df_b1_reduced, df_b1_features)

#Opdracht 3d)
def InterpetClusters(df_b1_clusters):
    cluster_summary = df_b1_clusters.groupby('cluster').mean()
    print(cluster_summary)

    plt.figure(figsize=(10, 4))
    plt.scatter(df_b1_clusters['time'], df_b1_clusters['mean_b1x'], c=df_b1_clusters['cluster'], cmap='viridis')
    plt.title('RMS b1x over tijd, gekleurd per cluster')
    plt.xlabel('Sample index (tijd)')
    plt.ylabel('RMS b1x')
    plt.colorbar(label='Cluster')
    plt.show()
InterpetClusters(df_b1_clusters)