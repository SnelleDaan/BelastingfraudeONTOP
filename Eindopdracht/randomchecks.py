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
from scipy.signal import welch
import seaborn as sns
import os

def Excersize1a(show=True):
    df = pd.read_csv('Eindopdracht/train/bearing_conditions.csv', sep=";")
    if show:
        print(df.head())
        plt.hist(df)
        plt.show()
        
Excersize1a()
