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


#Opdracht 1a)
def LoadAndVisualizeData():
    # 1. Data inlezen
    data = pd.read_csv("Eindopdracht/shaft_radius.csv", header=None, names=["radius"])
    data["time"] = data.index  # Voeg een tijdkolom toe (0 t/m 999 uur)

    # 2. Train/test split (bijv. 80/20)
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

    # 3. Plot van trainingsdata
    plt.plot(train_data["time"], train_data["radius"], label="Train data", color="blue")
    plt.title("Shaft Radius over Time (Train Set)")
    plt.xlabel("Time [hours]")
    plt.ylabel("Shaft Radius [m]")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.xticks(ticks=range(0, 800, 100))
    plt.show()

LoadAndVisualizeData()