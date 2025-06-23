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




# Excersize 1a)
def Excersize1a():
    df = pd.read_csv('Eindopdracht/train/0.csv', sep=";")
    print(df.head())
    plt.plot(df['b4x'])
    plt.show()


# Excersize 1b)
fs = 20480
def spectral_flatness(signal):
    """
    Bereken spectral flatness via Welch's methode.
    """
    freqs, power = welch(signal, fs=fs)
    geometric_mean = np.exp(np.mean(np.log(power + 1e-12)))  # voorkom log(0)
    arithmetic_mean = np.mean(power)
    return geometric_mean / arithmetic_mean

def load_features(index):
    """
    Laadt het bestand en berekent waardes.
    
    Parameters:
    index (int): Index van het sample (bijv. 0 voor '0.csv')
    data_dir (str): Pad naar de directory waar de CSV-bestanden staan
    
    Returns:
    dict: Statistieken (mean, std, rms) voor b4x en b4y
    """
    filepath = 'Eindopdracht/train/' + str(index) + '.csv'
    df = pd.read_csv(filepath, sep=';')
    
    features = {}
    for col in list(df.columns.values):
        signal = df[col]
        features[f'mean_{col}'] = np.mean(signal)
        features[f'std_{col}'] = np.std(signal)
        features[f'rms_{col}'] = np.sqrt(np.mean(signal**2))
        features[f'kurtosis_{col}'] = kurtosis(signal)
        features[f'flatness_{col}'] = spectral_flatness(signal)

    return features

# Excersize 1c)
def Excersize1c():
    conditions = pd.read_csv('Eindopdracht/train/bearing_conditions.csv', sep=";")
    conditions.reset_index(inplace=True)

    # Feature extractie op alle bestanden
    bestanden =  os.listdir('Eindopdracht/train')
    csv_bestanden = [f for f in bestanden if f.endswith('.csv')]
    aantal = len(csv_bestanden)

    all_features = []
    for i in range(aantal-1):
        feat = load_features(i)
        feat['index'] = i
        all_features.append(feat)

    df_features = pd.DataFrame(all_features)
    df_full = df_features.merge(conditions, on='index')
    df_full.set_index('index', inplace=True)
    cols_to_drop = [col for col in df_full.columns if 'b1' in col]
    df_full = df_full.drop(columns=cols_to_drop)
    # Opslaan als CSV
    df_full.to_csv('Eindopdracht/bearing_features.csv', index=False)
    return df_full


def Excersize1cV3(df_full):
    sns.pairplot(df_full, hue='b4_state')
    plt.show()
    
def Excersize1cV2(df_full):
    X = df_full.drop(columns=['b4_state'])
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    te_verwijderen = [col for col in upper.columns if any(upper[col] > 0.8)]

    X_reduced = X.drop(columns=te_verwijderen)
    X_reduced["b4_state"] = df_full["b4_state"]  # Voeg stage weer toe voor de PCA-plot
    sns.heatmap(X_reduced.corr(), cmap='coolwarm', center=0, annot=True)
    plt.title('Feature correlatiematrix')
    plt.show()
    X_reduced.to_csv('Eindopdracht/bearing_features_reduced.csv', index=False)
    Excersize1cV3(X_reduced)






# Excersize 1e)
def Excersize1e(df_final):
    X = df_final.drop(columns=['b4_state'])   # stage is target
    y = df_final['b4_state']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


    # === 4. Schalen (voor SVM vereist) ===
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # === 5. Random Forest trainen ===
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    # === 6. SVM (RBF) trainen ===
    svm = SVC(kernel='rbf', random_state=42)
    svm.fit(X_train_scaled, y_train)
    svm_pred = svm.predict(X_test_scaled)

    # === 7. Evaluatie ===
    def evaluate_model(y_true, y_pred, model_name):
        print(f"\n----- {model_name} -----")
        print("Accuracy:", accuracy_score(y_true, y_pred))
        print("Balanced Accuracy:", balanced_accuracy_score(y_true, y_pred))
        print("F1-score:", f1_score(y_true, y_pred, average='weighted'))
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_true))
        disp.plot(cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.show()

    evaluate_model(y_test, rf_pred, "Random Forest")
    evaluate_model(y_test, svm_pred, "SVM (RBF)")

df_final = pd.read_csv('Eindopdracht/bearing_features_reduced.csv')
Excersize1e(df_final)