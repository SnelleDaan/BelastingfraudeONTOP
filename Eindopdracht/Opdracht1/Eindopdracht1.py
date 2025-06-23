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




# Excersize 1a)
def Excersize1a():
    """
    Opdracht1, lees een aantal bestanden in en plot ze.
    """
    df = pd.read_csv('Eindopdracht/train/0.csv', sep=";")
    df2 = pd.read_csv('Eindopdracht/train/780.csv', sep=";")
    df3 = pd.read_csv('Eindopdracht/train/822.csv', sep=";")
    df4 = pd.read_csv('Eindopdracht/train/1195.csv', sep=";")
    df5 = pd.read_csv('Eindopdracht/train/1486.csv', sep=";")
    fig, axs = plt.subplots(2,3)
    axs[0,0].plot(df['b4x'])
    axs[0,0].set_title('file 0')
    axs[0,1].plot(df2['b4x'])
    axs[0,1].set_title('file 780')
    axs[0,2].plot(df3['b4x'])
    axs[0,2].set_title('file 822')
    axs[1,0].plot(df4['b4x'])
    axs[1,0].set_title('file 1195')
    axs[1,1].plot(df5['b4x'])
    axs[1,1].set_title('file 1486')
    plt.tight_layout()
    plt.show()
Excersize1a()


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
def StoreResults():
    """
    Laadt alle ruwe acceleratie-data van bearing 4, extraheert statistische en spectrale features
    uit elk bestand, koppelt deze aan de corresponderende degradatiestage, verwijdert features
    gerelateerd aan bearing 1 en slaat het volledige feature-overzicht op als een CSV-bestand.

    Returns:
        pd.DataFrame: Een DataFrame met berekende features (alleen bearing 4) en bijbehorende labels.
    """
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
    df_full.to_csv('Eindopdracht/Opdracht1/bearing_features.csv', index=False)
    return df_full
StoreResults()
df_full = pd.read_csv('Eindopdracht/Opdracht1/bearing_features.csv')

def MakeSnsPairplot(df_full):
    """
    Deze functie maakt een sns pairplot van de meegegeven data.
    """
    sns.pairplot(df_full, hue='b4_state')
    plt.show()
    
def RemoveMulticol(df_full):
    """
    Verwijdert multicollineaire features uit het DataFrame op basis van correlatieanalyse.

    Berekent de absolute correlatiematrix van de features (zonder targetkolom 'b4_state'),
    en verwijdert alle features die een correlatie hoger dan 0.8 hebben met een andere feature.
    De overgebleven features worden terug gecombineerd met de targetkolom.

    De functie toont:
    - Een heatmap van de nieuwe correlatiematrix.
    - Een pairplot via MakeSnsPairplot().

    De gereduceerde feature set wordt opgeslagen als 'bearing_features_reduced.csv'.

    Args:
        df_full (pd.DataFrame): DataFrame met features en targetkolom 'b4_state'.

    Returns:
        None
    """
    X = df_full.drop(columns=['b4_state'])
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    te_verwijderen = [col for col in upper.columns if any(upper[col] > 0.8)]

    X_reduced = X.drop(columns=te_verwijderen)
    X_reduced["b4_state"] = df_full["b4_state"]  # Voeg stage weer toe voor de PCA-plot
    sns.heatmap(X_reduced.corr(), cmap='coolwarm', center=0, annot=True)
    plt.title('Feature correlatiematrix')
    plt.show()
    X_reduced.to_csv('Eindopdracht/Opdracht1/bearing_features_reduced.csv', index=False)
    MakeSnsPairplot(X_reduced)
RemoveMulticol(df_full)



# Excersize 1e)
def MakeModelsAndEvaluate(df_final):
    """
    Trained en evalueert twee classificatiemodellen (Random Forest en SVM met RBF-kernel)
    voor het voorspellen van de degradatiestage van lager 4 ('b4_state').

    Werkwijze:
    1. Splitst de dataset in train- en testsets (stratified).
    2. Schaal de features (vereist voor SVM).
    3. Trained een Random Forest Classifier.
    4. Trained een SVM Classifier met RBF-kernel.
    5. Voert model-evaluatie uit met accuracy, balanced accuracy, F1-score, classificatierapport en een confusion matrix.
    6. Past hyperparameter tuning toe (via GridSearchCV) op beide modellen en toont de best gevonden parameters.

    Args:
        df_final (pd.DataFrame): DataFrame met features en targetkolom 'b4_state'.

    Returns:
        None. Resultaten worden geprint en visualisaties worden getoond.
    """
    # === 1. Maak de X ===
    X = df_final.drop(columns=['b4_state'])   # stage is target
    # === 2. Maak de y ===
    y = df_final['b4_state']
    # === 3. Split de data in een train en test set ===
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

    # === 8. Parameter Tuning ===
    def ParameterTuningForest(X_train, y_train, X_test, y_test):
        param_grid_rf = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
        }
        rf = RandomForestClassifier(random_state=42)
        grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='accuracy')
        grid_rf.fit(X_train, y_train)

        print("Beste parameters (RF):", grid_rf.best_params_)
        print("Accuracy op testset:", grid_rf.score(X_test, y_test))

    def ParameterTuningSVM(X_train, y_train, X_test, y_test):
        param_grid_svm = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 0.01, 0.001]
        }

        svm = SVC(kernel='rbf')
        grid_svm = GridSearchCV(svm, param_grid_svm, cv=5, scoring='accuracy')
        grid_svm.fit(X_train, y_train)

        print("Beste parameters (SVM):", grid_svm.best_params_)
        print("Accuracy op testset:", grid_svm.score(X_test, y_test))

    ParameterTuningForest(X_train, y_train, X_test, y_test)
    ParameterTuningSVM(X_train, y_train, X_test, y_test)

df_final = pd.read_csv('Eindopdracht/Opdracht1/bearing_features_reduced.csv')
MakeModelsAndEvaluate(df_final)
