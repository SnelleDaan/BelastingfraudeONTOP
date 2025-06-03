import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

# Inladen data
df = pd.read_csv('Assignment2/Life_Expectancy_Data_v2.csv')
listocol = df.columns

to_remove = ['Under-five deaths', 'GDP', 'thinness 5-9 years', 'Schooling']
missing_percent = df.isnull().mean().sort_values(ascending=False) * 100
print(missing_percent[missing_percent > 0])
#population missing 22 percent

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    mask = (df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)
    return df[mask]

df = remove_outliers_iqr(df, 'GDP')
#print(df.head())


yo = listocol.drop(['Country', 'Status'])

# for col in yo:
#     plt.boxplot(df[col])
#     plt.title(col)
#     plt.show()
    
#columns with [infant deaths, percentage expenditure, measles, , HIV/AIDS] insane amounts of outliers
col_outlier = ['Infant deaths', 'Percentage expenditure', 'Measles', 'HIV/AIDS']
for col in col_outlier:
    plt.hist(df[col])
    plt.title(col)
    plt.show()