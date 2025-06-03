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
# Data bewerken
df['Status'] = df['Status'].map({'Developed': 1, 'Developing': 0})
le = LabelEncoder()
df['Country'] = le.fit_transform(df['Country'])

# Data opschonen
df['BMI'] = df['BMI'].fillna(df['BMI'].mean())
df['Alcohol'] = df['Alcohol'].fillna(df.groupby('Country')['Alcohol'].transform('mean'))
df['Alcohol'] = df['Alcohol'].fillna(df['Alcohol'].mean())
df['Polio'] = df['Polio'].fillna(df['Polio'].mean())
df['Total expenditure'] = df['Total expenditure'].fillna(df.groupby('Country')['Total expenditure'].transform('mean'))
df['Total expenditure'] = df['Total expenditure'].fillna(df['Total expenditure'].mean())


# drop deze want thinness 1-19 en 5-9 years missen allebei
df = df.dropna(subset=['thinness  1-19 years'])
# drop deze want alle nan waardes zijn ook nan bij schooling, dus te weinig zeggende informatie
df = df.dropna(subset=['Income composition of resources'])

print(df[df['thinness  1-19 years'].isna()])
print(df.isnull().sum())

# Multicollineariteit verwijderen
# corr = df.corr()
# sns.heatmap(corr, annot=True, cmap='coolwarm')
# plt.show()
to_remove = ['Under-five deaths', 'GDP', 'thinness 5-9 years', 'Schooling']
# population weg, want veel nan en weinig invloed
to_remove2 = ['Population']
df = df.drop(columns=to_remove)

# Split de data in man en women
df_man = df.copy()
cols_to_drop = [col for col in df.columns if '(women)' in col]
df_man = df_man.drop(columns=cols_to_drop)

df_woman = df.copy()
cols_to_drop = [col for col in df.columns if '(men)' in col]
df_woman = df_woman.drop(columns=cols_to_drop)

# Data splitten
X_men = df_man.drop(columns='Life expectancy (men)').copy()
y_men = df_man['Life expectancy (men)']
Xmen_train, Xmen_test, ymen_train, ymen_test = train_test_split(X_men, y_men, test_size=0.2, random_state=42)

X_women = df_woman.drop(columns='Life expectancy(women)').copy()
y_women = df_woman['Life expectancy(women)']
Xwomen_train, Xwomen_test, ywomen_train, ywomen_test = train_test_split(X_women, y_women, test_size=0.2, random_state=42)
