import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
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
df['Diphtheria '] = df['Diphtheria '].fillna(df['Diphtheria '].mean())

# drop deze want thinness 1-19 en 5-9 years missen allebei
df = df.dropna(subset=['thinness  1-19 years'])
# drop deze want alle nan waardes zijn ook nan bij schooling, dus te weinig zeggende informatie
df = df.dropna(subset=['Income composition of resources'])

# Multicollineariteit verwijderen
# corr = df.corr()
# sns.heatmap(corr, annot=True, cmap='coolwarm')
# plt.show()
to_remove = ['Under-five deaths', 'GDP', 'thinness 5-9 years', 'Schooling']
# population weg, want veel nan en weinig invloed op y
to_remove2 = ['Population']
df = df.drop(columns=to_remove)
df = df.drop(columns=to_remove2)

# Split de data in man en women
df_man = df.copy()
cols_to_drop = [col for col in df.columns if '(women)' in col]
df_man = df_man.drop(columns=cols_to_drop)

df_woman = df.copy()
cols_to_drop = [col for col in df.columns if '(men)' in col]
df_woman = df_woman.drop(columns=cols_to_drop)

# Data splitten men
X_men = df_man.drop(columns='Life expectancy (men)').copy()
y_men = df_man['Life expectancy (men)']
Xmen_train, Xmen_test, ymen_train, ymen_test = train_test_split(X_men, y_men, test_size=0.2, random_state=42)

# Data splitten women
X_women = df_woman.drop(columns='Life expectancy(women)').copy()
y_women = df_woman['Life expectancy(women)']
Xwomen_train, Xwomen_test, ywomen_train, ywomen_test = train_test_split(X_women, y_women, test_size=0.2, random_state=42)


numeric_cols = df.select_dtypes(include=['number']).columns

X_train_const_m = sm.add_constant(Xmen_train)
X_train_const_v = sm.add_constant(Xwomen_train)

# model_m = sm.OLS(ymen_train, X_train_const_m).fit()
# model_v = sm.OLS(ywomen_train, X_train_const_v).fit()
# # print(X_train_const_m.isnull().sum())
# # print(any(X_train_const_m.isna()))
# Xmen_test = sm.add_constant(Xmen_test)
# Xwomen_test = sm.add_constant(Xwomen_test)

# pred_m = model_m.predict(Xmen_test)
# pred_v = model_v.predict(Xwomen_test)
# print(pred_m)
# r2_score_m = r2_score(ymen_test,pred_m)
# r2_score_v = r2_score(ywomen_test ,pred_v)
# plt.plot(pred_m, 'o', color='r')
# plt.plot(ymen_test, 'o', color='b')
# plt.show()
# print(r2_score_m, r2_score_v)
# # data heeft duidelijk geen linear model