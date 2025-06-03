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

# Multicollineariteit verwijderen
# corr = df.corr()
# sns.heatmap(corr, annot=True, cmap='coolwarm')
# plt.show()
to_remove = ['Under-five deaths', 'GDP', 'thinness 5-9 years', 'Schooling']

# Data splitten
X = df[[]]
y = df['Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
