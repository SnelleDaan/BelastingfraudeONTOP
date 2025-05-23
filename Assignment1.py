import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import seaborn as sns
from ucimlrepo import fetch_ucirepo 

# # fetch dataset 
# iris = fetch_ucirepo(id=53) 

# # data (as pandas dataframes) 
# X = iris.data.features 
# y = iris.data.targets 

# df_iris1 = pd.concat([X, y], axis=1)  
# df_iris = df_iris1[df_iris1['class']!='Iris-virginica'].copy()

# # sns.pairplot(df_iris, hue = 'class')
# # plt.show()

# X = df_iris[["petal length", "petal width"]]
# y = df_iris['class']
# # metadata 
# # variable information 



# #sns.pairplot(df_iris, hue = 'class')
# #plt.show()

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# clf = SVC(C=1000000, kernel='linear')
# clf.fit(X_train, y_train)



# w = clf.coef_[0]
# a = -w[0] / w[1]
# x_ = np.linspace(X_train.iloc[:, 0].min() - 0.5 * X_train.iloc[:, 0].std(),
#                  X_train.iloc[:, 0].max() + 0.5 * X_train.iloc[:, 0].std())
# b = clf.intercept_[0]
# y_ = a * x_ - b / w[1]
# y_minus_one = a * x_ + (-b - 1) / w[1]
# y_plus_one  = a * x_ + (-b + 1) / w[1]



# plt.plot(x_, y_, 'k-')
# plt.plot(x_, y_minus_one, 'b--')
# plt.plot(x_, y_plus_one,  'r--')

# plt.scatter(X['petal length'][y == "Iris-setosa"], X['petal width'][y == "Iris-setosa"], c = 'b', label = -1)
# plt.scatter(X['petal length'][y ==  "Iris-versicolor"], X['petal width'][y ==  "Iris-versicolor"], c = 'r', label =  1)
# plt.xlabel(df_iris.columns[0])
# plt.ylabel(df_iris.columns[1])
# plt.axhline(y = 0, color = 'k', linewidth = 0.5)
# plt.axvline(x = 0, color = 'k', linewidth = 0.5)

# # Plot support vectors
# support_vectors = clf.support_vectors_
# plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s = 100, 
#             facecolor = 'None', edgecolor = 'k', linestyle = 'dashed', linewidth = 2, label = 'support vectors')

# plt.legend()
# plt.box(False)
# plt.show()

# # Prediction on X (check)
# df_iris['y_hat'] = clf.predict(X)
# print(df_iris.head())

# # Test on new x
# print(f'clf((2, 0)) = {clf.predict([[2, 0]])[0]}')







# fetch dataset 
wine = fetch_ucirepo(id=109) 

# data (as pandas dataframes) 
X = wine.data.features 
y = wine.data.targets 
df_wine1 = pd.concat([X, y], axis=1)


df_wine = df_wine1[df_wine1["class"] != 3].copy()

X = df_wine[['Alcohol', 'Color_intensity', 'Proline']]
y = df_wine['class']

clf = SVC(C = 2, kernel = 'rbf', gamma = 0.01)
clf.fit(X, y)

# metadata 
df_wine.head()
# variable information 
sns.pairplot(df_wine, hue = 'class')
plt.show()
# clf = SVC(C=2, kernel='rbf', gamma=0.01)
# clf.fit(X, y)