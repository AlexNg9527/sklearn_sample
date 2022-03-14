from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris['data'],
                                                    iris['target'],
                                                    test_size=0.2)

print('The size of X_train is ', X_train.shape)
print('The size of y_train is ', y_train.shape)
print('The size of X_test is ', X_test.shape)
print('The size of y_test is ', y_test.shape)

model = LogisticRegression(solver='lbfgs', multi_class='multinomial')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)  # 预测的类别
p_pred = model.predict_proba(X_test)  # 预测该类别的信心
# print(y_test, '\n')
# print(y_pred, '\n')
# print(p_pred)

s = ['Class 1 Prob', 'Class 2 Prob', 'Class 3 Prob']
prob_DF = pd.DataFrame(p_pred, columns=s)
prob_DF['Predicted Class'] = y_pred
print(prob_DF.head())

model = KMeans(n_clusters=3)

X = iris.data[:, 0:2]
model.fit(X)

index_pred = model.predict(X_test[:, 0:2])
print(index_pred)
print(y_test)

r_hex, g_hex, dt_hex = '#FFAAAA', '#AAFFAA', '#AAAAFF'
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
# cmap_bold1 = ListedColormap(['#FFAAAA', '#AAAAFF', '#AAFFAA'])
# cmap_bold2 = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold1 = ListedColormap([r_hex, g_hex, dt_hex])
cmap_bold2 = ListedColormap([r_hex, dt_hex, g_hex])
centroid = model.cluster_centers_

true_centroid = np.vstack((X_test[y_test == 0, 0:2].mean(axis=0),
                           X_test[y_test == 1, 0:2].mean(axis=0),
                           X_test[y_test == 2, 0:2].mean(axis=0)))

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_test[:, 0], X_test[:, 1], c=index_pred, cmap=cmap_bold1)
plt.scatter(centroid[:, 0], centroid[:, 1], marker='o', s=200,
            edgecolors='k', c=[0, 1, 2], cmap=cmap_light)
plt.xlabel('sepal length(cm)')
plt.ylabel('sepal width(cm)')
plt.title('Predicted Class')

plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=index_pred, cmap=cmap_bold2)
plt.scatter(centroid[:, 0], centroid[:, 1], marker='o', s=200,
            edgecolors='k', c=[0, 1, 2], cmap=cmap_light)
plt.xlabel('sepal length(cm)')
plt.ylabel('sepal width(cm)')
plt.title('True Class')

plt.show()  # KMeans 算法里标注的类别索引和真实类别索引不一样
