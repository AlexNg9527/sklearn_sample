from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn import datasets
import numpy as np
from matplotlib import pyplot as plt

model = KMeans(n_clusters=3)
iris = datasets.load_iris()
X = iris.data[:, 0:2]
model.fit(X)

r_hex, g_hex, dt_hex = '#FFAAAA', '#AAFFAA', '#AAAAFF'
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
# cmap_bold1 = ListedColormap(['#FFAAAA', '#AAAAFF', '#AAFFAA'])
# cmap_bold2 = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold1 = ListedColormap([r_hex, g_hex, dt_hex])
cmap_bold2 = ListedColormap([r_hex, dt_hex, g_hex])
centroid = model.cluster_centers_
label = iris.target
true_centroid = np.vstack((X[label == 0, :].mean(axis=0),
                           X[label == 1, :].mean(axis=0),
                           X[label == 2, :].mean(axis=0)))

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap=cmap_bold1)
plt.scatter(centroid[:, 0], centroid[:, 1], marker='o', s=200,
            edgecolors='k', c=[0, 1, 2], cmap=cmap_light)
plt.xlabel('sepal length(cm)')
plt.ylabel('sepal width(cm)')
plt.title('Cluster Class')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=iris.target, cmap=cmap_bold2)
plt.scatter(true_centroid[:, 0], true_centroid[:, 1], marker='o',
            s=200, edgecolors='k', c=[0, 2, 1], cmap=cmap_light)
plt.xlabel('sepal length(cm)')
plt.ylabel('sepal width(cm)')
plt.title('True Class')

plt.show()  # KMeans 算法里标注的类别索引和真实类别索引不一样
