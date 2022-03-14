from sklearn.cluster import KMeans
from sklearn import datasets

iris = datasets.load_iris()
model = KMeans(n_clusters=3)

X = iris.data[:, 0:2]
model.fit(X)
print(model.cluster_centers_, '\n')  # 簇中心。三个簇那么有三个坐标
print(model.labels_, '\n')  # 聚类后的标签
print(model.inertia_, '\n')  # 所有点到对应的簇中心的距离平方和 (越小越好)
print(iris.target)

# # 无监督学习
# from sklearn.xxx import SomeModel
#
# # xxx 可以是 cluster 或 decomposition 等
#
#
# model = SomeModel(hyperparameter)
# model.fit(X)
