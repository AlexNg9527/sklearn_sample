from sklearn.datasets import load_iris
from sklearn import datasets
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

iris = load_iris()
# 数据集包括 150 条鸢尾花的四个特征 (萼片长/宽和花瓣长/宽) 和三个类别
# iris.keys()

# data：特征值 (数组)
# target：标签值 (数组)
# target_names：标签 (列表)
# DESCR：数据集描述
# feature_names：特征 (列表)
# filename：iris.csv 文件路径

# iris.data.shape # (150, 4)
# iris.target

iris_data = pd.DataFrame(iris.data,
                         columns=iris.feature_names)
iris_data['species'] = iris.target_names[iris.target]
iris_data.head(3).append(iris_data.tail(3))

sns.pairplot(iris_data, hue='species', palette='husl')
plt.show()

