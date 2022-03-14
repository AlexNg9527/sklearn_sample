from sklearn.datasets import load_iris
from sklearn import datasets
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.linear_model import LinearRegression

model = LinearRegression(normalize=True)

x = np.arange(10)
y = 2 * x + 1
plt.plot(x, y, 'o')
plt.show()

X = x[:, np.newaxis]  # sample*feather
model.fit(X, y)

print(model.coef_)
print(model.intercept_)

# # 有监督学习
# from sklearn.xxx import SomeModel
#
# # xxx 可以是 linear_model 或 ensemble 等
#
#
# model = SomeModel(hyperparameter)
# model.fit(X, y)