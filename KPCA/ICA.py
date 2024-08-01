import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA

# 加载数据集并转置
data = pd.read_csv('iris_main_6.csv')
print(data)
mu = np.mean(data, axis=0)
data = (data - mu) / np.std(data, axis=0)

# 初始化 ICA 模型并降维到 3 维
ica = FastICA(n_components=3)
X_ica = ica.fit_transform(data)
print('X_ica:', X_ica)

components = ica.components_
print('compoents:', components)
icaspace = np.dot(data, components.T)
print('icaspace:', icaspace)
feature_weights = abs(components.sum(axis=0))
print('feature_weights:', feature_weights)
feature_weights = feature_weights / np.sum(feature_weights)

print(feature_weights)
