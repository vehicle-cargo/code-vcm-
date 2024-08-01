import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# 假设我们有一个二维数组data，每一列是一个指标，每一行是一个观测值
data = pd.read_csv('iris_2.csv')
data = np.array(data)
print(data)
# 数据标准化
scaler = StandardScaler()
data_std = scaler.fit_transform(data)

# 进行主成分分析，假设我们保留所有主成分
pca = PCA(n_components=data.shape[1])
pca.fit(data_std)

# 计算各个主成分的累积贡献率
cum_ratio = np.cumsum(pca.explained_variance_ratio_)


# 找到累积贡献率超过85%的主成分数量
n_components = np.argmax(cum_ratio >= 0.85) + 1
print(n_components)
# 重新进行主成分分析，只保留上一步找到的主成分数量
pca = PCA(n_components=n_components)
data_pca = pca.fit_transform(data_std)

# 计算原始所有指标的权重
weights = abs(np.dot(pca.components_.T, pca.explained_variance_ratio_))
weights=weights/np.sum(weights)
print("原始所有指标的权重：", weights)
