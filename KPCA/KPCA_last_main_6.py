import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA

# 读入数据
data = pd.read_csv('iris_main_6.csv')
print(data)


# 定义KPCA特征权重分析函数
def kpca_feature_weighting(data, kernel='rbf', n_components=None):
    n_samples, n_features = data.shape

    # 计算主成分分析所需的矩阵
    covariance_matrix = np.cov(data.T)
    eig_values, eig_vectors = np.linalg.eig(covariance_matrix)
    print("eig_values", eig_values)
    print("累计方差贡献1：", np.cumsum(eig_values) / np.sum(eig_values))
    print("eig_vectors:", eig_vectors)

    # 计算KPCA核矩阵和主成分分析所需的矩阵
    kpca = KernelPCA(n_components=n_components, kernel=kernel)
    kernel_matrix = kpca.fit_transform(data)
    print(kernel_matrix)
    weights = np.abs(eig_vectors.T @ kernel_matrix) / np.sqrt(np.sum(eig_values))

    # 计算各特征的权重
    feature_weights = np.mean(weights, axis=1).tolist()
    # 权重归一化
    feature_weights = feature_weights / np.sum(feature_weights)

    return feature_weights


# 使用KPCA特征权重分析函数
weights = kpca_feature_weighting(data)

print("Weight:")
# 输出结果
for weight in (weights):
    weight = round(weight, 2)
    print(weight)
