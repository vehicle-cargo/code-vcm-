import numpy as np
from sklearn.decomposition import PCA

# 原始数据矩阵
data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])

# 创建PCA对象
pca = PCA()

# 执行主成分分析
pca.fit(data)

# 打印指标在不同主成分线性组合的系数
print("指标系数：")
for i in range(pca.components_.shape[0]):
    print("主成分 {} : {}".format(i+1, pca.components_[i]))

# 打印综合得分模型
print("\n综合得分模型：")
for i in range(pca.explained_variance_ratio_.shape[0]):
    print("主成分 {} 的方差解释比例: {:.4f}".format(i+1, pca.explained_variance_ratio_[i]))

# 打印每个原始指标的权重
print("\n原始指标权重：")
for i in range(pca.components_.shape[1]):
    print("指标 {} 的权重: {:.4f}".format(i+1, np.abs(pca.components_[:, i]).sum()))

# 打印每个原始指标的权重百分比
total_variance = np.sum(pca.explained_variance_ratio_)
print("\n原始指标权重百分比：")
for i in range(pca.components_.shape[1]):
    weight_percentage = np.abs(pca.components_[:, i]).sum() / total_variance
    print("指标 {} 的权重百分比: {:.2f}%".format(i+1, weight_percentage * 100))