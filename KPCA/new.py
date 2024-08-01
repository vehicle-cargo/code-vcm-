import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel

# 原始数据矩阵
data = np.array([
    [3, 4, 3, 4, 3, 4],
    [2, 3, 2, 5, 3, 4],
    [5, 3, 4, 5, 3, 5],
    [3, 2, 3, 2, 4, 5],
    [4, 5, 3, 2, 3, 4],
    [3, 5, 2, 4, 3, 4]
])

# 去中心化数据
scaler = StandardScaler(with_mean=True, with_std=False)
data_centered = scaler.fit_transform(data)

# PCA分析
pca = PCA()
pca.fit(data_centered)
pca_variances = pca.explained_variance_ratio_

# 计算累计方差贡献率
cumulative_variances_pca = np.cumsum(pca_variances)
print("PCA 累计方差贡献率:", cumulative_variances_pca)

# SVD分析
U, S, VT = np.linalg.svd(data_centered)
svd_variances = (S**2) / np.sum(S**2)

# 计算累计方差贡献率
cumulative_variances_svd = np.cumsum(svd_variances)
print("SVD 累计方差贡献率:", cumulative_variances_svd)

# KPCA分析
kpca = KernelPCA(kernel="rbf")
data_kpca = kpca.fit_transform(data_centered)

# 计算核矩阵
K = rbf_kernel(data_centered)

# 中心化核矩阵
N = K.shape[0]
one_n = np.ones((N, N)) / N
K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n

# 计算特征值
eigenvalues = np.linalg.eigvalsh(K_centered)
eigenvalues = eigenvalues[::-1]  # 降序排列

# 计算累计方差贡献率
kpca_variances = eigenvalues / np.sum(eigenvalues)
cumulative_variances_kpca = np.cumsum(kpca_variances)
print("KPCA 累计方差贡献率:", cumulative_variances_kpca)

# 确定主成分个数
threshold = 0.97
num_components_pca = np.argmax(cumulative_variances_pca >= threshold) + 1
num_components_svd = np.argmax(cumulative_variances_svd >= threshold) + 1
num_components_kpca = np.argmax(cumulative_variances_kpca >= threshold) + 1

print(f"PCA 选择的主成分个数: {num_components_pca}")
print(f"SVD 选择的主成分个数: {num_components_svd}")
print(f"KPCA 选择的主成分个数: {num_components_kpca}")
