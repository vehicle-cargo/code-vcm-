# 输入数据

# 特征根 主成分携带的信息量 即方差
# 主成分贡献方法率：主成分的方差占所有方差的比例
# 特征向量  单位特征向量
import numpy as np
import pandas as pd

# 读入数据
csv_file = 'iris_2.csv'  # T,Z,L,t,s,d
X = pd.read_csv(csv_file)
X = np.array(X)

# 数据预处理：标准化
mu = np.mean(X, axis=0)
X_norm = (X - mu) / np.std(X, axis=0)
# 计算协方差矩阵
covMat = np.cov(X_norm, rowvar=False)

# 奇异值分解，U为特征向量，S为特征值
_, S, U = np.linalg.svd(covMat)
print(S)

# 确定主成分个数，也就是k值，一般要利用信息的80%以上
lambda_ = S  # 特征值
# print('累积信息占比：', np.cumsum(lambda_) / np.sum(lambda_))
k = np.nonzero(np.cumsum(lambda_) / np.sum(lambda_) > 0.8)[0][-1] + 1
print(k)
# 成分矩阵
u = U[:, :k]
print(u.shape)
# 各个指标在主成分中的综合重要度
a = np.zeros(X.shape[1])
for i in range(k):
    a = a + lambda_[i] / np.sum(lambda_[:k]) * u[:, i]
print(a.shape)

qz = np.dot(X, a) / np.sum(np.dot(X, a))
print('属性权重为：', qz)
