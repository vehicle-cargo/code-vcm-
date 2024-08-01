import csv
import numpy as np
from fractions import Fraction

# 读取 CSV 文件并解析数据

# 将数据转换为 NumPy 数组
matrix = np.array([
    [1, 3, 5, 7, 2, 5],
    [1/3, 1, 4, 6, 1/2, 3],
    [1/5, 1/4, 1, 5, 1/3, 2],
    [1/7, 1/6, 1/5, 1, 1/6, 1/4],
    [1/2, 2, 3, 6, 1, 4],
    [1/5, 1/3, 1/2, 4, 1/4, 1]
])
print(matrix)
# 计算每个标准的权重
n = matrix.shape[0]
eig_val, eig_vec = np.linalg.eig(matrix)
max_eig_val = max(eig_val)
max_eig_vec = eig_vec[:, list(eig_val).index(max_eig_val)]
weights = max_eig_vec / sum(max_eig_vec)

# 输出结果
for i in range(n):
    print("标准{}的权重为：{}".format(i+1, round(weights[i].real, 2)))
