import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 车货匹配折线图
# 数据
# x_1 =  [1, 2, 3, 4, 5, 6, 7, 9, 8, v10, 11, 12, 13, 14, 15, 17, 18, 16, 19, 20, 21, 25, 26, 27, 22, 23, 28, 29, 30, 31, 32, 33]  # x轴数据
# x_2= [2, 4, 5, 6, 7, 9, v10, 1, 3, 8, 11, 12, 13, 14, 17, 15, 18, 19, 20, 25, 16, 26, 27, 30]
# y1 =  [1, 7, 6, 8, 4, 3, 9, 2, 12, 15, 11, 14, 18, 5, v10, 13, 16, 26, 24, 22, 20, 21, 25, 19, 29, 28, 33, 31, 17, 34, 30, 27]  # 第一条线的y轴数据
# y2 =  [7, 8, 4, 3, 6, 2, 9, 11, 1, 12, 5, 14, 16, 15, 13, 20, 23, 17, 21, 24, 26, 18, 25, 29]  # 第二条线的y轴数据
#
# # 标注的值
# # annotations1 = ['A', 'B', 'C', 'D', 'E']  # 第一条线的标注值
# # annotations2 = [1, 2, 3, 4, 5]  # 第二条线的标注值
#
# # 绘图
# plt.plot(x_1, y1, label='KPCA-BResNet algorithm', color='blue', marker='o')  # 绘制第一条线，标签为'Line 1'，蓝色，圆形标记点
# plt.plot(x_2, y2, label='IDBN algorithm', color='red', marker='s')  # 绘制第二条线，标签为'Line 2'，红色，方形标记点
#
# # 标注值
# # for i in range(len(x_1)):
# #     plt.annotate((x_1[i], y1[i]),  ha='center')
# # for i in range(len(x_2)):
# #     plt.annotate( (x_2[i], y2[i]),ha='center')
# # 图例
# plt.legend()
# # 坐标轴标签
# plt.xlabel("Vehicle number")
# plt.ylabel("Cargo number")
# # 显示图形
# plt.show()


# # 单时间片下的匹配结果的热力图
# sin_time_data = pd.read_csv("dataset/sin_hot_data.csv", index_col=0)
# print(sin_time_data)
# # sin_time_data = sin_time_data.corr()
# plt.rcParams['font.size'] = 13
# plt.rc('font', family='Times New Roman')
# fig, ax = plt.subplots(figsize=(16, 12))
# sns.heatmap(sin_time_data,
#             annot=True, vmin=0.3, vmax=0.45, square=True, cmap="Blues")
# # 获取colorbar对象
# cbar = ax.collections[0].colorbar
# cbar.ax.tick_params(labelsize=13)
# # 添加边框
# cbar.outline.set_visible(True)
# # 设置边框的线宽和颜色
# cbar.outline.set_linewidth(2)
# cbar.outline.set_edgecolor('black')
# # 添加边框
# ax.spines['top'].set_visible(True)
# ax.spines['bottom'].set_visible(True)
# ax.spines['left'].set_visible(True)
# ax.spines['right'].set_visible(True)
#
# # 设置边框的线宽和颜色
# ax.spines['top'].set_linewidth(2)
# ax.spines['bottom'].set_linewidth(2)
# ax.spines['left'].set_linewidth(2)
# ax.spines['right'].set_linewidth(2)
#
# ax.spines['top'].set_color('black')
# ax.spines['bottom'].set_color('black')
# ax.spines['left'].set_color('black')
# ax.spines['right'].set_color('black')
# ax.collections[0].set_edgecolor("black")  # 设置边框颜色为黑色
# font2 = {'family': 'Times New Roman',
#          'weight': 'normal',
#          'size': 20,
#          }
# ax.set_ylabel('Vehicle Number', font2)
# ax.set_xlabel('Cargo Number', font2)
# x1_label = ax.get_xticklabels()
# plt.tick_params(labelsize=20)
# [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
# y1_label = ax.get_yticklabels()
# [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]
#
# plt.show()

# # 多时间片中的对比实验
# t = 0
vehicle_one_noDW = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
cargo_one_noDW = [1, 7, 6, 8, 4, 3, 9, 2, 5, 10]
vehicle_cargo_match = [0.476, 0.473, 0.46, 0.392, 0.631, 0.408, 0.398, 0.376, 0.334, 0]
plt.rc('font', family='Times New Roman')
fig, ax = plt.subplots(figsize=(12, 10))
sin_time_data = pd.DataFrame(np.zeros((10, 10)), index=sorted(vehicle_one_noDW), columns=sorted(cargo_one_noDW))
for i in range(0, len(vehicle_cargo_match)):
    sin_time_data.loc[vehicle_one_noDW[i], cargo_one_noDW[i]] = vehicle_cargo_match[i]
print(sin_time_data)
sns.heatmap(sin_time_data,
            annot=True, vmin=0, vmax=0.63, square=True, cmap="Blues", annot_kws={"size": 40})
# 获取colorbar对象
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=35)
# 添加边框
cbar.outline.set_visible(True)
# 设置边框的线宽和颜色
cbar.outline.set_linewidth(2)
cbar.outline.set_edgecolor('black')
# 添加边框
ax.spines['top'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(True)

# 设置边框的线宽和颜色
ax.spines['top'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)

ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
ax.collections[0].set_edgecolor("black")  # 设置边框颜色为黑色

# 设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=30)

# 设置横纵坐标的名称以及对应字体格式
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 40,
         }
ax.set_title('data', font2)
ax.set_ylabel('Vehicle Number', font2)
ax.set_xlabel('Cargo Number', font2)
plt.show()
# t=1
vehicle_one_noDW = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
cargo_one_noDW = [15, 11, 12, 18, 13, 10, 14, 19, 16, 17]
vehicle_cargo_match = [0.379, 0.451, 0.494, 0.442, 0.412, 0.382, 0, 0.443, 0.401, 0.365]
plt.rc('font', family='Times New Roman')
fig, ax = plt.subplots(figsize=(12, 10))
sin_time_data = pd.DataFrame(np.zeros((10, 10)), index=sorted(vehicle_one_noDW), columns=sorted(cargo_one_noDW))
for i in range(0, len(vehicle_cargo_match)):
    sin_time_data.loc[vehicle_one_noDW[i], cargo_one_noDW[i]] = vehicle_cargo_match[i]
print(sin_time_data)
sns.heatmap(sin_time_data,
            annot=True, vmin=0, vmax=0.63, square=True, cmap="Blues", annot_kws={"size": 35})
# 获取colorbar对象
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=30)

# 添加边框
cbar.outline.set_visible(True)

# 设置边框的线宽和颜色
cbar.outline.set_linewidth(2)
cbar.outline.set_edgecolor('black')
# 添加边框
ax.spines['top'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(True)

# 设置边框的线宽和颜色
ax.spines['top'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)

ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
ax.collections[0].set_edgecolor("black")  # 设置边框颜色为黑色
# 设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=30)

# 设置横纵坐标的名称以及对应字体格式
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 40,
         }

ax.set_title('data', font2)
ax.set_ylabel('Vehicle Number', font2)
ax.set_xlabel('Cargo Number', font2)
plt.show()
# t=2
vehicle_one_noDW = [16, 20, 21, 22, 23, 24, 25, 26, 27, 28]
cargo_one_noDW = [26, 24, 20, 25, 28, 23, 21, 14, 27, 22]
vehicle_cargo_match = [0.467, 0.417, 0.484, 0.352, 0.413, 0, 0.424, 0.424, 0.354, 0.321]
plt.rc('font', family='Times New Roman')
fig, ax = plt.subplots(figsize=(12, 10))
sin_time_data = pd.DataFrame(np.zeros((10, 10)), index=sorted(vehicle_one_noDW), columns=sorted(cargo_one_noDW))
for i in range(0, len(vehicle_cargo_match)):
    sin_time_data.loc[vehicle_one_noDW[i], cargo_one_noDW[i]] = vehicle_cargo_match[i]
print(sin_time_data)
sns.heatmap(sin_time_data,
            annot=True, vmin=0, vmax=0.63, square=True, cmap="Blues", annot_kws={"size": 35})
# 获取colorbar对象
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=30)

# 添加边框
cbar.outline.set_visible(True)

# 设置边框的线宽和颜色
cbar.outline.set_linewidth(2)
cbar.outline.set_edgecolor('black')
# 添加边框
ax.spines['top'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(True)

# 设置边框的线宽和颜色
ax.spines['top'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)

ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
ax.collections[0].set_edgecolor("black")  # 设置边框颜色为黑色
# 设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=30)

# 设置横纵坐标的名称以及对应字体格式
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 40,
         }

ax.set_title('data', font2)
ax.set_ylabel('Vehicle Number', font2)
ax.set_xlabel('Cargo Number', font2)
plt.show()
# t=3
vehicle_one_noDW = [24, 29, 30, 31, 32, 33, 34, 35, 36, 37]
cargo_one_noDW = [23, 31, 29, 35, 30, 36, 33, 34, 32, 37]
vehicle_cargo_match = [0, 0.377, 0.528, 0.434, 0.437, 0.436, 0.395, 0, 0.372, 0]
plt.rc('font', family='Times New Roman')
fig, ax = plt.subplots(figsize=(12, 10))
sin_time_data = pd.DataFrame(np.zeros((10, 10)), index=sorted(vehicle_one_noDW), columns=sorted(cargo_one_noDW))
for i in range(0, len(vehicle_cargo_match)):
    sin_time_data.loc[vehicle_one_noDW[i], cargo_one_noDW[i]] = vehicle_cargo_match[i]
print(sin_time_data)
sns.heatmap(sin_time_data,
            annot=True, vmin=0, vmax=0.63, square=True, cmap="Blues", annot_kws={"size": 35})
# 获取colorbar对象
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=30)

# 添加边框
cbar.outline.set_visible(True)

# 设置边框的线宽和颜色
cbar.outline.set_linewidth(2)
cbar.outline.set_edgecolor('black')
# 添加边框
ax.spines['top'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(True)

# 设置边框的线宽和颜色
ax.spines['top'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)

ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
ax.collections[0].set_edgecolor("black")  # 设置边框颜色为黑色
# 设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=30)

# 设置横纵坐标的名称以及对应字体格式
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 40,
         }

ax.set_title('data', font2)
ax.set_ylabel('Vehicle Number', font2)
ax.set_xlabel('Cargo Number', font2)
plt.show()
# #idbn
# # # t=0
# vehicle_one_noDW = [1, 2, 4, 6, 7, 8, 10]
# cargo_one_noDW = [7, 6, 9, 1, 5, 10, 4]
# vehicle_cargo_match = [0.325, 0.277, 0.045, 0.217, 0.06, 0.2, 0.064]
# plt.rc('font', family='Times New Roman')
# fig, ax = plt.subplots(figsize=(12, 10))
# sin_time_data = pd.DataFrame(np.zeros((7, 7)), index=sorted(vehicle_one_noDW), columns=sorted(cargo_one_noDW))
# for i in range(0, len(vehicle_cargo_match)):
#     sin_time_data.loc[vehicle_one_noDW[i], cargo_one_noDW[i]] = vehicle_cargo_match[i]
# print(sin_time_data)
# sns.heatmap(sin_time_data,
#             annot=True, vmin=0, vmax=0.3, square=True, cmap="Blues", annot_kws={"size": 35})
# # 获取colorbar对象
# cbar = ax.collections[0].colorbar
# cbar.ax.tick_params(labelsize=30)
#
# # 添加边框
# cbar.outline.set_visible(True)
#
# # 设置边框的线宽和颜色
# cbar.outline.set_linewidth(2)
# cbar.outline.set_edgecolor('black')
# # 添加边框
# ax.spines['top'].set_visible(True)
# ax.spines['bottom'].set_visible(True)
# ax.spines['left'].set_visible(True)
# ax.spines['right'].set_visible(True)
#
# # 设置边框的线宽和颜色
# ax.spines['top'].set_linewidth(2)
# ax.spines['bottom'].set_linewidth(2)
# ax.spines['left'].set_linewidth(2)
# ax.spines['right'].set_linewidth(2)
#
# ax.spines['top'].set_color('black')
# ax.spines['bottom'].set_color('black')
# ax.spines['left'].set_color('black')
# ax.spines['right'].set_color('black')
# ax.collections[0].set_edgecolor("black")  # 设置边框颜色为黑色
# # 设置坐标刻度值的大小以及刻度值的字体
# plt.tick_params(labelsize=30)
#
# # 设置横纵坐标的名称以及对应字体格式
# font2 = {'family': 'Times New Roman',
#          'weight': 'normal',
#          'size': 40,
#          }
#
# ax.set_title('IDBN', font2)
# ax.set_ylabel('Vehicle Number', font2)
# ax.set_xlabel('Cargo Number', font2)
# plt.show()
# #
# #
# # t=1
# vehicle_one_noDW = [5, 9, 11, 12, 13, 14, 15, 17]
# cargo_one_noDW = [14, 11, 3, 16, 17, 2, 13, 8]
# vehicle_cargo_match = [0.112, 0.167, 0.316, 0.078, 0.196, 0.23, 0.079, 0.247]
# plt.rc('font', family='Times New Roman')
# fig, ax = plt.subplots(figsize=(12, 10))
# sin_time_data = pd.DataFrame(np.zeros((8, 8)), index=sorted(vehicle_one_noDW), columns=sorted(cargo_one_noDW))
# for i in range(0, len(vehicle_cargo_match)):
#     sin_time_data.loc[vehicle_one_noDW[i], cargo_one_noDW[i]] = vehicle_cargo_match[i]
# print(sin_time_data)
# sns.heatmap(sin_time_data,
#             annot=True, vmin=0, vmax=0.3, square=True, cmap="Blues", annot_kws={"size": 35})
# # 获取colorbar对象
# cbar = ax.collections[0].colorbar
# cbar.ax.tick_params(labelsize=30)
#
# # 添加边框
# cbar.outline.set_visible(True)
#
# # 设置边框的线宽和颜色
# cbar.outline.set_linewidth(2)
# cbar.outline.set_edgecolor('black')
# # 添加边框
# ax.spines['top'].set_visible(True)
# ax.spines['bottom'].set_visible(True)
# ax.spines['left'].set_visible(True)
# ax.spines['right'].set_visible(True)
#
# # 设置边框的线宽和颜色
# ax.spines['top'].set_linewidth(2)
# ax.spines['bottom'].set_linewidth(2)
# ax.spines['left'].set_linewidth(2)
# ax.spines['right'].set_linewidth(2)
#
# ax.spines['top'].set_color('black')
# ax.spines['bottom'].set_color('black')
# ax.spines['left'].set_color('black')
# ax.spines['right'].set_color('black')
# ax.collections[0].set_edgecolor("black")  # 设置边框颜色为黑色
# # 设置坐标刻度值的大小以及刻度值的字体
# plt.tick_params(labelsize=30)
#
# # 设置横纵坐标的名称以及对应字体格式
# font2 = {'family': 'Times New Roman',
#          'weight': 'normal',
#          'size': 40,
#          }
#
# ax.set_title('IDBN', font2)
# ax.set_ylabel('Vehicle Number', font2)
# ax.set_xlabel('Cargo Number', font2)
# plt.show()
# #
# # t=2
# vehicle_one_noDW = [3, 16, 18, 19, 20, 21, 22, 23]
# cargo_one_noDW = [18, 19, 22, 12, 21, 24, 25, 15]
# vehicle_cargo_match = [0.09, 0.06, 0.08, 0.23, 0.03, 0.09, 0.09, 0.09]
# plt.rc('font', family='Times New Roman')
# fig, ax = plt.subplots(figsize=(12, 10))
# sin_time_data = pd.DataFrame(np.zeros((8, 8)), index=sorted(vehicle_one_noDW), columns=sorted(cargo_one_noDW))
# for i in range(0, len(vehicle_cargo_match)):
#     sin_time_data.loc[vehicle_one_noDW[i], cargo_one_noDW[i]] = vehicle_cargo_match[i]
# print(sin_time_data)
# sns.heatmap(sin_time_data,
#             annot=True, vmin=0, vmax=0.3, square=True, cmap="Blues", annot_kws={"size": 35})
# # 获取colorbar对象
# cbar = ax.collections[0].colorbar
# cbar.ax.tick_params(labelsize=30)
#
# # 添加边框
# cbar.outline.set_visible(True)
#
# # 设置边框的线宽和颜色
# cbar.outline.set_linewidth(2)
# cbar.outline.set_edgecolor('black')
# # 添加边框
# ax.spines['top'].set_visible(True)
# ax.spines['bottom'].set_visible(True)
# ax.spines['left'].set_visible(True)
# ax.spines['right'].set_visible(True)
#
# # 设置边框的线宽和颜色
# ax.spines['top'].set_linewidth(2)
# ax.spines['bottom'].set_linewidth(2)
# ax.spines['left'].set_linewidth(2)
# ax.spines['right'].set_linewidth(2)
#
# ax.spines['top'].set_color('black')
# ax.spines['bottom'].set_color('black')
# ax.spines['left'].set_color('black')
# ax.spines['right'].set_color('black')
# ax.collections[0].set_edgecolor("black")  # 设置边框颜色为黑色
# # 设置坐标刻度值的大小以及刻度值的字体
# plt.tick_params(labelsize=30)
#
# # 设置横纵坐标的名称以及对应字体格式
# font2 = {'family': 'Times New Roman',
#          'weight': 'normal',
#          'size': 40,
#          }
#
# ax.set_title('IDBN', font2)
# ax.set_ylabel('Vehicle Number', font2)
# ax.set_xlabel('Cargo Number', font2)
# plt.show()
# # t=3
# vehicle_one_noDW = [24, 25, 26, 27, 28, 29, 30, 32]
# cargo_one_noDW = [31, 29, 33, 32, 27, 26, 28, 30]
# vehicle_cargo_match = [0.21, 0.149, 0.144, 0.254, 0.083, 0.293, 0.126, 0.322]
# plt.rc('font', family='Times New Roman')
# fig, ax = plt.subplots(figsize=(12, 10))
# sin_time_data = pd.DataFrame(np.zeros((8, 8)), index=sorted(vehicle_one_noDW), columns=sorted(cargo_one_noDW))
# for i in range(0, len(vehicle_cargo_match)):
#     sin_time_data.loc[vehicle_one_noDW[i], cargo_one_noDW[i]] = vehicle_cargo_match[i]
# print(sin_time_data)
# sns.heatmap(sin_time_data,
#             annot=True, vmin=0, vmax=0.3, square=True, cmap="Blues", annot_kws={"size": 35})
# # 获取colorbar对象
# cbar = ax.collections[0].colorbar
# cbar.ax.tick_params(labelsize=30)
#
# # 添加边框
# cbar.outline.set_visible(True)
#
# # 设置边框的线宽和颜色
# cbar.outline.set_linewidth(2)
# cbar.outline.set_edgecolor('black')
# # 添加边框
# ax.spines['top'].set_visible(True)
# ax.spines['bottom'].set_visible(True)
# ax.spines['left'].set_visible(True)
# ax.spines['right'].set_visible(True)
#
# # 设置边框的线宽和颜色
# ax.spines['top'].set_linewidth(2)
# ax.spines['bottom'].set_linewidth(2)
# ax.spines['left'].set_linewidth(2)
# ax.spines['right'].set_linewidth(2)
#
# ax.spines['top'].set_color('black')
# ax.spines['bottom'].set_color('black')
# ax.spines['left'].set_color('black')
# ax.spines['right'].set_color('black')
# ax.collections[0].set_edgecolor("black")  # 设置边框颜色为黑色
# # 设置坐标刻度值的大小以及刻度值的字体
# plt.tick_params(labelsize=30)
#
# # 设置横纵坐标的名称以及对应字体格式
# font2 = {'family': 'Times New Roman',
#          'weight': 'normal',
#          'size': 40,
#          }
#
# ax.set_title('IDBN', font2)
# ax.set_ylabel('Vehicle Number', font2)
# ax.set_xlabel('Cargo Number', font2)
# plt.show()
#
# # # kpcadata-与传统算法-成功率指标对比试验
# plt.rcParams['font.sans-serif'] = ['SimHei']
# # 运行配置参数中的字体为黑体
# plt.rcParams['axes.unicode_minus'] = False
#
# labels = ['t0', 't1', 't2', 't3']
# err_attr = {"elinewidth": 1.5, "color": "black", "capsize": 6}
# cargo_means_kpca_weight = [0.80, 0.90, 0.73, 0.73]
# # kpca_err = [0, 0, 0.07, 0.07]
# cargo_means_pca_weight = [0.73, 0.87, 0.70, 0.70]
# # pca_err = [0.03, 0.03, 0.07, 0]
# vcm_means_random_weight = [0.60, 0.60, 0.67, 0.60]
# # ra_err = [0, 0.1, 0.03, 0]
#
# x = np.arange(len(labels))  # the label locations
# width = 0.25  # the width of the bars
#
# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width, cargo_means_kpca_weight, width, label='KPCA-BResNet')
# rects2 = ax.bar(x, cargo_means_pca_weight, width, label='IDBN')
# rects3 = ax.bar(x + width, vcm_means_random_weight, width, label='Contrast')
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Success rate', fontsize=20)
# ax.set_xticks(x)
# ax.set_xlabel('Time slice', fontsize=20)
# ax.set_xticklabels(labels)
#
# ax.legend(bbox_to_anchor=(0.625, 0.765), loc=3, borderaxespad=0, fontsize=15)
# # def autolabel(rects):
# #     """Attach a text label above each bar in *rects*, displaying its height."""
# #     for rect in rects:
# #         height = rect.get_height()
# #         ax.annotate('{}'.format(height),
# #                     xy=(rect.get_x() + rect.get_width() / 2, height),
# #                     xytext=(0, 3),  # 3 points vertical offset
# #                     textcoords="offset points",
# #                     ha='center', va='bottom')
# #
# #
# # autolabel(rects1)
# # autolabel(rects2)
# # autolabel(rects3)
# plt.ylim(0, 1.1)
# fig.tight_layout()
# plt.xticks(fontsize=15)  # 调整刻度标签字号为12
# # 设置y轴标签字号
# plt.yticks(fontsize=15)  # 调整刻度标签字号为12
# plt.show()
# #
# #
# # #
# #
# #
# # # # kpc——adata 与其他算法-当前时间片未匹配成功 下个时间片匹配成功得匹配成功率
# # plt.rcParams['axes.unicode_minus'] = False
# #
# # labels = ['t0', 't1', 't2', 't3']
# # err_attr = {"elinewidth": 1.5, "color": "black", "capsize": 6}
# # cargo_means_kpca_weight = [0, 1, 1, 0.78]
# # # kpca2_err = [0, 0, 0, 0.03]
# # cargo_means_pca_weight = [0, 0.83, 0.83, 0.50]
# # # pca2_err = [0, 0.13, 0.1, 0.1]
# # vcm_means_random_weight = [0, 0.50, 0.56, 0.41]
# # # r2_err = [0, 0.25, 0.2, 0.1]
# # x = np.arange(len(labels))  # the label locations
# # width = 0.25  # the width of the bars
# # fig, ax = plt.subplots()
# # rects1 = ax.bar(x - width, cargo_means_kpca_weight, width, label='KPCA-BResNet')
# # rects2 = ax.bar(x, cargo_means_pca_weight, width, label='IDBN ')
# # rects3 = ax.bar(x + width, vcm_means_random_weight, width, label='Contrast')
# #
# # # Add some text for labels, title and custom x-axis tick labels, etc.
# # ax.set_ylabel('Rematch success rate', fontsize=20)
# # ax.set_xticks(x)
# # ax.set_xlabel('Time slice', fontsize=20)
# # ax.set_xticklabels(labels)
# # ax.legend(bbox_to_anchor=(0.625, 0.765), loc=3, borderaxespad=0, fontsize=15)
# #
# # # def autolabel(rects):
# # #     """Attach a text label above each bar in *rects*, displaying its height."""
# # #     for rect in rects:
# # #         height = rect.get_height()
# # #         ax.annotate('{}'.format(height),
# # #                     xy=(rect.get_x() + rect.get_width() / 2, height),
# # #                     xytext=(0, 3),  # 3 points vertical offset
# # #                     textcoords="offset points",
# # #                     ha='center', va='bottom')
# # #
# # #
# # # autolabel(rects1)
# # # autolabel(rects2)
# # # autolabel(rects3)
# # plt.ylim(0, 1.1)
# # fig.tight_layout()
# # plt.xticks(fontsize=15)  # 调整刻度标签字号为12
# # # 设置y轴标签字号
# # plt.yticks(fontsize=15)  # 调整刻度标签字号为12
# # plt.show()
# #
# #
# #
# #
# # # kpcadata-与idbn和传统算法对比-用户满意度
# # plt.rcParams['font.sans-serif'] = ['SimHei']
# # # 运行配置参数中的字体为黑体
# # plt.rcParams['axes.unicode_minus'] = False
# # labels = ['t0', 't1', 't2', 't3']
# # err_attr = {"elinewidth": 1.5, "color": "black", "capsize": 6}
# # cargo_means_kpca_weight = [0.60, 0.63, 0.54, 0.55]
# # # kpca_err = [0, 0.005, 0.01, 0.07]
# # cargo_means_pca_weight = [0.53, 0.58, 0.51, 0.49]
# # # pca_err = [0.03, 0.07, 0.05, 0.05]
# # vcm_means_random_weight = [0.46, 0.45, 0.48, 0.45]
# # # rwa_err = [0.1, 0.05, 0.02, 0.01]
# #
# # x = np.arange(len(labels))  # the label locations
# # width = 0.25  # the width of the bars
# #
# # fig, ax = plt.subplots()
# # rects1 = ax.bar(x - width, cargo_means_kpca_weight, width, label='KPCA-BResNet')
# # rects2 = ax.bar(x, cargo_means_pca_weight, width, label='IDBN')
# # rects3 = ax.bar(x + width, vcm_means_random_weight, width, label='Contrast')
# #
# # # Add some text for labels, title and custom x-axis tick labels, etc.
# # ax.set_ylabel('Owners satisfaction', fontsize=20)
# # ax.set_xticks(x)
# # ax.set_xlabel('Time slice', fontsize=20)
# # ax.set_xticklabels(labels)
# #
# # ax.legend(bbox_to_anchor=(0.625, 0.765), loc=3, borderaxespad=0, fontsize=15)
# #
# # # def autolabel(rects):
# # #     """Attach a text label above each bar in *rects*, displaying its height."""
# # #     for rect in rects:
# # #         height = rect.get_height()
# # #         ax.annotate('{}'.format(height),
# # #                     xy=(rect.get_x() + rect.get_width() / 2, height),
# # #                     xytext=(0, 3),  # 3 points vertical offset
# # #                     textcoords="offset points",
# # #                     ha='center', va='bottom')
# # #
# # #
# # # autolabel(rects1)
# # # autolabel(rects2)
# # # autolabel(rects3)
# # plt.ylim(0, 0.8)
# # fig.tight_layout()
# # plt.xticks(fontsize=15)  # 调整刻度标签字号为12
# # # 设置y轴标签字号
# # plt.yticks(fontsize=15)  # 调整刻度标签字号为12
# # plt.show()
# #
# #
# #
# #
# # # idbn-data 成功率指标
# # plt.rcParams['font.sans-serif'] = ['SimHei']
# # # 运行配置参数中的字体为黑体
# # plt.rcParams['axes.unicode_minus'] = False
# #
# # labels = ['t0', 't1', 't2', 't3']
# # err_attr = {"elinewidth": 1.5, "color": "black", "capsize": 6}
# #
# # cargo_means_kpca_weight = [0.73, 0.77, 0.8, 0.73]
# # # kpca_err = [0.03, 0.03, 0.1, 0.07]
# # cargo_means_pca_weight = [0.70, 0.77, 0.77, 0.70]
# # # pca_err = [0.05, 0.03, 0.07, 0.07]
# # vcm_means_random_weight = [0.67, 0.70, 0.57, 0.60]
# # # rwa_err = [0.07, 0, 0.07, 0.1]
# #
# # x = np.arange(len(labels))  # the label locations
# # width = 0.25  # the width of the bars
# #
# # fig, ax = plt.subplots()
# # rects1 = ax.bar(x - width, cargo_means_kpca_weight, width, label='KPCA-BResNet')
# # rects2 = ax.bar(x, cargo_means_pca_weight, width, label='IDBN')
# # rects3 = ax.bar(x + width, vcm_means_random_weight, width, label='Contrast')
# #
# # # Add some text for labels, title and custom x-axis tick labels, etc.
# # ax.set_ylabel('Success rate', fontsize=20)
# # ax.set_xticks(x)
# # ax.set_xlabel('Time slice', fontsize=20)
# # ax.set_xticklabels(labels)
# #
# # ax.legend(bbox_to_anchor=(0.625, 0.765), loc=3, borderaxespad=0, fontsize=15)
# #
# # # def autolabel(rects):
# # #     """Attach a text label above each bar in *rects*, displaying its height."""
# # #     for rect in rects:
# # #         height = rect.get_height()
# # #         ax.annotate('{}'.format(height),
# # #                     xy=(rect.get_x() + rect.get_width() / 2, height),
# # #                     xytext=(0, 3),  # 3 points vertical offset
# # #                     textcoords="offset points",
# # #                     ha='center', va='bottom')
# # #
# # #
# # # autolabel(rects1)
# # # autolabel(rects2)
# # # autolabel(rects3)
# # plt.ylim(0, 1.1)
# # fig.tight_layout()
# # plt.xticks(fontsize=15)  # 调整刻度标签字号为12
# # # 设置y轴标签字号
# # plt.yticks(fontsize=15)  # 调整刻度标签字号为12
# # plt.show()
# #
# #
# #
# #
# # #  # 当前时间片匹配失败，下个时间片继续匹配成功得匹配成功率
# # plt.rcParams['axes.unicode_minus'] = False
# # labels = ['t0', 't1', 't2', 't3']
# # err_attr = {"elinewidth": 1.5, "color": "black", "capsize": 6}
# # cargo_means_kpca_weight = [0, 0.67, 0.56, 0.89]
# # # kpca_err = [0, 0.01, 0.06, 0.1]
# # cargo_means_pca_weight = [0, 0.58, 0.44, 0.67]
# # # pca_err = [0, 0.08, 0.1, 0.2]
# # vcm_means_random_weight = [0, 0.50, 0.33, 0.45]
# # # rwa_err = [0, 0, 0.01, 0.05]
# #
# # x = np.arange(len(labels))  # the label locations
# # width = 0.25  # the width of the bars
# #
# # fig, ax = plt.subplots()
# # rects1 = ax.bar(x - width, cargo_means_kpca_weight, width, label='KPCA-BResNet')
# # rects2 = ax.bar(x, cargo_means_pca_weight, width, label='IDBN')
# # rects3 = ax.bar(x + width, vcm_means_random_weight, width, label='Contrast')
# #
# # # Add some text for labels, title and custom x-axis tick labels, etc.
# # ax.set_ylabel('Rematch success rate', fontsize=20)
# # ax.set_xticks(x)
# # ax.set_xlabel('Time slice', fontsize=20)
# # ax.set_xticklabels(labels)
# # ax.legend(bbox_to_anchor=(0.625, 0.765), loc=3, borderaxespad=0, fontsize=15)
# #
# # # def autolabel(rects):
# # #     """Attach a text label above each bar in *rects*, displaying its height."""
# # #     for rect in rects:
# # #         height = rect.get_height()
# # #         ax.annotate('{}'.format(height),
# # #                     xy=(rect.get_x() + rect.get_width() / 2, height),
# # #                     xytext=(0, 3),  # 3 points vertical offset
# # #                     textcoords="offset points",
# # #                     ha='center', va='bottom')
# # #
# # #
# # # autolabel(rects1)
# # # autolabel(rects2)
# # # autolabel(rects3)
# # plt.ylim(0, 1.2)
# # fig.tight_layout()
# # plt.xticks(fontsize=15)  # 调整刻度标签字号为12
# # # 设置y轴标签字号
# # plt.yticks(fontsize=15)  # 调整刻度标签字号为12
# # plt.show()
# #
# #
# # plt.rcParams['font.sans-serif'] = ['SimHei']
# # # 运行配置参数中的字体为黑体
# # plt.rcParams['axes.unicode_minus'] = False
# #
# # # vcm_means_random_weight = [0.67, 0.70, 0.57, 0.60]
# # labels = ['t0', 't1', 't2', 't3']
# # err_attr = {"elinewidth": 1.5, "color": "black", "capsize": 6}
# # cargo_means_kpca_weight = [0.40, 0.44, 0.43, 0.40]  # 0.08 0.03 0.05 0.05
# # # kpca_err = [0.02, 0.05, 0.05, 0.01]
# # cargo_means_pca_weight = [0.37, 0.41, 0.41, 0.38]  # 0.05 0.06 0.1 0.1
# # # pca_err = [0.05, 0.05, 0.05, 0.02]
# # vcm_means_random_weight = [0.35, 0.38, 0.30, 0.32]  # 0.05 0.005 0.05 0.06
# # # rwa_err = [0.05, 0.05, 0.02, 0.05]
# #
# # # x = np.arange(len(labels))  # the label locations
# # width = 0.25  # the width of the bars
# #
# # fig, ax = plt.subplots()
# # rects1 = ax.bar(x - width, cargo_means_kpca_weight, width, label='KPCA-BResNet')
# # rects2 = ax.bar(x, cargo_means_pca_weight, width, label='IDBN')
# # rects3 = ax.bar(x + width, vcm_means_random_weight, width, label='Contrast')
# #
# # # Add some text for labels, title and custom x-axis tick labels, etc.
# # ax.set_ylabel('Owners satisfaction', fontsize=20)
# # ax.set_xticks(x)
# # ax.set_xlabel('Time slice', fontsize=20)
# # ax.set_xticklabels(labels)
# #
# # ax.legend(bbox_to_anchor=(0.625, 0.765), loc=3, borderaxespad=0, fontsize=15)
# #
# # # def autolabel(rects):
# # #     """Attach a text label above each bar in *rects*, displaying its height."""
# # #     for rect in rects:
# # #         height = rect.get_height()
# # #         ax.annotate('{}'.format(height),
# # #                     xy=(rect.get_x() + rect.get_width() / 2, height),
# # #                     xytext=(0, 3),  # 3 points vertical offset
# # #                     textcoords="offset points",
# # #                     ha='center', va='bottom')
# # #
# # #
# # # autolabel(rects1)
# # # autolabel(rects2)
# # # autolabel(rects3)
# # plt.ylim(0, 0.65)
# # fig.tight_layout()
# # plt.xticks(fontsize=15)  # 调整刻度标签字号为12
# # # 设置y轴标签字号
# # plt.yticks(fontsize=15)  # 调整刻度标签字号为12
# # plt.show()
# #
# # # 权重分配方法——匹配成功率指标
# # plt.rcParams['font.sans-serif'] = ['SimHei']
# # # 运行配置参数中的字体为黑体
# # plt.rcParams['axes.unicode_minus'] = False
# # labels = ['t0', 't1', 't2', 't3']
# # err_attr = {"elinewidth": 1.5, "color": "black", "capsize": 6}
# # cargo_means_kpca_weight = [0.8, 0.9, 0.73, 0.73]0 0
# # cargo_means_pca_weight = [0.77, 0.73, 0.70, 0.63]
# # vcm_means_random_weight = [0.43, 0.57, 0.50, 0.53]
# # x = np.arange(len(labels))  # the label locations
# # width = 0.25  # the width of the bars
# # fig, ax = plt.subplots()
# # rects1 = ax.bar(x - width, cargo_means_kpca_weight, width, label='KPCA-BResNet')
# # rects2 = ax.bar(x, cargo_means_pca_weight, width, label='KPCA-BResNet')
# # rects3 = ax.bar(x + width, vcm_means_random_weight, width, label='RWA-BCNN')
# # # Add some text for labels, title and custom x-axis tick labels, etc.
# # ax.set_ylabel('Success rate',fontsize=20)
# # ax.set_xticks(x)
# # ax.set_xlabel('Time slice',fontsize=20)
# # ax.set_title('KPCA-BresNet dataset', fontsize=20)
# # ax.legend(bbox_to_anchor=(0.625, 0.765), loc=3, borderaxespad=0,fontsize=15)
# # # def autolabel(rects):
# # #     """Attach a text label above each bar in *rects*, displaying its height."""
# # #     for rect in rects:
# # #         height = rect.get_height()
# # #         ax.annotate('{}'.format(height),
# # #                     xy=(rect.get_x() + rect.get_width() / 2, height),
# # #                     xytext=(0, 3),  # 3 points vertical offset
# # #                     textcoords="offset points",
# # #                     ha='center', va='bottom')
# # #
# # #
# # # autolabel(rects1)
# # # autolabel(rects2)
# # # autolabel(rects3)
# # plt.ylim(0, 1.2)
# # fig.tight_layout()
# # plt.xticks(fontsize=15)  # 调整刻度标签字号为12
# # # 设置y轴标签字号
# # plt.yticks(fontsize=15)  # 调整刻度标签字号为12
# #
# # plt.show()
# # #
# #
# #
# # # 权重分配 当前时间片匹配失败 下个时间片匹配成功得匹配成功率
# # plt.rcParams['font.sans-serif'] = ['SimHei']
# # # 运行配置参数中的字体为黑体
# # plt.rcParams['axes.unicode_minus'] = False
# # labels = ['t0', 't1', 't2', 't3']
# # err_attr = {"elinewidth": 1.5, "color": "black", "capsize": 6}
# # cargo_means_kpca_weight = [0, 1, 1, 0.78]
# # cargo_means_pca_weight = [0, 1, 0.58, 0.55]
# # vcm_means_random_weight = [0, 0.4, 0.24, 0.41]
# # x = np.arange(len(labels))  # the label locations
# # width = 0.25  # the width of the bars
# #
# # fig, ax = plt.subplots()
# # rects1 = ax.bar(x - width, cargo_means_kpca_weight, width, label='KPCA-BResNet')
# # rects2 = ax.bar(x, cargo_means_pca_weight, width, label='KPCA-BResNet')
# # rects3 = ax.bar(x + width, vcm_means_random_weight, width, label='RWA-BCNN')
# #
# # # Add some text for labels, title and custom x-axis tick labels, etc.
# # ax.set_ylabel('Rematch success rate',fontsize=20)
# # ax.set_xticks(x)
# # ax.set_xlabel('Time slice',fontsize=20)
# # ax.set_xticklabels(labels)
# # ax.legend(bbox_to_anchor=(0.625, 0.765), loc=3, borderaxespad=0,fontsize=15)
# # # def autolabel(rects):
# # #     """Attach a text label above each bar in *rects*, displaying its height."""
# # #     for rect in rects:
# # #         height = rect.get_height()
# # #         ax.annotate('{}'.format(height),
# # #                     xy=(rect.get_x() + rect.get_width() / 2, height),
# # #                     xytext=(0, 3),  # 3 points vertical offset
# # #                     textcoords="offset points",
# # #                     ha='center', va='bottom')
# # #
# # #
# # # autolabel(rects1)
# # # autolabel(rects2)
# # # autolabel(rects3)
# #
# # fig.tight_layout()
# # plt.ylim(0, 1.1)
# # plt.xticks(fontsize=15)  # 调整刻度标签字号为12
# # # 设置y轴标签字号
# # plt.yticks(fontsize=15)  # 调整刻度标签字号为12
# # plt.show()
# # # 权重分配-用户满意度指标对比试验
# # plt.rcParams['font.sans-serif'] = ['SimHei']
# # # 运行配置参数中的字体为黑体
# # plt.rcParams['axes.unicode_minus'] = False
# #
# # labels = ['t0', 't1', 't2', 't3']
# # # err_attr = {"elinewidth": 1.5, "color": "black", "capsize": 6}
# # cargo_means_kpca_weight = [0.60, 0.63, 0.54, 0.55]
# #
# # cargo_means_pca_weight = [0.47, 0.45, 0.43, 0.40]
# #
# # vcm_means_random_weight = [0.30, 0.36, 0.34, 0.35]
# #
# # x = np.arange(len(labels))  # the label locations
# # width = 0.25  # the width of the bars
# #
# # fig, ax = plt.subplots()
# # rects1 = ax.bar(x - width, cargo_means_kpca_weight, width, label='KPCA-BResNet')
# # rects2 = ax.bar(x, cargo_means_pca_weight, width, label='KPCA-BResNet')
# # rects3 = ax.bar(x + width, vcm_means_random_weight, width, label='RWA-BCNN')
# #
# # # Add some text for labels, title and custom x-axis tick labels, etc.
# # ax.set_ylabel('Owners satisfaction',fontsize=20)
# # ax.set_xticks(x)
# # ax.set_xlabel('Time slice',fontsize=20)
# # ax.set_xticklabels(labels)
# # ax.legend(bbox_to_anchor=(0.625, 0.765), loc=3, borderaxespad=0,fontsize=15)
# # # def autolabel(rects):
# # #     """Attach a text label above each bar in *rects*, displaying its height."""
# # #     for rect in rects:
# # #         height = rect.get_height()
# # #         ax.annotate('{}'.format(height),
# # #                     xy=(rect.get_x() + rect.get_width() / 2, height),
# # #                     xytext=(0, 3),  # 3 points vertical offset
# # #                     textcoords="offset points",
# # #                     ha='center', va='bottom')
# # #
# # #
# # # autolabel(rects1)
# # # autolabel(rects2)
# # # autolabel(rects3)
# # fig.tight_layout()
# # plt.ylim(0, 0.8)
# # plt.xticks(fontsize=15)  # 调整刻度标签字号为12
# # # 设置y轴标签字号
# # plt.yticks(fontsize=15)  # 调整刻度标签字号为12
# # plt.show()
# # t=0
vehicle_one_noDW = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
cargo_one_noDW = [1, 2, 5, 3, 7, 8, 6, 4, 10, 9]
vehicle_cargo_match = [0.214, 0, 0, 0.118, 0, 0.447, 0.132, 0.193, 0, 0.113]
plt.rc('font', family='Times New Roman')
fig, ax = plt.subplots(figsize=(12, 10))
sin_time_data = pd.DataFrame(np.zeros((10, 10)), index=sorted(vehicle_one_noDW), columns=sorted(cargo_one_noDW))
for i in range(0, len(vehicle_cargo_match)):
    sin_time_data.loc[vehicle_one_noDW[i], cargo_one_noDW[i]] = vehicle_cargo_match[i]
print(sin_time_data)
sns.heatmap(sin_time_data,
            annot=True, vmin=0, vmax=0.6, square=True, cmap="Blues", annot_kws={"size": 35})
# 获取colorbar对象
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=30)

# 添加边框
cbar.outline.set_visible(True)

# 设置边框的线宽和颜色
cbar.outline.set_linewidth(2)
cbar.outline.set_edgecolor('black')
# 添加边框
ax.spines['top'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(True)

# 设置边框的线宽和颜色
ax.spines['top'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)

ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
ax.collections[0].set_edgecolor("black")  # 设置边框颜色为黑色
# 设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=30)

# 设置横纵坐标的名称以及对应字体格式
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 40,
         }

ax.set_title('IDBN', font2)
ax.set_ylabel('Vehicle Number', font2)
ax.set_xlabel('Cargo Number', font2)
plt.show()
#
#
# t=1
vehicle_one_noDW = [2, 3, 5, 9, 11, 12, 13, 14, 15, 16]
cargo_one_noDW = [13, 15, 10, 11, 2, 5, 7, 12, 14, 16]
vehicle_cargo_match = [0.247, 0.215, 0.198, 0.168, 0, 0.138, 0.27, 0, 0, 0]
plt.rc('font', family='Times New Roman')
fig, ax = plt.subplots(figsize=(12, 10))
sin_time_data = pd.DataFrame(np.zeros((10, 10)), index=sorted(vehicle_one_noDW), columns=sorted(cargo_one_noDW))
for i in range(0, len(vehicle_cargo_match)):
    sin_time_data.loc[vehicle_one_noDW[i], cargo_one_noDW[i]] = vehicle_cargo_match[i]
print(sin_time_data)
sns.heatmap(sin_time_data,
            annot=True, vmin=0, vmax=0.6, square=True, cmap="Blues", annot_kws={"size": 35})
# 获取colorbar对象
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=30)

# 添加边框
cbar.outline.set_visible(True)

# 设置边框的线宽和颜色
cbar.outline.set_linewidth(2)
cbar.outline.set_edgecolor('black')
# 添加边框
ax.spines['top'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(True)

# 设置边框的线宽和颜色
ax.spines['top'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)

ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
ax.collections[0].set_edgecolor("black")  # 设置边框颜色为黑色
# 设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=30)

# 设置横纵坐标的名称以及对应字体格式
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 40,
         }

ax.set_title('IDBN', font2)
ax.set_ylabel('Vehicle Number', font2)
ax.set_xlabel('Cargo Number', font2)
plt.show()
#
# t=2
vehicle_one_noDW = [11, 14, 15, 16, 17, 18, 19, 20, 21, 22]
cargo_one_noDW = [22, 16, 2, 21, 14, 17, 12, 20, 19, 18]
vehicle_cargo_match = [0.319, 0.138, 0.211, 0.168, 0.198, 0, 0.136, 0, 0.14, 0.402]
plt.rc('font', family='Times New Roman')
fig, ax = plt.subplots(figsize=(12, 10))
sin_time_data = pd.DataFrame(np.zeros((10, 10)), index=sorted(vehicle_one_noDW), columns=sorted(cargo_one_noDW))
for i in range(0, len(vehicle_cargo_match)):
    sin_time_data.loc[vehicle_one_noDW[i], cargo_one_noDW[i]] = vehicle_cargo_match[i]
print(sin_time_data)
sns.heatmap(sin_time_data,
            annot=True, vmin=0, vmax=0.6, square=True, cmap="Blues", annot_kws={"size": 35})
# 获取colorbar对象
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=30)

# 添加边框
cbar.outline.set_visible(True)

# 设置边框的线宽和颜色
cbar.outline.set_linewidth(2)
cbar.outline.set_edgecolor('black')
# 添加边框
ax.spines['top'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(True)

# 设置边框的线宽和颜色
ax.spines['top'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)

ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
ax.collections[0].set_edgecolor("black")  # 设置边框颜色为黑色
# 设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=30)

# 设置横纵坐标的名称以及对应字体格式
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 40,
         }

ax.set_title('IDBN', font2)
ax.set_ylabel('Vehicle Number', font2)
ax.set_xlabel('Cargo Number', font2)
plt.show()
# t=3
vehicle_one_noDW = [18, 20, 23, 24, 25, 26, 27, 28, 29, 30]
cargo_one_noDW = [17, 24, 30, 23, 27, 25, 20, 26, 29, 28]
vehicle_cargo_match = [0.176, 0.238, 0.158, 0, 0.2, 0, 0.218, 0, 0.163, 0]
plt.rc('font', family='Times New Roman')
fig, ax = plt.subplots(figsize=(12, 10))
sin_time_data = pd.DataFrame(np.zeros((10, 10)), index=sorted(vehicle_one_noDW), columns=sorted(cargo_one_noDW))
for i in range(0, len(vehicle_cargo_match)):
    sin_time_data.loc[vehicle_one_noDW[i], cargo_one_noDW[i]] = vehicle_cargo_match[i]
print(sin_time_data)
sns.heatmap(sin_time_data,
            annot=True, vmin=0, vmax=0.6, square=True, cmap="Blues", annot_kws={"size": 35})
# 获取colorbar对象
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=30)

# 添加边框
cbar.outline.set_visible(True)

# 设置边框的线宽和颜色
cbar.outline.set_linewidth(2)
cbar.outline.set_edgecolor('black')
# 添加边框
ax.spines['top'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(True)

# 设置边框的线宽和颜色
ax.spines['top'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)

ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
ax.collections[0].set_edgecolor("black")  # 设置边框颜色为黑色
# 设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=30)

# 设置横纵坐标的名称以及对应字体格式
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 40,
         }

ax.set_title('IDBN', font2)
ax.set_ylabel('Vehicle Number', font2)
ax.set_xlabel('Cargo Number', font2)
plt.show()
