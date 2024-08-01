import pandas as pd
import numpy as np
import vcm_new

real_data = pd.read_csv('realdata.csv', header=None)
real_data_dict = {}
for item in real_data.iterrows():
    mycontainer = [item[1][i] for i in range(1, 14)]
    real_data_dict[item[1][0]] = mycontainer



# 初始化车辆队列
vehicle_queue = {}
# 初始化货物队列
cargo_queue = {}
# 匹配成功的车辆队列
vehicle_success_queue = {}
# 匹配成功的货物队列
cargo_success_queue = {}
# 类型匹配度
type_match_list = [['低栏车', '日用百货', 0.8], ['低栏车', '特殊货物', 0.3], ['低栏车', '机器零件', 1],
                   ['低栏车', '生鲜果蔬', 0.5],
                   ['低栏车', '砂石散货', 0.8], ['低栏车', '五金机械', 1],
                   ['高栏车', '日用百货', 0.8], ['高栏车', '特殊货物', 0.5], ['高栏车', '机器零件', 1],
                   ['高栏车', '生鲜果蔬', 0.5],
                   ['高栏车', '砂石散货', 1], ['高栏车', '五金机械', 1],

                   ['厢式车', '日用百货', 1], ['厢式车', '特殊货物', 0.8], ['厢式车', '机器零件', 1],
                   ['厢式车', '生鲜果蔬', 0.5],
                   ['厢式车', '砂石散货', 0.8], ['厢式车', '五金机械', 1],
                   ['冷藏车', '日用百货', 0.1], ['冷藏车', '特殊货物', 0.1], ['冷藏车', '机器零件', 0.1],
                   ['冷藏车', '生鲜果蔬', 1],
                   ['冷藏车', '砂石散货', 0.1], ['冷藏车', '五金机械', 0.1],
                   ['通风箱车', '日用百货', 0.8], ['通风箱车', '特殊货物', 0.5], ['通风箱车', '机器零件', 0.8],
                   ['通风箱车', '生鲜果蔬', 1],
                   ['通风箱车', '砂石散货', 0.5], ['通风箱车', '五金机械', 0.8],
                   ['平板车', '日用百货', 0.6], ['平板车', '特殊货物', 0.6], ['平板车', '机器零件', 0.8],
                   ['平板车', '生鲜果蔬', 0.6],
                   ['平板车', '砂石散货', 0.5], ['平板车', '五金机械', 0.8]]




# 每次存放数据中，最大的货物、车辆的编号，以便下一个时间片添加数据
vehicle_maxNum = []
cargo_maxNum = []
# 存放每个时刻所有成功的几率，每一项都是pd.dataFrame
vcm_success_all = []
# 成功的队列
success_queue = []
# 匹配成功的车辆及货物的编号集合
success_vehicle_num = []
success_cargo_num = []
# 存放每个时刻匹配失败的车辆的编号，动态权重时便于找到
faile_vehicle_time = []
faile_cargo_time = []
vehicle_side_value = []
import pandas as pd
from queue import Queue
import ast

# 读取CSV文件
vehicle_df = pd.read_csv('vehiclefull.csv', header=0)
vehicle_df.columns = vehicle_df.columns.str.strip()  # 去除列名的前后空格

# 初始化一个队列
vehicle_queue = Queue()

# 逐行将CSV文件中的数据存入队列
for idx, row in vehicle_df.iterrows():
    try:
        vehicle_data = [
            row['type'],
            int(row['quality']),
            int(row['volumn']),
            ast.literal_eval(row['startPoint']),
            ast.literal_eval(row['endPoint']),
            ast.literal_eval(row['time']),
            int(row['b_index'])
        ]
        vehicle_queue.put(vehicle_data)
    except ValueError as e:
        print(f"Error processing row {idx + 1}: {e}")
    except SyntaxError as e:
        print(f"Error parsing list in row {idx + 1}: {e}")

# 创建最终的队列字典
final_vehicle_queue = {}
idx = 1
while not vehicle_queue.empty():
    final_vehicle_queue[idx] = vehicle_queue.get()
    idx += 1

# 打印最终的队列字典
print(f"Total items in the queue: {len(final_vehicle_queue)}")
# print(final_vehicle_queue)
# 读取CSV文件
cargo_df = pd.read_csv('cargofull..csv', header=0)
cargo_df.columns = cargo_df.columns.str.strip()  # 去除列名的前后空格

# 初始化一个队列
cargo_queue = Queue()

# 逐行将CSV文件中的数据存入队列
for idx, row in cargo_df.iterrows():
    try:
        cargo_data = [
            row['type'],
            int(row['quality']),
            int(row['volumn']),
            ast.literal_eval(row['startPoint']),
            ast.literal_eval(row['endPoint']),
            ast.literal_eval(row['time']),
            int(row['a_index'])
        ]
        cargo_queue.put(cargo_data)
    except ValueError as e:
        print(f"Error processing row {idx + 1}: {e}")
    except SyntaxError as e:
        print(f"Error parsing list in row {idx + 1}: {e}")

# 创建最终的队列字典
final_cargo_queue = {}
idx = 1
while not cargo_queue.empty():
    final_cargo_queue[idx] = cargo_queue.get()
    idx += 1

# 打印最终的队列字典
print(f"Total items in the queue: {len(final_cargo_queue)}")
# print(final_cargo_queue)

vehicle_prior = vcm_new.vehicleAttrDis(final_vehicle_queue)
cargo_prior = vcm_new.cargoAttrDis(final_cargo_queue)
print('车辆先验概率：', vehicle_prior)
print('货物先验概率：', cargo_prior)

# 存放每个时刻所有成功的几率，每一项都是pd.dataFrame
vcm_success_all = []
# 将n辆车，m个货物得到的m*n个概率值转化为DataFrame. 分别是pd.DataFrame的列索引，行索引，值
vcm_pro_column = []
vcm_pro_row = []
vcm_pro_value = []
# 得到此时队列中的车辆的索引
for vehicleKey, vehicleValue in final_vehicle_queue.items():
    vcm_pro_row.append(vehicleKey)
# 得到此时队列中的货物的索引
for cargoKey, cargoValue in final_cargo_queue.items():
    vcm_pro_column.append(cargoKey)
# 将值初始化
vcm_pro_value = [[] for i in vcm_pro_row]

# 得到每辆车和每个货物相匹配的概率值
i = 0
m = 0
# 车辆的索引和特征值 匹配是从车辆的角度出发的
for vehicleKey, vehicleValue in final_vehicle_queue.items():
    n = 0
    vcm_pro_value[i] = []
    w1 = 0.12
    w2 = 0.24
    w3 = 0.32
    w4 = 0.31
    # 对于每个货物来
    for cargoKey, cargoValue in final_cargo_queue.items():
        attrMatch = vcm_new.VCM(vehicleValue, cargoValue, real_data_dict, w1, w2, w3, w4)
        if attrMatch == 0:  # 如果属性匹配度为0 则最终的匹配度为0
            finalMatch = 0
        else:
            # prior为先验概率-环境匹配度
            finalMatch = 0.75 * attrMatch + (0.25 * vehicle_prior[m]) / (1 - cargo_prior[n])
            print(round(finalMatch, 2))
        vcm_pro_value[i].append(round(finalMatch, 2))
    i = i + 1
    print()

# 将列表转化为dataFrame
value_df = pd.DataFrame(vcm_pro_value, columns=vcm_pro_column, index=vcm_pro_row)
print(value_df)
vcm_success_all.append(value_df)

# 某时刻匹配概率  得到所有概率之和 与 概率非0 的个数，求得平均值
vcm_pro_sum = 0
vcm_pro_notZero_num = 0
vcm_pro_mean = 0
for value in vcm_pro_value:
    vcm_pro_sum += sum(value)
for value in vcm_pro_value:
    for i in value:
        if i != 0:
            vcm_pro_notZero_num += 1
vcm_pro_mean = vcm_pro_sum / vcm_pro_notZero_num
print('概率均值：', vcm_pro_mean)
#
# # 成功队列AND失败队列
# success_queue = []
# fail_vehicle_queue = []
# fail_cargo_queue = []
# rowIndex = 0
# columnIndex = 0
#
# for index, row in value_df.iterrows():
#     rowIndex = index
#     # 更新某一行
#     row = value_df.loc[rowIndex, :]
#     for i, j in row.items():
#         if max(row) == j:
#             columnIndex = i
#             break
#     if max(row) >= vcm_pro_mean:
#         success_vcm = [rowIndex, columnIndex, max(row)]
#         success_queue.append(success_vcm)
#         value_df.loc[:, columnIndex] = -1
#     # print(value_df)
#
# print('车货匹配成功的列表：', success_queue)
#
# # 存放车辆，货物队列中所有的的编号
# vehicle_num_list = []
# cargo_num_list = []
# for key, value in vehicle_queue.items():
#     vehicle_num_list.append(key)
# for key, value in final_cargo_queue.items():
#     cargo_num_list.append(key)
#
# # 匹配成功的车辆、货物编号集合
# success_vehicle_num = []
# success_cargo_num = []
# for success in success_queue:
#     success_vehicle_num.append(success[0])
#     success_cargo_num.append(success[1])
# print('匹配成功的车辆编号集合：', success_vehicle_num, '，匹配成功的货物编号集合：', success_cargo_num)
#
# fail_vehicle_num = [item for item in vehicle_num_list if item not in success_vehicle_num]
# fail_cargo_num = [item for item in cargo_num_list if item not in success_cargo_num]
# print('匹配失败的车辆编号集合：', fail_vehicle_num, '，匹配失败的货物编号集合：', fail_cargo_num)

