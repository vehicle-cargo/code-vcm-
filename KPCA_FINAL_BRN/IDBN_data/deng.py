#!/usr/bin/python3
# coding=utf-8
import vcm_new
import pandas as pd
import numpy as np
real_data = pd.read_csv('realdata.csv', header=None)
real_data_dict = {}
for item in real_data.iterrows():
    mycontainer = [item[1][i] for i in range(1, 14)]
    real_data_dict[item[1][0]] = mycontainer
vehicle_queue = {}
cargo_queue = {}
vehicle_success_queue = {}
cargo_success_queue = {}
type_match_list = [['低栏车', '日用百货', 0.8], ['低栏车', '特殊货物', 0.3], ['低栏车', '机器零件', 1], ['低栏车', '生鲜果蔬', 0.5],
                   ['低栏车', '砂石散货', 0.8], ['低栏车', '五金机械', 1],
                   ['高栏车', '日用百货', 0.8], ['高栏车', '特殊货物', 0.5], ['高栏车', '机器零件', 1], ['高栏车', '生鲜果蔬', 0.5],
                   ['高栏车', '砂石散货', 1], ['高栏车', '五金机械', 1],
                   ['厢式车', '日用百货', 1], ['厢式车', '特殊货物', 0.8], ['厢式车', '机器零件', 1], ['厢式车', '生鲜果蔬', 0.5],
                   ['厢式车', '砂石散货', 0.8], ['厢式车', '五金机械', 1],
                   ['冷藏车', '日用百货', 0.1], ['冷藏车', '特殊货物', 0.1], ['冷藏车', '机器零件', 0.1], ['冷藏车', '生鲜果蔬', 1],
                   ['冷藏车', '砂石散货', 0.1], ['冷藏车', '五金机械', 0.1],
                   ['通风箱车', '日用百货', 0.8], ['通风箱车', '特殊货物', 0.5], ['通风箱车', '机器零件', 0.8], ['通风箱车', '生鲜果蔬', 1],
                   ['通风箱车', '砂石散货', 0.5], ['通风箱车', '五金机械', 0.8],
                   ['平板车', '日用百货', 0.6], ['平板车', '特殊货物', 0.6], ['平板车', '机器零件', 0.8], ['平板车', '生鲜果蔬', 0.6],
                   ['平板车', '砂石散货', 0.5], ['平板车', '五金机械', 0.8]]

# 设置三个时间片
time = [0, 1, 2, 3]
# 从数据集中读取数据
vehicle_df = pd.read_csv('deng/v12.csv', header=None)
cargo_df = pd.read_csv('deng/c12.csv', header=None)
# vehicle_df = pd.read_csv('./dataSet/vehicleDataSetFour.csv', header=None)
# cargo_df = pd.read_csv('./dataSet/cargoDataSetFour.csv', header=None)
# 存放每次匹配 队列中最大的货物或者车辆编号。便于下一个时间片进行添加数据
vehicle_maxNum = []
cargo_maxNum = []
# 存放所有的，每个时刻的成功几率，每一项都是pd.dataFrame
vcm_success_all = []
# 成功队列
success_queue = []
# 匹配成功的车辆、货物编号集合
success_vehicle_num = []
success_cargo_num = []
# 存放每个时刻匹配失败的车辆编号。动态权重时便于找到
fail_vehicle_time = []
fail_cargo_time = []
vehicle_side_value = []

for t in time:
    print('当前时间为t=', t)
    vcm_new.appendVehicle(vehicle_df, vehicle_maxNum, vehicle_queue, t)
    vcm_new.appendCargo(cargo_df, cargo_maxNum, cargo_queue, t)

    # 获得车辆和货物的属性分布概率
    vehicle_prior = vcm_new.vehicleAttrDis(vehicle_queue)
    cargo_prior = vcm_new.cargoAttrDis(cargo_queue)
    print('车辆先验概率：', vehicle_prior)
    print('货物先验概率：', cargo_prior)

    # 将n辆车，m个货物得到的m*n个概率值转化为DataFrame. 分别是pd.DataFrame的列索引，行索引，值
    vcm_pro_column = []
    vcm_pro_row = []
    vcm_pro_value = []
    # 得到此时队列中的车辆的索引
    for vehicleKey, vehicleValue in vehicle_queue.items():
        vcm_pro_row.append(vehicleKey)
    # 得到此时队列中的货物的索引
    for cargoKey, cargoValue in cargo_queue.items():
        vcm_pro_column.append(cargoKey)
    vcm_pro_value = [[] for i in vcm_pro_row]
    vcm_pro_important_degree = [[] for i in vcm_pro_row]
    vcm_pro_value_attr_degree = [[] for i in vcm_pro_row]
    i = 0
    m = 0
    for vehicleKey, vehicleValue in vehicle_queue.items():
        n = 0
        vcm_pro_value[i] = []
        w1 = 0.16
        w2 = 0.09
        w3 = 0.47
        w4 = 0.28
        for cargoKey, cargoValue in cargo_queue.items():
            attrMatch = vcm_new.VCM(vehicleValue, cargoValue,real_data_dict, w1, w2, w3, w4)
            if attrMatch == 0:
                finalMatch = 0
            else:
                # finalMatch = (attrMatch * vehicle_prior[m]) / cargo_prior[n]
                finalMatch = 0.8 * attrMatch + (0.2 * vehicle_prior[m]) / (1 - cargo_prior[n])
            print(round(finalMatch, 2), end=" ")
            vcm_pro_value[i].append(round(finalMatch, 3))
            # 属性匹配度
            vcm_pro_value_attr_degree[i].append(round(attrMatch, 3))
            # 环境匹配度
            important_degree = vehicle_prior[m] / (1 - cargo_prior[n])
            vcm_pro_important_degree[i].append(round(important_degree, 3))
            n = n + 1
        i = i + 1
        m = m + 1
        print()
    print('综合匹配度：' )
    value_df = pd.DataFrame(vcm_pro_value, columns=vcm_pro_column, index=vcm_pro_row)
    print(value_df)
    print('属性匹配度：')
    vcm_pro_value_attr_degree = pd.DataFrame(vcm_pro_value_attr_degree, columns=vcm_pro_column, index=vcm_pro_row)
    print(vcm_pro_value_attr_degree)
    print('环境匹配度：')
    vcm_pro_important_degree = pd.DataFrame(vcm_pro_important_degree, columns=vcm_pro_column, index=vcm_pro_row)
    print(vcm_pro_important_degree)
    vcm_success_all.append(value_df)
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

    rowIndex = 0
    columnIndex = 0
    # 遍历dataFrame，选出匹配成功的车辆货物编号。在成功之后，把相应的列置为-1，也就是说，其他车辆不能再选择此货物
    for index, row in value_df.iterrows():
        rowIndex = index
        # 更新某一行
        row = value_df.loc[rowIndex, :]
        for i, j in row.items():
            if max(row) == j:
                columnIndex = i
                break
        if max(row) >= vcm_pro_mean:
            success_vcm = [rowIndex, columnIndex, max(row)]
            success_queue.append(success_vcm)
            value_df.loc[:, columnIndex] = -1
        # print(value_df)

    print('车货匹配成功的列表：', success_queue)

    # 存放车辆，货物队列中所有的的编号
    vehicle_num_list = []
    cargo_num_list = []
    for key, value in vehicle_queue.items():
        vehicle_num_list.append(key)
    for key, value in cargo_queue.items():
        cargo_num_list.append(key)

    # 遍历成功匹配集合，筛选出匹配成功的车辆和货物并放入对应集合中
    for success in success_queue:
        if success[0] not in success_vehicle_num:
            success_vehicle_num.append(success[0])
        if success[1] not in success_cargo_num:
            success_cargo_num.append(success[1])
    success = [item for item in vehicle_num_list if item in success_vehicle_num]
    print('匹配成功的车辆编号集合：', success_vehicle_num, '，匹配成功的货物编号集合：', success_cargo_num)

    fail_vehicle_num = [item for item in vehicle_num_list if item not in success_vehicle_num]
    fail_cargo_num = [item for item in cargo_num_list if item not in success_cargo_num]
    fail_vehicle_time.append(fail_vehicle_num)
    fail_cargo_time.append(fail_cargo_num)
    print('匹配失败的车辆编号集合：', fail_vehicle_num, '，匹配失败的货物编号集合：', fail_cargo_num)

    # 遍历车辆队列。将匹配成功的加入到匹配成功队里，匹配失败的继续留在车辆队列
    for vehicleKey, vehicleValue in list(vehicle_queue.items()):
        # print(vehicleKey, vehicleValue)
        if vehicleKey in success_vehicle_num:
            vehicle_success_queue[vehicleKey] = vehicleValue
            vehicle_queue.pop(vehicleKey)
    # 遍历货物队列。将匹配成功的加入到匹配成功队里，匹配失败的继续留在货物队列
    for cargoKey, cargoValue in list(cargo_queue.items()):
        if cargoKey in success_cargo_num:
            cargo_success_queue[cargoKey] = cargoValue
            cargo_queue.pop(cargoKey)
    print('车辆队列中剩余车辆：', vehicle_queue)
    print('匹配成功队列中的车辆：', vehicle_success_queue)
    print('货物队列中剩余货物：', cargo_queue)
    print('匹配成功队列中的货物：', cargo_success_queue)
    print('-' * 60)
    two = 0.5 * vcm_pro_mean + 0.5 * len(success) / 10
    vehicle_side_value.append(two)

print('车主满意度：', np.round(vehicle_side_value, 2))

print('匹配失败车辆（t=0,1,2,3,4）:', fail_vehicle_time)
print('匹配失败货物（t=0,1,2,3,4）:', fail_cargo_time)
