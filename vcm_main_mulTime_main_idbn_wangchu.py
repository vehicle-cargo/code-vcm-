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
# vehicle_queue = {1: ['低栏车', 11000, 50, [108.322574, 22.833533], [113.263955, 23.154211], [2019050709, 2019051915]],
#                  2: ['高栏车', 8500, 80, [113.263955, 23.154211], [108.322574, 22.833533], [2019050812, 2019052516]],
#                  3: ['厢式车', 8000, 80, [108.322574, 22.833533], [114.420973, 23.159162], [2019050710, 2019051012]],
#                  4: ['冷藏车', 7000, 75, [108.322574, 22.833533], [112.476087, 23.080553], [2019050908, 2019051512]],
#                  5: ['通风箱车', 11000, 80, [109.395618, 24.315365], [113.263955, 23.154211], [2019050708, 2019052215]],
#                  6: ['厢式车', 6000, 70, [113.263955, 23.154211], [110.28662, 25.267723], [2019050616, 2019052116]],
#                  7: ['低栏车', 11000, 50, [113.434202, 22.519376], [108.655118, 22.060841], [2019050608, 2019051416]],
#                  8: ['高栏车', 8500, 80, [113.263955, 23.154211], [110.28662, 25.267723], [2019050612, 2019052616]],
#                  9: ['平板车', 12000, 40, [116.425052, 39.934032], [113.51597, 22.292177], [2019050708, 2019052215]],
#                  v10: ['高栏车', 8500, 60, [116.425052, 39.934032], [117.208789, 39.095388], [2019050612, 2019052616]],
#                  11: ['冷藏车', 7000, 75, [113.25872, 23.139562], [114.051164, 22.609383], [2019050708, 2019052215]],
#                  12: ['平板车', 15000, 50, [113.25872, 23.139562], [113.51597, 22.292177], [2019050812, 2019052516]],
#                  13: ['低栏车', 10000, 45, [112.544412, 37.881898], [113.104074, 36.215097], [2019050612, 2019052616]],
#                  14: ['高栏车', 8500, 80, [113.104074, 36.215097], [112.544412, 37.881898], [2019050812, 2019052516]],
#                  15: ['通风箱车', 11000, 80, [112.544412, 37.881898], [112.85052, 35.493965], [2019050708, 2019052215]],
#                  16: ['高栏车', 8500, 80, [113.104074, 36.215097], [116.425052, 39.934032], [2019050608, 2019051416]],
#                  17: ['厢式车', 6000, 70, [113.104074, 36.215097], [116.425052, 39.934032], [2019050708, 2019052215]],
#                  18: ['低栏车', 11000, 50, [113.104074, 36.215097], [117.208789, 39.095388], [2019050608, 2019051416]],
#                  19: ['高栏车', 8500, 80, [112.85052, 35.493965], [116.425052, 39.934032], [2019050612, 2019052616]],
#                  20: ['通风箱车', 11000, 80, [112.85052, 35.493965], [117.208789, 39.095388], [2019050508, 2019052015]],
#                  21: ['低栏车', 11000, 50, [113.25872, 23.139562], [114.051164, 22.609383], [2019050812, 2019052516]],
#                  22: ['厢式车', 6000, 70, [106.535893, 29.590094], [104.060293, 30.593689], [2019051108, 2019052115]],
#                  23: ['高栏车', 8500, 80, [113.25872, 23.139562], [104.060293, 30.593689], [2019050612, 2019052616]],
#                  24: ['厢式车', 6000, 70, [112.544412, 37.881898], [112.85052, 35.493965], [2019051308, 2019052215]],
#                  25: ['冷藏车', 7000, 75, [106.535893, 29.590094], [104.060293, 30.593689], [2019050812, 2019052516]],
#                  26: ['厢式车', 6000, 70, [113.362213, 40.097111], [116.425052, 39.934032], [2019050616, 2019052116]],
#                  27: ['通风箱车', 11000, 80, [113.362213, 40.097111], [104.060293, 30.593689], [2019051008, 2019051715]],
#                  28: ['高栏车', 8500, 80, [111.019703, 35.033296], [103.839542, 36.071046], [2019050612, 2019052616]],
#                  29: ['低栏车', 11000, 50, [117.208789, 39.095388], [113.25872, 23.139562], [2019050908, 2019052515]],
#                  30: ['冷藏车', 7000, 75, [106.535893, 29.590094], [104.060293, 30.593689], [2019050616, 2019052116]], }
# cargo_queue = {1: ['日用百货', 5000, 30, [108.322574, 22.833533], [113.263955, 23.154211], [2019051212, 2019052516]],
#                2: ['特殊货物', 7000, 40, [109.395618, 24.315365], [113.263955, 23.154211], [2019051209, 2019052315]],
#                3: ['特殊货物', 5000, 35, [108.322574, 22.833533], [113.263955, 23.154211], [2019051109, 2019052512]],
#                4: ['日用百货', 8000, 55, [110.28662, 25.267723], [108.322574, 22.833533], [2019051209, 2019052216]],
#                5: ['日用百货', 5000, 50, [109.395618, 24.315365], [113.263955, 22.833533], [2019051109, 2019052314]],
#                6: ['机器零件', 10000, 50, [113.25872, 23.139562], [113.51597, 22.292177], [2019051209, 2019052016]],
#                7: ['生鲜果蔬', 6500, 30, [113.25872, 23.139562], [114.051164, 22.609383], [2019051109, 2019052114]],
#                8: ['砂石散货', 10000, 50, [113.263955, 23.154211], [110.28662, 25.267723], [2019050708, 2019051915]],
#                9: ['五金机械', 8000, 40, [108.322574, 22.833533], [113.263955, 23.154211], [2019051109, 2019051814]],
#                v10: ['日用百货', 6000, 45, [116.425052, 39.934032], [113.51597, 22.292177], [2019051209, 2019051815]],
#                11: ['特殊货物', 7000, 40, [113.104074, 36.215097], [116.425052, 39.934032], [2019050612, 2019052616]],
#                12: ['生鲜果蔬', 6500, 30, [116.425052, 39.934032], [113.51597, 22.292177], [2019051209, 2019051815]],
#                13: ['机器零件', 10000, 50, [112.85052, 35.493965], [116.425052, 39.934032], [2019050908, 2019052515]],
#                14: ['生鲜果蔬', 6500, 30, [113.362213, 40.097111], [116.425052, 39.934032], [2019050612, 2019052616]],
#                15: ['特殊货物', 7000, 40, [113.25872, 23.139562], [104.060293, 30.593689], [2019051209, 2019051815]],
#                16: ['五金机械', 8000, 40, [106.535893, 29.590094], [104.060293, 30.593689], [2019050812, 2019052516]],
#                17: ['生鲜果蔬', 6500, 30, [111.019703, 35.033296], [103.839542, 36.071046], [2019050908, 2019052515]],
#                18: ['机器零件', 10000, 50, [113.104074, 36.215097], [112.544412, 37.881898], [2019050612, 2019052616]],
#                19: ['日用百货', 6000, 45, [117.208789, 39.095388], [113.25872, 23.139562], [2019050812, 2019052516]],
#                20: ['砂石散货', 10000, 50, [113.104074, 36.215097], [116.425052, 39.934032], [2019050710, 2019051012]],
#                21: ['砂石散货', 10000, 50, [111.019703, 35.033296], [103.839542, 36.071046], [2019050709, 2019051915]],
#                22: ['生鲜果蔬', 6500, 30, [113.434202, 22.519376], [108.655118, 22.060841], [2019051209, 2019051815]],
#                23: ['机器零件', 10000, 50, [113.263955, 23.154211], [108.322574, 22.833533], [2019051209, 2019051815]],
#                24: ['五金机械', 8000, 40, [106.535893, 29.590094], [104.060293, 30.593689], [2019050710, 2019051012]],
#                25: ['生鲜果蔬', 6500, 30, [109.395618, 24.315365], [113.263955, 23.154211], [2019050708, 2019052215]],
#                26: ['日用百货', 6000, 45, [117.208789, 39.095388], [113.25872, 23.139562], [2019051209, 2019051815]],
#                27: ['砂石散货', 10000, 50, [113.104074, 36.215097], [116.425052, 39.934032], [2019050709, 2019051915]],
#                28: ['生鲜果蔬', 6500, 30, [113.263955, 23.154211], [108.322574, 22.833533], [2019051109, 2019052512]],
#                29: ['日用百货', 5000, 30, [106.535893, 29.590094], [104.060293, 30.593689], [2019050708, 2019052215]],
#                30: ['机器零件', 10000, 50, [109.395618, 24.315365], [113.263955, 22.833533], [2019051109, 2019052512]]}
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
vehicle_df = pd.read_csv('./vehicleDataSet.csv', header=None)
cargo_df = pd.read_csv('./cargoDataSet.csv', header=None)
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

    # 将值初始化
    vcm_pro_value = [[] for i in vcm_pro_row]
    vcm_pro_important_degree = [[] for i in vcm_pro_row]
    vcm_pro_value_attr_degree = [[] for i in vcm_pro_row]

    # 得到每辆车和每个货相匹配的概率值
    i = 0
    m = 0
    for vehicleKey, vehicleValue in vehicle_queue.items():
        n = 0
        vcm_pro_value[i] = []
        w1 = 0.16
        w2 = 0.09
        w3 = 0.47
        w4 = 0.28
        q = 0
        # 动态权重,判断出现了几次，即有几次没匹配成功
        for v in range(0, len(fail_vehicle_time)):
            if vehicleKey in fail_vehicle_time[v]:
                q = q + 1
        if q > 0:
            if (w4 + 0.1 * q) > 0.5:
                w4 = 0.5
                w1 = w1 - (0.5 - 0.28) / 3
                w2 = w2 - (0.5 - 0.28) / 3
                w3 = w3 - (0.5 - 0.28) / 3
            else:
                w4 = w4 + 0.1 * q
                w1 = w1 - (0.1 * q) / 3
                w2 = w2 - (0.1 * q) / 3
                w3 = w3 - (0.1 * q) / 3
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

    # print(vcm_pro_value)
    # 将列表转化为dataFrame
    print('综合匹配度：' )
    value_df = pd.DataFrame(vcm_pro_value, columns=vcm_pro_column, index=vcm_pro_row)
    print(value_df)
    print('属性匹配度：')
    vcm_pro_value_attr_degree = pd.DataFrame(vcm_pro_value_attr_degree, columns=vcm_pro_column, index=vcm_pro_row)
    print(vcm_pro_value_attr_degree)
    print('环境匹配度：')
    vcm_pro_important_degree = pd.DataFrame(vcm_pro_important_degree, columns=vcm_pro_column, index=vcm_pro_row)
    print(vcm_pro_important_degree)
    # outputpath = './resultData/noNumber/1010framedataT' + str(t) + '.csv'
    # outputpath = './resultData/Dw_dataFrameFourT' + str(t) + '.csv'
    # value_df.to_csv(outputpath, sep=',', index=True, header=True)
    vcm_success_all.append(value_df)

    # vcm_pro_value = [[0.52, 0.52, 0.52, 0.0, 0.0], [0.39, 0.46, 0.38, 0.0, 0.48], [0.0, 0.0, 0.0, 0.0, 0.0],
    #                  [0.18, 0.21, 0.21, 0.0, 0.0], [0.24, 0.36, 0.24, 0.38, 0.34], [0.33, 0.0, 0.31, 0.0, 0.44],
    #                  [0.2, 0.21, 0.19, 0.0, 0.0], [0.43, 0.52, 0.42, 0.0, 0.54]]
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


print('匹配失败车辆（t=0,1,2,3,4）:', fail_vehicle_time)
print('匹配失败货物（t=0,1,2,3,4）:', fail_cargo_time)
