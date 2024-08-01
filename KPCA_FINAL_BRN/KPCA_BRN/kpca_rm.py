import pandas as pd
import numpy as np
import vcm_change_new

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
# 设置3个时间片
time = [0, 1, 2, 3]
# 从数据集中读取数据

vehicle_df = pd.read_csv('../KPCA_BRN/new_data/kbr/v5.csv',
                         header=None)  # header=None自动添加列字段
cargo_df = pd.read_csv('../KPCA_BRN/new_data/kbr/c5.csv', header=None)
# 40*6
# print(vehicle_df)
# print(cargo_df)
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
for t in time:
    print('当前时间为t=', t)
    # 从数据集中读取车辆的数据-不够十辆
    vcm_change_new.appendVehicle(vehicle_df, vehicle_maxNum, vehicle_queue, t)
    # 从数据集中读取货物的数据
    vcm_change_new.appendCargo(cargo_df, cargo_maxNum, cargo_queue, t)
    # 获得车辆和货物的属性概率分布（先验概率）
    # 根据车辆和货物队列中的各个货物的信息计算各个属性占所有车辆和货物特征属性的分布规律
    vehicle_prior = vcm_change_new.vehicleAttrDis(vehicle_queue)
    cargo_prior = vcm_change_new.cargoAttrDis(cargo_queue)
    print('车辆的先验概率', vehicle_prior)
    print('货物先验概率：', cargo_prior)
    # 将10辆车和十个货物得到的10*10的概率值转化为Dataframe分别是pd.DataFrame的列索引，行索引，值
    vcm_pro_column = []  # 车辆
    vcm_pro_row = []  # 货物
    vcm_pro_value = []  # 车辆与货物匹配的概率值
    # 得到此时队列中车辆的索引
    for vehicleKey, vehicleValue in vehicle_queue.items():
        vcm_pro_row.append(vehicleKey)
    for cargoKey, cargoValue in cargo_queue.items():
        vcm_pro_column.append(cargoKey)

    # 将g值初始化
    vcm_pro_value = [[] for i in vcm_pro_row]  # 匹配概率值初始化
    vcm_pro_important_degree = [[] for i in vcm_pro_row]
    vcm_pro_value_attr_degree = [[] for i in vcm_pro_row]

    # 得到每辆车和每个货物相匹配的概率值
    i = 0
    m = 0
    # 车辆的索引和特征值 匹配是从车辆的角度出发的
    for vehicleKey, vehicleValue in vehicle_queue.items():
        n = 0
        vcm_pro_value[i] = []
        w1 = 0.2
        w2 = 0.19
        w3 = 0.13
        w4 = 0.19
        w5 = 0.18
        w6 = 0.12
        # 对于每个货物来
        for cargoKey, cargoValue in cargo_queue.items():
            attrMatch = vcm_change_new.VCM(vehicleValue, cargoValue, real_data_dict, w1, w2, w3, w4, w5, w6)
            if attrMatch == 0:  # 如果属性匹配度为0 则最终的匹配度为0
                finalMatch = 0
            else:
                # prior为先验概率-环境匹配度
                finalMatch = 0.75 * attrMatch + (0.25 * vehicle_prior[m]) / (1 - cargo_prior[n])

            print(round(finalMatch, 2), end="")  # 结果保留到小数点后两位
            # 对应的每一个车辆与货物的匹配的概率值
            vcm_pro_value[i].append(round(finalMatch, 3))
            # 记录属性匹配度
            vcm_pro_value_attr_degree[i].append(round(attrMatch, 3))
            # 记录环境匹配度
            important_degree = vehicle_prior[m] / (1 - cargo_prior[n])
            vcm_pro_important_degree[i].append(round(important_degree, 3))
            n = n + 1
        i = i + 1
        m = m + 1
        print()
    print('综合匹配度：')
    # 将综合匹配度取出来
    value_df = pd.DataFrame(vcm_pro_value, columns=vcm_pro_column, index=vcm_pro_row)
    print(value_df)
    print('属性匹配度：')
    vcm_pro_value_attr_degree = pd.DataFrame(vcm_pro_value_attr_degree, columns=vcm_pro_column, index=vcm_pro_row)
    print(vcm_pro_value_attr_degree)
    print('环境匹配度：')  # 先验概率
    vcm_pro_important_degree = pd.DataFrame(vcm_pro_important_degree, columns=vcm_pro_column, index=vcm_pro_row)
    print(vcm_pro_important_degree)
    vcm_success_all.append(value_df)  # 某一时刻匹配成功的概率

    # 某时刻匹配概率得到所有概率之和与概率非零的个数，求平均值
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
    # 遍历dataFrame,选出匹配成功的车辆和货物编号，在成功之后，把相应的列置为-1，也就是其他车辆不能再选择此货物
    for index, row in value_df.iterrows():
        rowIndex = index
        # 更新某一行
        # 综合匹配度
        row = value_df.loc[rowIndex, :]
        for i, j in row.items():
            if max(row) == j:
                columnIndex = i
                break

        if max(row) >= 0.37:
            success_vcm = [rowIndex, columnIndex, max(row)]
            success_queue.append(success_vcm)
            value_df.loc[:, columnIndex] = -1  # 行全选货物列置为-1 标记已经匹配成功了

    print('车货匹配成功的列表：', success_queue)

    # 存放车辆，货物队列中的所有编号
    vehicle_num_list = []
    cargo_num_list = []
    degree_num_list = []
    for key, value in vehicle_queue.items():  # 车辆的编号
        vehicle_num_list.append(key)
    for key, value in cargo_queue.items():  # 货物的编号
        cargo_num_list.append(key)
    # 遍历成功匹配的集合，筛选匹配成功的车辆和货物放入对应的集合中
    for success in success_queue:  # 匹配成功的队列
        if success[0] not in success_vehicle_num:
            success_vehicle_num.append(success[0])

        if success[1] not in success_cargo_num:
            success_cargo_num.append(success[1])

        degree_num_list.append(success[2])
    degree = [item for item in degree_num_list if item in degree_num_list]
    success = [item for item in vehicle_num_list if item in success_vehicle_num]
    print('success:', success)
    print('degree:', degree)
    # 计算每个时间片内匹配成功车辆和货物的综合匹配度之和
    degree_sum = sum(degree)
    print(f'时间片 t={t} 内匹配成功车辆和货物的综合匹配度之和:', degree_sum)

    two = 0.3 * (0.5 * degree_sum + 0.5 * len(success) / 10)

    vehicle_side_value.append(two)
    success = [item for item in vehicle_num_list if item in success_vehicle_num]

    print('匹配成功的车辆的编号集合：', success_vehicle_num, ',匹配成功的货物集合的编号：', success_cargo_num)
    faile_vehilce_num = [item for item in vehicle_num_list if item not in success_vehicle_num]
    faile_cargo_num = [item for item in cargo_num_list if item not in success_cargo_num]

    # 匹配失败时会考虑时间动态权重
    faile_vehicle_time.append(faile_vehilce_num)
    faile_cargo_time.append(faile_cargo_num)
    print('匹配失败的车辆编号集合：', faile_vehilce_num, ',匹配失败的货物的编号集合：', faile_cargo_num)

    # 遍历车辆队列，将匹配成功的加入到匹配成功的队列里，将匹配失败的继续留在车辆队列
    for vehicleKey, vehicleValue in list(vehicle_queue.items()):
        if vehicleKey in success_vehicle_num:
            vehicle_success_queue[vehicleKey] = vehicleValue
            vehicle_queue.pop(vehicleKey)  # 匹配成功后从匹配队列里删除

    for cargoKey, cargoValue in list(cargo_queue.items()):
        if cargoKey in success_cargo_num:
            cargo_success_queue[cargoKey] = cargoValue
            cargo_queue.pop(cargoKey)  # 从货物队列里删除

    print('车辆队列中剩余的车辆:', vehicle_queue)
    print('匹配成功队列中的车辆:', vehicle_success_queue)
    print('货物队列中剩余的货物:', cargo_queue)
    print('匹配成功的货物队列:', cargo_success_queue)
    print('-' * 60)
    success_queue = []

print('车主满意度：', np.round(vehicle_side_value, 2))
print('匹配失败的车辆(t=0,1,2,3,4):', faile_vehicle_time)
print('匹配失败的货物(t=0,1,2,3,4):', faile_cargo_time)
