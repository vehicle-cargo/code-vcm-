import distance as dt
import datetime
from collections import Counter

# 车辆队列-8辆车
vehicle_queue = {
    1: ['低栏车', 8000, 40, [108.322574, 22.833533], [113.263955, 23.154211], [2019050709, 2019051915]],
    2: ['高栏车', 8000, 60, [113.263955, 23.154211], [108.322574, 22.833533], [2019050812, 2019052516]],
    3: ['厢式车', 6000, 50, [108.322574, 22.833533], [114.420973, 23.159162], [2019050710, 2019051012]],
    4: ['冷藏车', 7000, 45, [108.322574, 22.833533], [112.476087, 23.080553], [2019050908, 2019051512]],
    5: ['通风箱车', 10000, 80, [109.395618, 24.315365], [113.263955, 23.154211], [2019050708, 2019052215]],
    6: ['厢式车', 6000, 50, [113.263955, 23.154211], [110.28662, 25.267723], [2019050616, 2019052116]],
    7: ['低栏车', 8000, 40, [113.434202, 22.519376], [108.655118, 22.060841], [2019050608, 2019051416]],
    8: ['高栏车', 8000, 60, [113.263955, 23.154211], [110.28662, 25.267723], [2019050612, 2019052616]]
}
# 货物队列-5个货物
cargo_queue = {
    1: ['日用百货', 5000, 30, [108.322574, 22.833533], [113.263955, 23.154211], [2019051212, 2019052516]],
    2: ['特殊货物', 7000, 40, [109.395618, 24.315365], [113.263955, 23.154211], [2019051209, 2019052315]],
    3: ['特殊货物', 5000, 35, [108.322574, 22.833533], [113.263955, 23.154211], [2019051109, 2019052512]],
    4: ['日用百货', 8000, 70, [110.28662, 25.267723], [108.322574, 22.833533], [2019051209, 2019052216]],
    5: ['日用百货', 5000, 50, [109.395618, 24.315365], [113.263955, 22.833533], [2019051109, 2019052314]]
}
# 类型匹配度事先定义好的
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


# 类型匹配度
def getTypeMatch(vehicle, cargo):
    # d-vehicle1.csv='厢式车
    # d-cargo1.csv='特殊货物'
    # 初始化类型匹配度
    typeMatchDegree = 0
    for item in type_match_list:
        if vehicle == item[0] and cargo == item[1]:
            typeMatchDegree = item[2]
    return typeMatchDegree


# 质量或者体积匹配度
def getQualityVolumnMatch(VQ, VV, CQ, CV):
    """

    :param VQ: 车辆的质量
    :param VV: 车辆的体积
    :param CQ: 货物的质量
    :param CV: 货物的体积
    :return:
    """

    # 初始化质量或体积匹配度
    QualityVolumnMatchDegree = 0
    # 判断清货还是重货，轻货看质量，重货看体积
    if (CV / 0.006) > CQ:  # 重货
        QualityVolumnMatchDegree = CV / VV
    else:
        QualityVolumnMatchDegree = CQ / VQ

    # 如果货物质量体积分别都大于车辆的质量和体积则匹配度为0
    if QualityVolumnMatchDegree > 1:
        QualityVolumnMatchDegree = 0

    return QualityVolumnMatchDegree


# 路径重合度
def getFlowMatch(VO, VD, CO, CD):
    """

    :param VO: d-vehicle1.csv origin:车辆的起始地
    :param VD: d-vehicle1.csv destination:车辆的目的地
    :param CO: d-cargo1.csv origin:货物的起始地
    :param CD: d-cargo1.csv destination:货物的目的地
    :return:
    """
    # 分别表示，货物起始点到终点距离，车辆货物起始地距离，车辆货物终点距离
    cargo_origin_destination = 0
    vehicle_cargo_origin = 0
    vehicle_cargo_destination = 0
    flowMatchDegree = 0
    VO = ",".join('%s' % id for id in VO)
    VD = ",".join('%s' % id for id in VD)
    CO = ",".join('%s' % id for id in CO)
    CD = ",".join('%s' % id for id in CD)
    cargo_origin_destination = int(dt.getDistance(VO, VD)) // 1000
    vehicle_cargo_origin = int(dt.getDistance(VO, CO)) // 1000
    vehicle_cargo_destination = int(dt.getDistance(VD, CD)) // 1000
    flowMatchDegree = cargo_origin_destination / (
            cargo_origin_destination + vehicle_cargo_origin + vehicle_cargo_destination)
    return flowMatchDegree


# 时间匹配度
def timeMatch(VS, VD, CS, CD):
    """

     :param VS: 车辆进入时间
     :param VD:车辆匹配截至时间
     :param CS: 货物进入时间
     :param CD: 货物截止时间
     :return:
     """
    # 初始化时间匹配度
    timeMatchDegree = 0
    # vS = 2019050709
    # vD = 2019051915
    # cS = 2019051212
    # cD = 2019052516
    # 年月日以列表的形式存储每个时间
    VS_list = [int(str(VS)[0:4]), int(str(VS)[4:6]), int(str(VS)[6:8])]
    VD_list = [int(str(VD)[0:4]), int(str(VD)[4:6]), int(str(VD)[6:8])]
    CS_list = [int(str(CS)[0:4]), int(str(CS)[4:6]), int(str(CS)[6:8])]
    CD_list = [int(str(CD)[0:4]), int(str(CD)[4:6]), int(str(CD)[6:8])]
    # 最小的截至时间（车辆或者货物的最晚的截至时间）
    min_deadline = min(VD, CD)
    # 最大开始时间（车辆或者货物的最大开始时间）
    max_start = max(VS, CS)
    # 最小截至时间列表
    min_deadline_list = [int(str(min_deadline)[0:4]), int(str(min_deadline)[4:6]), int(str(min_deadline)[6:8])]
    # 最大开始时间列表
    max_start_list = [int(str(max_start)[0:4]), int(str(max_start)[4:6]), int(str(max_start)[6:8])]
    # d1-d2表示最小的截止时间减去最大的开始时间
    # d3-d4表示货物的开始时间和截至时间之间的天数
    d1 = datetime.date(min_deadline_list[0], min_deadline_list[1], min_deadline_list[2])
    d2 = datetime.date(max_start_list[0], max_start_list[1], max_start_list[2])
    d3 = datetime.date(CS_list[0], CS_list[1], CS_list[2])
    d4 = datetime.date(CD_list[0], CD_list[1], CD_list[2])
    interval_days = (d1 - d2).days
    cargo_interval_days = (d3 - d4).days
    if interval_days > 0:
        timeMatchDegree = interval_days / cargo_interval_days
    else:
        # 否则可能会超时造成时间节点没有交集
        timeMatchDegree = 0
    return timeMatchDegree


# 车辆属性概率分布
def vehicleAttrDis(vehicle_queue):
    # 车辆的数量
    length = len(vehicle_queue)
    # 类型列表
    type_list = []
    # 质量列表
    quality_list = []
    # 体积列表
    volumn_list = []
    # 起始地列表
    start_list = []
    # 目的地列表
    destination_list = []
    # 时间列表
    time_list = []
    # 将车辆中的对应所有出现的属性放入对应的属性列表
    for item, value in vehicle_queue.items():
        type_list.append(value[0])
        quality_list.append(value[1])
        volumn_list.append(value[2])
        start_list.append(value[3])
        destination_list.append(value[4])
        time_list.append(value[5])
    # type_count 存放所有类型分别出现的次数
    type_count = Counter(type_list)
    # quality_count 存放不同质量出现的次数
    quality_count = Counter(quality_list)
    # 存放不同体积出现的次数
    volumn_count = Counter(volumn_list)

    # start_count表示起始地相同的有几个
    def cal_start_count(abc):
        start_count = 0
        for start in start_list:
            if abc == start:
                start_count += 1
        return start_count

    # destination_count表示目的地相同的有几个
    def cal_destination_count(abc):
        destination_count = 0
        for destination in destination_list:
            if abc == destination:
                destination_count += 1
        return destination_count

    # time_count 表示起始、截至时间相同的个数
    def cal_time_count(abc):
        time_count = 0
        for time in time_list:
            if abc == time:
                time_count += 1
        return time_count

    # 存放每一辆车的各个具体属性的分布
    attr_distri_list = []
    # 计算每辆车的属性分布
    for item, value in vehicle_queue.items():
        # 车辆的对应的各个属性分布
        vehicle_attr = []
        # 每一辆车的类型占总车辆的概率分布
        type_distri = type_count[value[0]] / length
        # 每一辆车对应的质量概率分布
        quality_distri = quality_count[value[1]] / length
        # 体积分布
        volumn_distri = volumn_count[value[2]] / length
        # 起始地相同分布
        start_distri = cal_start_count(value[3]) / length
        # 目的地相同分布
        destination_distri = cal_destination_count(value[4]) / length
        # 时间相同分布
        time_distri = cal_time_count(value[5]) / length
        vehicle_attr.append(type_distri)
        vehicle_attr.append(quality_distri)
        vehicle_attr.append(quality_distri)
        vehicle_attr.append(volumn_distri)
        vehicle_attr.append(start_distri)
        vehicle_attr.append(destination_distri)
        vehicle_attr.append(time_distri)
        # 每一辆车的各个属性对应概率分布
        # 每一个属性占所有车辆个数的比例
        attr_distri_list.append(vehicle_attr)
    # 所有属性
    total_attr = 0
    for i in attr_distri_list:
        total_attr += sum(i)

    # 存放每一辆车的整体属性分布
    vehicle_attr_list = []
    for i in attr_distri_list:
        vehicle_attr_list.append(sum(i))
    # 车辆的属性概率（先验概率）最终的结果每种属性占所有属性的概率值
    # round函数四舍五入的方法 round(数值，2)取小数点前两位
    attr_distribution_final_all = [round(i / total_attr, 2) for i in vehicle_attr_list]
    return attr_distribution_final_all


# 货物属性的先验概率
def cargoAttrDis(cargo_queue):
    length = len(cargo_queue)
    type_list = []
    quality_list = []
    volumn_list = []
    start_list = []
    destination_list = []
    time_list = []
    for item, value in cargo_queue.items():
        type_list.append(value[0])
        quality_list.append(value[1])
        volumn_list.append(value[2])
        start_list.append(value[3])
        destination_list.append(value[4])
        time_list.append(value[5])

    # type_count各种类型出现的次数
    type_count = Counter(type_list)
    # quality_count存放各种质量出现的次数
    quality_count = Counter(quality_list)
    # volumn_count存放各种体积出现的次数
    volumn_count = Counter(volumn_list)

    # start_count表示起始地相同的有几个
    def cal_start_count(abc):
        # 初始化
        start_count = 0
        for start in start_list:
            if abc == start:
                start_count += 1
        return start_count

    # destination_count表示目的地相同的有几个
    def cal_destination_count(abc):
        # 初始化
        destination_count = 0
        for destination in destination_list:
            if destination == abc:
                destination_count += 1
        return destination_count

    # time_count表示起始时间和截至时间相同的个数
    def cal_time_count(abc):
        # 初始化
        time_count = 0
        for time in time_list:
            if time == abc:
                time_count += 1

        return time_count

    # 存放每一个货物的各个属性的具体分布
    attr_distri_list = []
    # 计算每个货物的属性分布
    for item, value in cargo_queue.items():
        vehicle_attr = []
        type_distri = type_count[value[0]] / length  # 每一个类型占总货物辆的概率分布
        qualisty_distri = quality_count[value[1]] / length
        volumn_distri = volumn_count[value[2]] / length
        start_distri = cal_start_count(value[3]) / length
        destination_distri = cal_destination_count(value[4]) / length
        time_distri = cal_time_count(value[5]) / length
        vehicle_attr.append(type_distri)  # 每一个类型的概率分布
        vehicle_attr.append(qualisty_distri)
        vehicle_attr.append(volumn_distri)
        vehicle_attr.append(start_distri)
        vehicle_attr.append(destination_distri)
        vehicle_attr.append(time_distri)
        attr_distri_list.append(vehicle_attr)  # 各个属性的概率分布

    total_attr = 0
    for i in attr_distri_list:
        total_attr += sum(i)

    # 存放每一个货物的整体属性的分布
    vehicle_attr_list = []
    for i in attr_distri_list:
        vehicle_attr_list.append(sum(i))

    # 货物属性分布概率（先验结果）
    attr_distribution_final_all = [round(i / total_attr, 2) for i in vehicle_attr_list]
    return attr_distribution_final_all


a = ['底栏车', 8000, 40, [108.322574, 22.833533], [113.263955, 23.154211], [2019050709, 2019051915]]
b = ['日用百货', 5000, 30, [108.322574, 22.833533], [113.263955, 23.154211], [2019051212, 2019052516]]
c = ['日用百货', 5000, 50, [109.395618, 24.315365], [113.263955, 22.833533], [2019051109, 2019052314]]


# 计算属性匹配度

def VCM(a, b, w1=0.16, w2=0.09, w3=0.47, w4=0.28):
    # 类型匹配度
    typeMatch = getTypeMatch(a[0], b[0])
    # 体积或质量匹配度
    gvMatch = getQualityVolumnMatch(a[1], a[2], b[1], b[2])
    # 路径匹配度
    flowMatch = getFlowMatch(a[3], a[4], b[3], b[4])
    # 时间匹配度
    timeMatchs = timeMatch(a[5][0], a[5][1], b[5][0], b[5][1])
    # 属性匹配度
    attrMatch = w1 * typeMatch + w2 * gvMatch + w3 * flowMatch + w4 * timeMatchs
    # 对于属性匹配度，保留两位小数
    attrMatch = round(attrMatch, 2)
    # 如果任何一个属性匹配度为为0 则不满足匹配的条件
    if typeMatch == 0 or gvMatch == 0 or flowMatch == 0 or timeMatchs == 0:
        attrMatch = 0
    return attrMatch


def rePaperVCM(a, b, w1=0.16, w2=0.09, w3=0.47, w4=0.28):
    typeMatch = getTypeMatch(a[0], b[0])
    gvMatch = getQualityVolumnMatch(a[1], a[2], b[1], b[2])
    flowMatch = getFlowMatch(a[3], a[4], b[3], b[4])
    timeMatchs = timeMatch(a[5][0], a[5][1], b[5][0], b[5][1])
    # 将属性匹配度表示为连乘积的形式
    attrMatch = typeMatch * gvMatch * flowMatch * timeMatchs
    attrMatch = round(attrMatch)
    if typeMatch == 0 or gvMatch == 0 or flowMatch == 0 or timeMatchs == 0:
        attrMatch = 0
    return attrMatch


# 从数据集中读取数据并添加到vehicle_queue
def appendVehicle(vehicle_df, vehicle_max_Num, vehicle_queue, t):
    # 如果车辆队列中的车辆少于10辆，那么读取数据集并添加至少10辆车
    if (len(vehicle_queue) < 10):
        for index, row in vehicle_df.iterrows():  # .iterrows()返回行索引及包含行本身的对象
            vehicle_attr = []
            # 上一次vehicle_queue中存放着的车辆的最大编号是num,那么当车辆数少于10辆的时候，应该从num+1开始读数据
            if t == 0:
                num = 0
            else:
                num = vehicle_max_Num[t - 1]
            # 0是表头
            if index > num:
                for i in range(0, len(row)):  # 遍历每一行的元素
                    if i == 1:
                        row[i] = int(row[i])  # 质量
                    if i == 2:
                        row[i] = int(row[i])  # 体积
                    if i == 3 or i == 4:  # 路径
                        row[i] = (row[i])[1:-1]
                        row[i] = row[i].split(',')
                        for j in range(0, len(row[i])):
                            row[i][j] = float(row[i][j])  # 时间

                    if i == 5:
                        row[i] = (row[i])[1:-1]
                        row[i] = row[i].split(',')
                        for j in range(0, len(row[i])):
                            row[i][j] = int(row[i][j])
                    # 每一个车辆的属性
                    vehicle_attr.append(row[i])
                    # 将车辆的信息添加至车辆队列中
                vehicle_queue[index] = vehicle_attr
                # 保存上个时间片添加的车辆的最大的编号
                if len(vehicle_queue) == 10:
                    vehicle_max_Num.append(index)
                    break


# 从数据集中读取数据添加到cargo_queue
# 参数1：数据集的df形式，2：队列中最大的货物编号，3：货物队列，4：时间
def appendCargo(cargo_df, cargo_maxNum, cargo_queue, t):
    # 如果货物的数量少于十个，则读取数据集并添加至十个货物
    if (len(cargo_queue)) < 10:
        for index, row in cargo_df.iterrows():
            cargo_attr = []
            # 上一次vehicle_queue存储的货物的最大编号为num那么少于十个的时候就要添加至十个，从num+1开始读取
            if t == 0:  # 初始队列的最大标记
                num = 0
            else:
                num = cargo_maxNum[t - 1]

            # 0为表头 将数据集中的数据添加到货物队列
            if index > num:  # 将数据集中的数据类型做转换
                for i in range(0, len(row)):
                    if i == 1:
                        row[i] = int(row[i])
                    if i == 2:
                        row[i] = int(row[i])
                    if i == 3 or i == 4:  # 时间
                        row[i] = (row[i])[1:-1]
                        row[i] = row[i].split(',')
                        for j in range(0, len(row[i])):
                            row[i][j] = float(row[i][j])
                    if i == 5:
                        row[i] = (row[i])[1:-1]
                        row[i] = row[i].split(',')
                        for j in range(0, len(row[i])):
                            row[i][j] = int(row[i][j])

                    cargo_attr.append(row[i])
                cargo_queue[index] = cargo_attr
                if len(cargo_queue) == 10:
                    cargo_maxNum.append(index)
                    break
