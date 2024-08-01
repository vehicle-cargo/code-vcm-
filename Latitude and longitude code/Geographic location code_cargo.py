import json
import os

import pandas as pd


def isPointinPolygon(point, rangelist, apices):  # [[0,0],[1,1],[0,1],[0,0]] [1,0.8]
    # # 判断是否在外包矩形内，如果不在，直接返回false
    maxlng = apices["maxlng"]
    minlng = apices["minlng"]
    maxlat = apices["maxlat"]
    minlat = apices["minlat"]
    # print(maxlng, minlng, maxlat, minlat)
    if (point[0] > maxlng or point[0] < minlng or
            point[1] > maxlat or point[1] < minlat):
        return False
    for range_i in rangelist:
        if In(point, range_i):
            return True
    return False


def In(point, rangelist):
    if len(rangelist) < 3:
        return False
    a = rangelist[0]
    res = rayIntersectsSegment(point, rangelist[len(rangelist) - 1], a)
    for b in rangelist[1:]:
        if rayIntersectsSegment(point, a, b):
            res = not res
        a = b
    return res


def rayIntersectsSegment(point, a, b):
    return (a[1] > point[1]) != (b[1] > point[1]) and point[0] < (b[0] - a[0]) * (point[1] - a[1]) / (b[1] - a[1]) + a[
        0]


def loadOriginalJson(filename):
    fin = open(filename, encoding="utf-8")
    data_json = json.load(fin)
    return data_json


def Orientation(point):
    provice_name = ""
    city_name = ""
    file_path = "../dataset/省市边界线.json"
    data_json = loadOriginalJson(file_path)
    for provice in data_json["provice"]:
        if isPointinPolygon(point, provice["coordinates"], provice["apices"]):
            provice_name = provice["name"]
            for city in provice["subordinate"]:
                if isPointinPolygon(point, city["coordinates"], city["apices"]):
                    city_name = city["name"]
    return provice_name, city_name


if __name__ == "__main__":
    data = [[108.322574, 22.833533],
            [113.263955, 23.154211],
            [109.395618, 24.315365],
            [113.263955, 23.154211],
            [108.322574, 22.833533],
            [113.263955, 23.154211],
            [110.28662, 25.267723],
            [108.322574, 22.833533],
            [109.395618, 24.315365],
            [113.263955, 22.833533],
            [113.25872, 23.139562],
            [113.51597, 22.292177],
            [113.25872, 23.139562],
            [114.051164, 22.609383],
            [113.263955, 23.154211],
            [110.28662, 25.267723],
            [108.322574, 22.833533],
            [113.263955, 23.154211],
            [116.425052, 39.934032],
            [113.51597, 22.292177],
            [113.104074, 36.215097],
            [116.425052, 39.934032],
            [116.425052, 39.934032],
            [113.51597, 22.292177],
            [112.85052, 35.493965],
            [116.425052, 39.934032],
            [113.362213, 40.097111],
            [116.425052, 39.934032],
            [113.25872, 23.139562],
            [104.060293, 30.593689],
            [106.535893, 29.590094],
            [104.060293, 30.593689],
            [111.019703, 35.033296],
            [103.839542, 36.071046],
            [113.104074, 36.215097],
            [112.544412, 37.881898],
            [117.208789, 39.095388],
            [113.25872, 23.139562],
            [113.104074, 36.215097],
            [116.425052, 39.934032],
            [111.019703, 35.033296],
            [103.839542, 36.071046],
            [113.434202, 22.519376],
            [108.655118, 22.060841],
            [113.263955, 23.154211],
            [108.322574, 22.833533],
            [106.535893, 29.590094],
            [104.060293, 30.593689],
            [109.395618, 24.315365],
            [113.263955, 23.154211],
            [117.208789, 39.095388],
            [113.25872, 23.139562],
            [113.104074, 36.215097],
            [116.425052, 39.934032],
            [113.263955, 23.154211],
            [108.322574, 22.833533],
            [106.535893, 29.590094],
            [104.060293, 30.593689],
            [109.395618, 24.315365],
            [113.263955, 22.833533],
            [112.544412, 37.881898],
            [113.104074, 36.215097],
            [113.104074, 36.215097],
            [112.544412, 37.881898],
            [112.85052, 35.493965],
            [117.208789, 39.095388],
            [113.25872, 23.139562],
            [114.051164, 22.609383],
            [112.544412, 37.881898],
            [113.104074, 36.215097],
            [112.85052, 35.493965],
            [117.208789, 39.095388],
            [106.535893, 29.590094],
            [104.060293, 30.593689],
            [113.25872, 23.139562],
            [104.060293, 30.593689],
            [113.104074, 36.215097],
            [112.544412, 37.881898],
            [111.019703, 35.033296],
            [103.839542, 36.071046]
            ]
    for i in range(len(data)):
        print(i, Orientation(data[i]))
