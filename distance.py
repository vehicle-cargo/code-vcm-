"""
利用高德地图的api实现地址和经纬度的转变
"""
import requests


def getcode(address):
    requests.adapters.DEFAULT_RETRIES = 5
    parameters = {'address': address, 'key': '4d51d4b88017607b3571af22633a2171'}
    base = 'http://restapi.amap.com/v3/geocode/geo'
    response = requests.get(base, parameters)
    answer = response.json()
    s = requests.session()
    s.keep_alive = False
    return answer['geocodes'][0]['location']


def getDistance(address, destination):
    requests.adapters.DEFAULT_RETRIES = 5
    parameters = {'origins': address, 'destination': destination, 'key': '4d51d4b88017607b3571af22633a2171'}
    base = 'https://restapi.amap.com/v3/distance'
    response = requests.get(base, parameters)
    answer = response.json()
    s = requests.session()
    s.keep_alive = False
    return answer['results'][0]['distance']


if __name__ == '__main__':
    address = '宁夏回族自治区银川市'
    destination = '宁夏回族自治区吴忠市'
    addresscode = getcode(address)
    destinationcode = getcode(destination)
    distance = int(getDistance(addresscode, destinationcode)) // 1000
    print(distance)
