from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
start_lng = -8.652
start_lat = 41.142
end_lng = -8.578
end_lat = 41.174

interval = 64

lng_interval = abs(end_lng - start_lng) / interval   
log_interval = abs(end_lat - start_lat) / interval



def cal_lng_lat(lng, lat):
    if lng>=start_lng and lng < end_lng and lat>=start_lat and lat<end_lat:
        latitude=int(np.floor(abs(lat-start_lat) / log_interval))
        longitude=int(np.floor(abs(lng-start_lng) / lng_interval))
        return True, longitude, latitude
    else:
        return False, 0, 0

def cal_flow(file_path):
    res_data = np.zeros((24, interval, interval))
    for filename in tqdm(os.listdir(file_path)):
        txt_file = os.path.join(file_path, filename)
        with open(txt_file, 'r') as f:
            for line in f.readlines():
                attrs = line.rstrip().split(',')
                if attrs[0] == '#':
                    last_lng_index = -1
                    last_lat_index = -1
                    last_time_index = -1
                    continue
                else:
                    # print(filename)
                    time = attrs[0].split(" ")
                    hour = int(time[1].split(":")[0])
                    if attrs[5] == 'None':continue
                    lng = float(attrs[5])
                    lat = float(attrs[4])
                    loc_flag, lng_index, lat_index = cal_lng_lat(lng, lat)  #计算轨迹点在哪一个网格

                    if loc_flag == False:
                        continue
                    if lng_index == last_lng_index and lat_index == last_lat_index and hour == last_time_index:
                        continue
                    last_lng_index = lng_index
                    last_lat_index = lat_index    #在一个时间段内，相邻时刻，只记录一次车辆
                    last_time_index = hour

                    res_data[hour, lng_index, lat_index] += 1
    return res_data
    
# 生成路况数据
if __name__ == '__main__':
    dataset = "Porto"
    file_path = "/data/WeiTongLong/trajectory/pre_process/data/{}/train/".format(dataset)
    res_data = cal_flow(file_path)
    np.save("flow.npy", res_data)
