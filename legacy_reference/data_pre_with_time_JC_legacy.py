# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:27:49 2024
生成的数据集要结合天和周的时间编码
@author: 天天向上
"""
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import pickle 
from math import radians, sin, cos, sqrt, atan2

# 读取8月的CSV文件:利用8月的数据，来推出9月初的数据分布
file_path = 'data/huaqi20190809/JC-201908-citibike-tripdata.csv'  
data_1 = pd.read_csv(file_path,dtype={'start station latitude': str, 'start station longitude': str,'end station latitude': str,'end station longitude':str})
# 读取9月的CSV文件
file_path = 'data/huaqi20190809/JC-201909-citibike-tripdata.csv' 
data_2 = pd.read_csv(file_path,dtype={'start station latitude': str, 'start station longitude': str,'end station latitude': str,'end station longitude':str})

# 清洗数据
# 检测经纬度是否有超出（对于无桩的要进行这一步操作）有桩的停在站点。这里省略
# 删除重复数据
data_1 = data_1.drop_duplicates()
data_2 = data_2.drop_duplicates()
# 设置时间格式
time_format = "%Y-%m-%d %H:%M:%S.%f"
# 转换 'starttime' 和 'stoptime' 列为 datetime 类型，使用提供的时间格式
data_1['starttime'] = pd.to_datetime(data_1['starttime'], format=time_format)
data_1['stoptime'] = pd.to_datetime(data_1['stoptime'], format=time_format)
data_2['starttime'] = pd.to_datetime(data_2['starttime'], format=time_format)
data_2['stoptime'] = pd.to_datetime(data_2['stoptime'], format=time_format)
# 删除骑行时间小于0且大于24小时的单车数量，这里只删除9月份的数据，8月的数据用来确定单车的分布
data_2 = data_2[(data_2['tripduration'] >= 0) & (data_2['tripduration'] <= 86400)]# 49244-49228少了16辆车
# 重置索引
data_2.reset_index(drop=True, inplace=True)

# 单车总数量
bike_numbers_1 = data_1['bikeid'].unique()
bike_numbers_2 = data_2['bikeid'].unique()
print("8月单车数:", len(bike_numbers_1)) # 8月单车数: 483
print("9月单车数:", len(bike_numbers_2)) # 9月单车数: 494
bike_list_1 = list(set(data_1['bikeid'].tolist()))
bike_list_2 = list(set(data_2['bikeid'].tolist()))

# 加载每个月的站点信息
# 按边提取共享单车起点站与终点站的站名
stationlist_1 = list(set(data_1['start station id'].tolist()+data_1['end station id'].tolist()))
stationlist_2 = list(set(data_2['start station id'].tolist()+data_2['end station id'].tolist()))
print("8月站点个数:", len(stationlist_1)) # 8月站点个数: 62 出站点52 入站点62（奇怪）
print("9月站点个数:", len(stationlist_2)) # 9月站点个数: 67 出站点51 入站点67

#加载站点的经纬度坐标
# 构建8月的站点信息字典
station_info_1 = {station: (data_1.loc[data_1['end station id'] == station, 'end station latitude'].values[0], 
                              data_1.loc[data_1['end station id'] == station, 'end station longitude'].values[0])
                   for station in stationlist_1}

# 构建9月的站点信息字典
station_info_2 = {station: (data_2.loc[data_2['end station id'] == station, 'end station latitude'].values[0], 
                              data_2.loc[data_2['end station id'] == station, 'end station longitude'].values[0])
                   for station in stationlist_2}




'''获得初始分布-------------------------------------------------------------------------------------'''
# 数据拼接，将两个月的数据拼接到一起，用前一个月的数据
df = pd.concat([data_1, data_2], axis=0, ignore_index=True)

# 将时间分为两整段，第一阶段用来推导第二阶段的单车的初始分布，但是需要获取第二阶段的某些单车的首次使用。
start_time = datetime(2019,8,1,0,0,0)
temple_time = datetime(2019,8,31,23,59,59)

bike_loc = {} # 记录单车当前停留位置
bike_secondor = {} # 记录该单车在第二阶段是否首次出现
station_count = {}

for index, row in df.iterrows():
    start_station = row["start station id"]
    end_station = row["end station id"]
    borrow_time  = row["starttime"]
    return_time = row["stoptime"]
    bike_id = row["bikeid"]
    # 如果归还时间在第一阶段
    if return_time < temple_time:
        bike_loc[bike_id] = end_station # 第一阶段终点即为单车定位，会随着记录不断更新。
    else:
        if bike_id in bike_secondor:
            bike_secondor[bike_id] = 0 # 如果单车不是第二阶段首次出现，则标签为0
        else:
            bike_secondor[bike_id] = 1 # 如果单车是第二阶段首次出现，则标签为1
            bike_loc[bike_id] = start_station # 第二阶段起点即为单车定位。

# 删除非第二阶段的车辆
for bike in bike_loc.copy():
    if bike not in bike_list_2:
        del bike_loc[bike]

# 遍历单车定位字典的值
for value in bike_loc.values():
  # 如果该站点已经存在于站点数量字典中，则增加对应站点的计数
  if value in station_count:
      station_count[value] += 1
  # 如果值不存在于站点数量字典中，则添加该站点，并初始化站点单车数量为1
  else:
      station_count[value] = 1

# 补充站点单车数量为0的站点
for station in stationlist_2:
    if station not in  station_count:
        station_count[station] = 0

# 输出第二阶段各站点单车初始数量以及校验单车总数量
counts = 0
for station, count in station_count.items():
    counts += count
    print(f"站点 {station}: {count} 辆共享单车")
print(f"总共 {counts} 辆共享单车") # 9月站点67，单车数量494

# 保存9月初的站点自行车数量分布
with open('station_count_09chu.pkl', 'wb') as file:
    # 使用pickle.dump()函数保存字典
    pickle.dump(station_count, file)
# 保存9月初的自行车分布
with open('bike_loc_09chu.pkl', 'wb') as file:
    # 使用pickle.dump()函数保存字典
    pickle.dump(bike_loc, file)


plt.figure(figsize=(10, 6))  # 设置图形的大小
# 绘制条形图，使用站点名称作为x轴标签
plt.bar(range(len(station_count)), list(station_count.values()), tick_label=list(station_count.keys()))
# 添加标题和标签
plt.title('各站点共享单车数量')
plt.xlabel('站点')
plt.ylabel('单车数量')
# 旋转x轴标签以便它们不会重叠
plt.xticks(rotation=90)
# 显示图形
plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
plt.show()


'''根据骑行记录建立距离邻接矩阵-------------------------------------------------------------------------------------'''

# 创建节点和经纬度坐标的映射字典
node_coordinates = station_info_2
# 将节点变成整数型，便于操作
node_coordinates = {int(key): value for key, value in station_info_2.items()}

# 提取所有唯一的节点编号
all_nodes = set()
all_nodes.update(node_coordinates.keys())
# 对节点编号进行排序并建立映射关系
node_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(all_nodes), start=0)}
# 创建图
G = nx.Graph()
# 添加节点
G.add_nodes_from(node_coordinates.keys())


# 提取所有唯一的节点编号:加0
node_coordinates[int(0)]=0
all_nodes_0 = set()
all_nodes_0.update(node_coordinates.keys())
# 对节点编号进行排序并建立映射关系
node_mapping_0 = {old_id: new_id for new_id, old_id in enumerate(sorted(all_nodes_0), start=0)}
# 创建图
G_0 = nx.Graph()
# 添加节点
G_0.add_nodes_from(node_coordinates.keys())

# 定义添加边的函数
def add_or_update_edge(graph, u, v):
    # 检查边是否存在
    if graph.has_edge(u, v):
        # 获取当前边的权重
        current_weight = graph[u][v].get('weight', 0)
        # 更新权重
        graph[u][v]['weight'] = current_weight + 1
    else:
        # 添加新边并设置权重为1
        graph.add_edge(u, v, weight=1)






'''根据初始分布推导流量数据-------------------------------------------------------------------------------------'''
# 获取第二阶段的每个时间窗的流量，以及每个时间窗的动态邻接矩阵，考虑添加骑行零站点和不添加零站点

# 定义起始时间和结束时间
begin_time = datetime(2019,9,1,0,0,0)  
end_time = datetime(2019,10,1,0,0,0)  
# end_time = datetime(2019,10,1,0,0,0)  

# 每指定时间窗分钟进行划分的时间间隔
interval = timedelta(minutes=5) # 5分钟一次
# 生成时间序列字典
time_list = [begin_time + interval * i for i in range(int((end_time - begin_time).total_seconds() // interval.total_seconds()))]
time_dict = {item: None for item in time_list}
time_dict[begin_time] = station_count

# 建立添加0站点的字典
station_count_0 = station_count.copy()
station_count_0[0] = 0 # 添加0站点
time_dict_0 = {item: None for item in time_list}
time_dict_0[begin_time] = station_count_0 # 建立两个时间序列字典，分别用来保存加0 和不加0的站点流量记录
bike_loc_0 = bike_loc.copy()

df = data_2 # 使用第二阶段数据
bike_move = {} # 单车本次记录下的起讫点
bike_move_pre = {} # 单车上次记录的起讫点
bike_time = {} # 单车本次次使用的起终点时间
bike_time_pre = {} # 单车上一次使用的起终点时间
# bike_loc = {} # 记录单车当前停留位置
rebalance_station_time = {} # 记录车辆的搬迁站点和搬迁时间
'''注意，之前的bike_loc是有用的，保留了九月初的分布'''

# 使用for循环遍历时间序列
for current_time in time_list[:-1]:
    # 遍历时间段内的所有单车记录,筛选符合条件的行
    filtered_df = df[(df['stoptime'] > current_time) & (df['starttime'] < current_time+interval)]
    # print(filtered_df.index)
    # 按照'starttime'列升序排序
    # filtered_df = filtered_df.sort_values('starttime', ascending=True)
    for index, row in filtered_df.iterrows():
        # print(index)这里的index其实对应着数据集df的index
        start_station = row["start station id"]
        end_station = row["end station id"]
        borrow_time = row["starttime"]
        return_time = row["stoptime"]
        bike_id = row["bikeid"]
        #记录当前单车当前记录下的起讫点和使用时间
        bike_move[bike_id] = [start_station,end_station]
        bike_time[bike_id] = [borrow_time,return_time]
        # 获得对应数据集df的index
        # 使用 & 运算符来确保两个条件都满足
        # indices = df.loc[(df['starttime'] == borrow_time) & (df['bikeid'] == bike_id)]
        # print(indices.index)
        #在这个区间内的流量分三种情况
        if  return_time >= current_time and return_time <= current_time+interval:
            bike_loc_0[bike_id] = end_station
            #new_station_count[end_station] += 0
            # 找到下一次出现bike_id的位置，看是不是被搬运,如果被搬运则更新loc
            matching_rows = df[(df['bikeid'] == bike_id) & (df.index > index)]
            if not matching_rows.empty:
                # print("车辆被再次使用")
                # 不需要保证这条记录不在组内，后面会循环到。
                next_index = matching_rows.index[0]  # 获取匹配行的第一个索引
                new_start_station = df.loc[next_index, 'start station id']
                new_start_time = df.loc[next_index, 'starttime']
                # 如果下一条记录的起点与当前记录的终点不同，并且下次记录的起点在一个时间窗范围内。
                if new_start_station != end_station and new_start_time<current_time+2*interval:
                    print("车辆被搬运过")
                    bike_loc_0[bike_id] = new_start_station     
            
        elif borrow_time > current_time and return_time > current_time+interval:
            bike_loc_0[bike_id] = 0
            # bike_loc[bike_id] = start_station
            add_or_update_edge(G_0, start_station, 0)
            
        else:
            # bike_loc[bike_id] = start_station
            bike_loc_0[bike_id] = 0
            add_or_update_edge(G_0, 0, end_station)
            #print(borrow_time,start_station,return_time,end_station,current_time)
            
            
        bike_move_pre[bike_id] = [start_station,end_station] #记录本次的起终点，留下次使用    
        bike_time_pre[bike_id] = [borrow_time,return_time] #记录本次的起终点时间
    #
    new_station_count = {}
    # 遍历单车定位字典的值
    for value in bike_loc_0.values():
      # 如果值已经存在于新字典中，则增加对应值的计数
      if value in new_station_count:
          new_station_count[value] += 1
      # 如果值不存在于新字典中，则添加该值，并初始化计数为1
      else:
          new_station_count[value] = 1
    for station in stationlist_2:
        if station not in new_station_count:
            new_station_count[station] = 0
    if 0 not in new_station_count:
        new_station_count[0] = 0
    time_dict_0[current_time+interval] = new_station_count

# 遍历字典，找到值对应的字典长度小于68的键值对
for key, value_dict in time_dict_0.items():
    if len(value_dict) < 68:
        print(f"找到键值对：键 '{key}' 对应的字典长度为 {len(value_dict)}")
    total_sum = sum(value_dict.values())  # 计算子字典所有值的总和
    if total_sum != 494:   # 检查总和是否不等于494
        print(f"键 '{key}' 对应的字典值之和为 {total_sum}，不等于494")



# 生成不添加0站点的流量

bike_move = {} # 单车本次记录下的起讫点
bike_move_pre = {} # 单车上次记录的起讫点
bike_time = {} # 单车本次次使用的起终点时间
bike_time_pre = {} # 单车上一次使用的起终点时间
# bike_loc = {} # 记录单车当前停留位置
rebalance_station_time = {} # 记录车辆的搬迁站点和搬迁时间
'''注意，之前的bike_loc是有用的，保留了九月初的分布'''

# 使用for循环遍历时间序列
for current_time in time_list[:-1]:
    # 遍历时间段内的所有单车记录,筛选符合条件的行
    filtered_df = df[(df['stoptime'] > current_time) & (df['starttime'] < current_time+interval)]
    # print(filtered_df.index)
    # 按照'starttime'列升序排序
    # filtered_df = filtered_df.sort_values('starttime', ascending=True)
    for index, row in filtered_df.iterrows():
        # print(index)这里的index其实对应着数据集df的index
        start_station = row["start station id"]
        end_station = row["end station id"]
        borrow_time = row["starttime"]
        return_time = row["stoptime"]
        bike_id = row["bikeid"]
        #记录当前单车当前记录下的起讫点和使用时间
        bike_move[bike_id] = [start_station,end_station]
        bike_time[bike_id] = [borrow_time,return_time]
        # 获得对应数据集df的index
        # 使用 & 运算符来确保两个条件都满足
        # indices = df.loc[(df['starttime'] == borrow_time) & (df['bikeid'] == bike_id)]
        # print(indices.index)
        #在这个区间内的流量分三种情况
        if  return_time >= current_time and return_time <= current_time+interval:
            bike_loc[bike_id] = end_station
            #new_station_count[end_station] += 0
            # 找到下一次出现bike_id的位置，看是不是被搬运,如果被搬运则更新loc
            matching_rows = df[(df['bikeid'] == bike_id) & (df.index > index)]
            if not matching_rows.empty:
                # print("车辆被再次使用")
                # 不需要保证这条记录不在组内，后面会循环到。
                next_index = matching_rows.index[0]  # 获取匹配行的第一个索引
                new_start_station = df.loc[next_index, 'start station id']
                new_start_time = df.loc[next_index, 'starttime']
                # 如果下一条记录的起点与当前记录的终点不同，并且下次记录的起点在一个时间窗范围内。
                if new_start_station != end_station and new_start_time<current_time+2*interval:
                    print("车辆被搬运过")
                    bike_loc[bike_id] = new_start_station     
            
        elif borrow_time > current_time and return_time > current_time+interval:
            # 赋值为空表示该单车当前没有落在任何站点
            bike_loc[bike_id] = None
        else:
            bike_loc[bike_id] = None
            
            
        bike_move_pre[bike_id] = [start_station,end_station] #记录本次的起终点，留下次使用    
        bike_time_pre[bike_id] = [borrow_time,return_time] #记录本次的起终点时间
    #
    new_station_count = {}
    # 遍历单车定位字典的值
    for value in bike_loc.values():
        # 跳过 None 值
        if value is None:
            continue
          # 如果值已经存在于新字典中，则增加对应值的计数
        if value in new_station_count:
            new_station_count[value] += 1
          # 如果值不存在于新字典中，则添加该值，并初始化计数为1
        else:
            new_station_count[value] = 1
    for station in stationlist_2:
        if station not in new_station_count:
            new_station_count[station] = 0
    time_dict[current_time+interval] = new_station_count

# 遍历字典，找到值对应的字典长度小于68的键值对
for key, value_dict in time_dict.items():
    if len(value_dict) > 67:
        print(f"找到键值对：键 '{key}' 对应的字典长度为 {len(value_dict)}")
    total_sum = sum(value_dict.values())  # 计算子字典所有值的总和
    # if total_sum == 494:   # 检查总和是否不等于494
    #     print(f"键 '{key}' 对应的字典值之和为 {total_sum}，等于494")

# 构建图神经网络数据集（类似于PEMS08类型）
'''
file_path = 'PEMS08.npz'  
pems08 = np.load(file_path)
p8data = pems08['data'] #p8data.shape(17856,170,3) 17856个时刻，170个节点，特征维度为3，第一维度是流量，第三个维度是速度
first_moment_data = p8data[0]  # 第一个时刻的所有节点的数据
# 打印第一个时刻的所有节点的数据
print("第一个时刻的所有节点的数据:")
print(first_moment_data)
'''
# 构建加0节点的数据集，根据time_dict_0，创建一个所有时刻的矩阵：时刻数量*节点个数*特征维度，8640*68*3

zero_matrix = np.zeros((8640, 68))
matrices_0 = [zero_matrix.copy() for _ in range(3)]

# 初始化矩阵
matrix_encode = np.zeros(((8640, 2)))

# 获取日期时间列表
dates = sorted(time_dict.keys())

# 遍历日期时间列表
for i, date in enumerate(dates):
    # 计算一天中的时间编码
    time_of_day = (date.hour * 60 + date.minute) / 1440.0
    # 计算一周中的日子编码
    day_of_week = (date.weekday()+1) / 7.0  # weekday() 返回的是 0-6，转换为 1-7
    
    # 填充矩阵
    matrix_encode[i, 0] = time_of_day
    matrix_encode[i, 1] = day_of_week

# 打印矩阵的前几行以检查
print(matrix_encode[:1441, :])

# 填充矩阵
time_index = 0
start_date = datetime(2019, 9, 1, 0, 0, 0)
for timestamp, node_values in sorted(time_dict_0.items()):
    # 将时间戳转换为矩阵中的时间索引
    time_delta = timestamp - start_date
    current_time_index = time_delta // timedelta(minutes=5)
    #print(current_time_index)
    # 填充当前时间点的矩阵
    i = 0
    for node_id, value in  sorted(node_values.items()):
        #print(node_id)
        matrices_0[0][current_time_index, i] = value
        i += 1
    # 更新时间索引
    time_index += 1

# 将 matrix_encode[:, 0] 复制 68 次形成新的数组
matrices_0[1] = np.repeat(matrix_encode[:, 0], 68).reshape(8640, -1)
matrices_0[2] = np.repeat(matrix_encode[:, 1], 68).reshape(8640, -1)
# 堆叠三个矩阵以形成最终的 8640x68x3 矩阵
final_matrix = np.stack(matrices_0, axis=-1)
# 将矩阵保存为npz文件
np.savez('huaqi_09_alldays_0_time.npz', data=final_matrix)

# 构建无0节点的数据集，根据time_dict
zero_matrix = np.zeros((8640, 67))
matrices = [zero_matrix.copy() for _ in range(3)]

# 填充矩阵
time_index = 0
start_date = datetime(2019, 9, 1, 0, 0, 0)
for timestamp, node_values in sorted(time_dict.items()):
    # 将时间戳转换为矩阵中的时间索引
    time_delta = timestamp - start_date
    current_time_index = time_delta // timedelta(minutes=5)
    #print(current_time_index)
    # 填充当前时间点的矩阵
    i = 0
    for node_id, value in  sorted(node_values.items()):
        matrices[0][current_time_index, i] = value
        i += 1
    # 更新时间索引
    time_index += 1

# 将 matrix_encode[:, 0] 复制 68 次形成新的数组
matrices[1] = np.repeat(matrix_encode[:, 0], 67).reshape(8640, -1)
matrices[2] = np.repeat(matrix_encode[:, 1], 67).reshape(8640, -1)
# 堆叠三个矩阵以形成最终的 8640x67x3 矩阵
final_matrix = np.stack(matrices, axis=-1)
# 将矩阵保存为npz文件
np.savez('huaqi_09_alldays_time.npz', data=final_matrix)





# 遍历数据帧，添加边
for index, row in data_2.iterrows():
    start_station = row['start station id']
    end_station = row['end station id']
    add_or_update_edge(G_0, start_station, end_station)
    add_or_update_edge(G, start_station, end_station)

G_r = nx.relabel_nodes(G, node_mapping)
G_0_r = nx.relabel_nodes(G_0, node_mapping_0)


# 计算两个经纬度坐标之间的haversine距离
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # 地球平均半径，单位为公里
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c *1000
    return distance


# 创建 node_mapping 的逆字典
inverse_node_mapping = {v: k for k, v in node_mapping.items()}
inverse_node_mapping_0 = {v: k for k, v in node_mapping_0.items()}
# 遍历图 G_r 的边并计算距离
edges_G_r = []
for u, v in G_r.edges():
    # 通过 inverse_node_mapping 获取站点ID
    u_id = inverse_node_mapping.get(u)
    v_id = inverse_node_mapping.get(v)
    lat1, lon1 = station_info_2[u_id]
    lat2, lon2 = station_info_2[v_id]
    distance = haversine(float(lat1), float(lon1), float(lat2), float(lon2))
    edges_G_r.append([u, v, distance])

# 遍历图 G_0_r 的边并计算距离
edges_G_0_r = []
for u, v in G_0_r.edges():
    if u==0 or v==0:
        edges_G_0_r.append([u, v, 0])
        continue
    u_id = inverse_node_mapping_0.get(u)
    v_id = inverse_node_mapping_0.get(v)
    lat1, lon1 = station_info_2[u_id]
    lat2, lon2 = station_info_2[v_id]
    distance = haversine(float(lat1), float(lon1), float(lat2), float(lon2))
    edges_G_0_r.append([u, v, distance])

# 创建 edges_G_r 的 DataFrame
df_G_r = pd.DataFrame(edges_G_r, columns=['from', 'to', 'cost'])

# 导出 edges_G_r 到 CSV 文件
df_G_r.to_csv('edges_data_09.csv', index=False)

# 创建 edges_G_0_r 的 DataFrame
df_G_0_r = pd.DataFrame(edges_G_0_r, columns=['from', 'to', 'cost'])

# 导出 edges_G_0_r 到 CSV 文件
df_G_0_r.to_csv('edges_data_09_0.csv', index=False)


















