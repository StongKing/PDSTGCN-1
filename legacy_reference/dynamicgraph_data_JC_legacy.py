# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 21:23:54 2024
本代码用来生成
@author: 13670
"""

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import pickle 
from math import radians, sin, cos, sqrt, atan2


# 读取9月的CSV文件
file_path = 'data/huaqi20190809/JC-201909-citibike-tripdata.csv' 
data_2 = pd.read_csv(file_path,dtype={'start station latitude': str, 'start station longitude': str,'end station latitude': str,'end station longitude':str})

# 清洗数据
# 检测经纬度是否有超出（对于无桩的要进行这一步操作）有桩的停在站点。这里省略
# 删除重复数据
data_2 = data_2.drop_duplicates()
# 设置时间格式
time_format = "%Y-%m-%d %H:%M:%S.%f"
# 转换 'starttime' 和 'stoptime' 列为 datetime 类型，使用提供的时间格式
data_2['starttime'] = pd.to_datetime(data_2['starttime'], format=time_format)
data_2['stoptime'] = pd.to_datetime(data_2['stoptime'], format=time_format)
# 删除骑行时间小于0且大于24小时的单车数量，这里只删除9月份的数据
data_2 = data_2[(data_2['tripduration'] >= 0) & (data_2['tripduration'] <= 86400)]# 49244-49228少了16辆车
# 重置索引
data_2.reset_index(drop=True, inplace=True)

# 单车总数量
bike_numbers_2 = data_2['bikeid'].unique()
print("9月单车数:", len(bike_numbers_2)) # 9月单车数: 494
bike_list_2 = list(set(data_2['bikeid'].tolist()))

# 加载每个月的站点信息
# 按边提取共享单车起点站与终点站的站名
stationlist_2 = list(set(data_2['start station id'].tolist()+data_2['end station id'].tolist()))
print("9月站点个数:", len(stationlist_2)) # 9月站点个数: 67 出站点51 入站点67

#加载站点的经纬度坐标
# 构建9月的站点信息字典
station_info_2 = {station: (data_2.loc[data_2['end station id'] == station, 'end station latitude'].values[0], 
                              data_2.loc[data_2['end station id'] == station, 'end station longitude'].values[0])
                   for station in stationlist_2}

# 创建节点和经纬度坐标的映射字典
node_coordinates = station_info_2
# 将节点变成整数型，便于操作
node_coordinates = {int(key): value for key, value in station_info_2.items()}


# # 使用'rb'模式打开文件，表示以二进制读模式
# with open('station_count_09chu.pkl', 'rb') as file:
#     # 使用pickle.load()函数读取字典
#     station_count = pickle.load(file)
# with open('bike_loc_09chu.pkl', 'rb') as file:
#     bike_loc = pickle.load(file)

# 定义起始时间和结束时间
begin_time = datetime(2019,9,1,0,0,0)  
end_time = datetime(2019,10,1,0,0,0)  

# 每指定时间窗分钟进行划分的时间间隔
interval1 = timedelta(minutes=5) # 5分钟一次,但是动态图需要维持一个预测时间长度
interval = interval1
# 生成时间序列字典
time_list = [begin_time + interval * i for i in range(int((end_time - begin_time).total_seconds() // interval.total_seconds()))]
time_dict = {item: None for item in time_list}
#time_dict[begin_time] = station_count

# # 提取所有唯一的节点编号
# all_nodes = set()
# all_nodes.update(node_coordinates.keys())
# # 对节点编号进行排序并建立映射关系
# node_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(all_nodes), start=0)}

# 提取所有唯一的节点编号:加0
# 计算所有节点经纬度的平均值
sum_lat = 0
sum_lon = 0
count = 0
for key, (lat, lon) in node_coordinates.items():
    if key != 0:  # 排除0号节点
        sum_lat += float(lat)
        sum_lon += float(lon)
        count += 1
# 计算平均值
avg_lat = sum_lat / count
avg_lon = sum_lon / count

node_coordinates[int(0)]=(avg_lat, avg_lon)
all_nodes_0 = set()
all_nodes_0.update(node_coordinates.keys())
# 对节点编号进行排序并建立映射关系
node_mapping_0 = {old_id: new_id for new_id, old_id in enumerate(sorted(all_nodes_0), start=0)}


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

# 定义存储图的字典
time_graphs = {}
df = data_2 
slide = timedelta(minutes=60)# 时间槽是预测长度一小时
# 使用for循环遍历时间序列
for current_time in time_list:
    # 初始化图
    G_0 = nx.Graph()
    G_0.add_nodes_from(node_coordinates.keys())
    # 为每个节点添加经纬度信息
    for node in G_0.nodes:
        lat, lon = node_coordinates[node]
        G_0.nodes[node]['latitude'] = float(lat)
        G_0.nodes[node]['longitude'] = float(lon)
    # 遍历时间段内的所有单车记录,筛选符合条件的行
    filtered_df = df[(df['stoptime'] > current_time) & (df['starttime'] < current_time+slide)]
    # print(filtered_df.index)
    # 按照'starttime'列升序排序
    filtered_df = filtered_df.sort_values('starttime', ascending=True)
    for index, row in filtered_df.iterrows():
        # print(index)这里的index其实对应着数据集df的index
        start_station = row["start station id"]
        end_station = row["end station id"]
        borrow_time = row["starttime"]
        return_time = row["stoptime"]
        bike_id = row["bikeid"]
        #在这个区间内的图分4种情况
        if  borrow_time < current_time and return_time < current_time+slide:
           add_or_update_edge(G_0, 0, end_station)
        elif borrow_time < current_time and return_time > current_time+slide:
            add_or_update_edge(G_0, 0, end_station)
        elif borrow_time >= current_time and return_time <= current_time+slide:
            add_or_update_edge(G_0, start_station, end_station)
        else:# borrow_time > current_time and return_time > current_time+slide:
            add_or_update_edge(G_0, start_station, 0)
    # 换图节点的编号
    G_0_r = nx.relabel_nodes(G_0, node_mapping_0)
    # 将当前时间槽的图存入字典
    time_graphs[current_time] = G_0_r


# 打印一个示例时间槽的图信息
example_time = time_list[0]
print(f"时间槽 {example_time} 的图:")
print(f"节点数: {time_graphs[example_time].number_of_nodes()}")
print(f"边数: {time_graphs[example_time].number_of_edges()}")

# 获取第一张图
first_graph = time_graphs[time_list[0]]

# 输出第一张图里面每个节点的名字
node_names = list(first_graph.nodes())
print("第一张图的节点名字列表:", node_names)

# 初始化一个列表来存储所有邻接矩阵
adj_matrixes = []

# 遍历time_graphs字典，将每个图的邻接矩阵添加到列表中
for time in time_list:
    graph = time_graphs[time]
    # 获取图的邻接矩阵
    adj_matrix = nx.to_numpy_array(graph)
    adj_matrixes.append(adj_matrix)

# 将列表保存到npz文件中
np.savez('adj_matrices.npz', A_t = adj_matrixes)
adjacency_data =np.load('adj_matrices.npz')
A_t = adjacency_data['A_t']  # shape: (T, N, N)
# 分割 A_t
train_A_t = A_t[:6048]
val_A_t = A_t[6048:7344]
test_A_t = A_t[7344:8640]


# 绘制图的函数
def plot_graph_with_coordinates(graph, title):
    plt.figure(figsize=(12, 8))
    
    # 提取节点的经纬度信息
    pos = {node: (data['longitude'], data['latitude']) 
           for node, data in graph.nodes(data=True)}
    
    # 绘制节点
    nx.draw_networkx_nodes(
        graph, pos, node_size=50, node_color="blue", alpha=0.7
    )
    
    # 绘制边，边的粗细根据权重调整
    edges = graph.edges(data=True)
    weights = [d['weight'] for _, _, d in edges]
    nx.draw_networkx_edges(
        graph, pos, edge_color="gray", width=[0.1 + w * 0.5 for w in weights]
    )
    
    # 添加节点标签
    nx.draw_networkx_labels(
        graph, pos, font_size=8, font_color="black", font_weight="bold"
    )
    
    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

# # 对每个时间槽绘制图
# for current_time, graph in time_graphs.items():
#     title = f"Graph for Time Slot: {current_time}"
#     plot_graph_with_coordinates(graph, title)
























