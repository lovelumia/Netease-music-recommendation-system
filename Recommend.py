import pandas as pd

# 读取歌曲数据
data_music = pd.read_csv("dfmusic.csv")
data_music.head()

# 读取歌单数据
display = pd.read_csv("dfplaylist.csv")
display.head()

#输入歌单id 输出对应播放列表musiclist
listid = 111450065
print("这个歌单的播放列表为：")
print(display[display.id==listid]["musiclist"].tolist())

#看一下数据的长度
len(data_music),len(display)

# 整理成训练集的样式，歌单，歌单中的歌曲，随机打分（1-6分），时间
import random
path = open("train.csv","w")
for i,j in zip(display[:20000].id,display[:20000].musiclist):
    for m,k in enumerate(eval(j)):
        path.write(str(i)+"\t"+k+"\t"+str(random.randint(1,6))+"\t"+str(m)+"\n")
path.close()

# neteasy_song_id_to_name_data.csv
# 保存  歌曲id=>歌曲名 
path = open("neteasy_song_id_to_name_data.csv","w",encoding="utf-8")
for i,j in zip(data_music.id,data_music.name):
    path.write(str(i)+","+str(j)+"\n")
path.close()

# neteasy_playlist_id_to_name_data.csv
#保存   歌单id=>歌单名 
path = open("neteasy_playlist_id_to_name_data.csv","w",encoding="utf-8")
for i,j in zip(display.id,display.name):
    path.write(str(i)+","+str(j)+"\n")
path.close()

# -*- coding:utf-8-*-
"""
利用surprise推荐库 KNN协同过滤算法推荐网易云歌单
python3.0环境
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)
import os
import csv
from surprise import KNNBaseline, Reader, KNNBasic, KNNWithMeans
from surprise import Dataset
 
 
def recommend_model():
    # 训练集
    file_path = os.path.expanduser('train.csv')
    # 指定文件格式 数据四列  歌单和对应的歌曲 
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    # 从文件读取数据
    music_data = Dataset.load_from_file(file_path, reader=reader)
    # 计算歌曲和歌曲之间的相似度
 
    train_set = music_data.build_full_trainset()
    print('开始使用协同过滤算法训练推荐模型...')
    algo = KNNBasic() # 初始化模型
    # 开始训练
    algo.fit(train_set)
    return algo
 
 
def playlist_data_preprocessing():
    # 读取 歌单id-》歌单名称 ，构造两个映射字典，可以通过 歌单id找到歌单名称，也可以 通过歌单名称找到歌单id
    csv_reader = csv.reader(open('neteasy_playlist_id_to_name_data.csv',encoding="utf-8"))
    id_name_dic = {}
    name_id_dic = {}
    for row in csv_reader:
        id_name_dic[row[0]] = row[1]
        name_id_dic[row[1]] = row[0]
    return id_name_dic, name_id_dic
 
 
def song_data_preprocessing():
    # 读取 歌曲id-》歌曲名称 ，构造两个映射字典，可以通过 歌曲id找到歌曲名称，也可以 通过歌曲名称找到歌曲id
    csv_reader = csv.reader(open('neteasy_song_id_to_name_data.csv',encoding="utf-8"))
    id_name_dic = {}
    name_id_dic = {}
    for row in csv_reader:
        id_name_dic[row[0]] = row[1]
        name_id_dic[row[1]] = row[0]
    return id_name_dic, name_id_dic
 
 
def playlist_recommend_main():
    print("加载歌单id到歌单名的字典映射...")
    print("加载歌单名到歌单id的字典映射...")
    id_name_dic, name_id_dic = playlist_data_preprocessing()
    print("字典映射成功...")
    print('构建数据集...')
    algo = recommend_model()
    print('模型训练结束...')
    # 随便找一首进行推荐  10 你也可以修改
    current_playlist_id = list(id_name_dic.keys())[541]
    print('当前的歌单id：' + current_playlist_id)
 
    current_playlist_name = id_name_dic[current_playlist_id]
    print('当前的歌单名字：' + current_playlist_name)
 
    playlist_inner_id = algo.trainset.to_inner_uid(current_playlist_id)
    print('当前的歌单内部id：' + str(playlist_inner_id))
 
    playlist_neighbors = algo.get_neighbors(playlist_inner_id, k=10)
    playlist_neighbors_id = (algo.trainset.to_raw_uid(inner_id) for inner_id in playlist_neighbors)
    # 把歌曲id转成歌曲名字
    playlist_neighbors_name = (id_name_dic[playlist_id] for playlist_id in playlist_neighbors_id)
    print("和歌单<", current_playlist_name, '> 最接近的10个歌单为：\n')
    for playlist_name in playlist_neighbors_name:
        print(playlist_name, name_id_dic[playlist_name])
 
 
playlist_recommend_main()

from __future__ import (absolute_import, division, print_function, unicode_literals)
from surprise.model_selection import cross_validate

import os
import csv
from surprise import KNNBaseline, Reader, KNNBasic, KNNWithMeans
from surprise import Dataset

file_path = os.path.expanduser('train.csv')
# 指定文件格式
reader = Reader(line_format='user item rating timestamp', sep='\t')
# 从文件读取数据
music_data = Dataset.load_from_file(file_path, reader=reader)
# 分成5折
# music_data.split(n_folds=5)
 
algo = KNNBasic()
# 五折交叉验证 打印 'RMSE', 'MAE' 这两个指标
perf = cross_validate(algo, music_data, measures=['RMSE', 'MAE'], cv=5,verbose=1)
print(perf)


