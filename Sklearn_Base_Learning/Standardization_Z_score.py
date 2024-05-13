'''
用于衡量数据点相对于数据集平均值的偏离程度的指标，可以帮助我们了解数据点在分布中的相对位置。
可以用一条正态分布曲线来表示，z-score 就是数据点在这条曲线上的位置。数据点离平均值越远，z-score 的绝对值就越大

数据的标准化（normalization）是将数据按比例缩放，使之落入一个小的特定区间。其目的是去除数据的单位限制，
    将其转化为无量纲的纯数值，便于不同单位或量级的指标能够进行比较和加权。
其中最典型的就是数据的归一化处理，即将数据统一映射到[0,1]区间上。
由于x1、x2的数量级相差巨大，训练效率低且训练过程曲折，训练效果差。

因此，标准化的好处有：
1、提升模型的收敛速度
2、提升模型的精度
3、深度学习中数据归一化可以防止模型梯度爆炸。

普通的z_score写法
import numpy as np

def calculate_z_score(data, data_point):
    mean = np.mean(data) //平均数
    std = np.std(data) //标准差（Standard Deviation）
    z_score = (data_point - mean) / std
    return z_score

# 示例用法
data = [12, 15, 18, 21, 24]
data_point = 18

z_score = calculate_z_score(data, data_point)
print("数据点的 z-score 为:", z_score)


// math method
import math
def get_average(data):  #求数组平均数
    return sum(data) / len(data)
def get_variance(data):#求数组方差
    average = get_average(data)
    return sum([(x - average) ** 2 for x in data]) / len(data)
def get_standard_deviation(data): #求数组标准差
    variance = get_variance(data)
    return math.sqrt(variance)
def get_z_score(data): #求数组的z-score归一化最后的结果
    avg = get_average(data)
    stan = get_standard_deviation(data)
    scores = [(i-avg)/stan for i in data]
    return scores
'''

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

data = np.arange(15).reshape(5, 3)
train_data, test_data = train_test_split(data, random_state=42)
scaler = StandardScaler()
scaler.fit(train_data)  # fit——获得标准化所需参数的过程
scale = scaler.scale_  # 查看标准差
# print(scale)
mean = scaler.mean_  # 查看均值
# print(mean)
var = scaler.var_  # 查看方差
sample_num = scaler.n_samples_seen_  # 有效的训练数据条数（系数矩阵中需要删去一些数据）

Nor_train_data = scaler.transform(train_data)
# 以上可归并为一句Nor_train_data = scaler.fit_transform(train_data)
print(Nor_train_data)
'''
快捷——fit_transform(data)直接获得标准化后的数据并且 保留 标准化所需参数特征
后再对test_data 做transform即可
print(scaler.fit_transform(train_data))
print(scaler.transform(test_data))
'''
# 标准化过程不会对参数data进行修改 （非原地操作）
