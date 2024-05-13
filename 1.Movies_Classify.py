import numpy as np
import matplotlib.pyplot as mp
import collections as c

# 实现knn算法 一般用于推测 不具备学习能力 主要是比较
'''
数据集合，也就是训练样本集。这个数据集有两个特征，
即打斗镜头数和接吻镜头数。
除此之外，我们也知道每个电影的所属类型，即分类标签
电影名称  打斗镜头 接吻镜头 电影类型
神雕侠侣  100       20      动作片
毒液：致命守护者  99 10     动作片
碟中谍6：全面瓦解 67  5     动作片
热情如火  40       125     动作片
泰坦尼克号    0      10     爱情片
倩女幽魂    10       20     爱情片
大话西游之月光宝盒 10  40    爱情片
烈火如歌         1    30     爱情片
'''

arr = np.array([[100, 200], [99, 10], [67, 5], [40, 125], [0, 10], [10, 20], [10, 40], [1, 30]])
tarr = np.array([1, 1, 1, 1, 0, 0, 0, 1])
x = arr[:, :1].T[0]
y = arr[:, 1:].T[0]
print("x轴数据:", x)
print("y轴数据:", y)

# 设置字体
mp.rcParams['font.family'] = ['STFangsong']
mp.title("电影类型图")
mp.xlabel("打斗镜头")
mp.ylabel("接吻镜头")
# 第三个参数 o表示使用 散点  r表示red红色
mp.plot(x, y, "or")
mp.show()

# 判断打斗镜头44  接吻镜头 12到底是哪种类型的片片了
ndata = [44, 12]
# 计算当前这个ndata的坐标和之前所有数据的坐标的距离 放在一个jl数组中
# 距离计算公式是 欧氏距离  (x-x1)**2 +(y-y1)**2 开平方根
# jl中每个下标的数据 就是ndata和对应位置xy坐标的距离
jl = [np.sqrt((ndata[0] - i[0]) ** 2 + (ndata[1] - i[1]) ** 2) for i in arr]
print("未排序的数据是", jl)
# argsort——对距离进行排序  然后获取排序后的|下标|    升序，若求降序可将arr取相反数后argsort
#  比如数组：      [10,12,8]
#  argsort升序   [2,0,1]
jlsort = np.argsort(jl)
print("排序的索引是", jlsort)
k = 3
print(jlsort[:k])
# 获取指定k 前三个值最小下标的标签 也就是前三个距离最近的都是什么类型的电影
# 比如[1,1,0]
flaga = [tarr[t] for t in jlsort[:k]]
print(flaga)
# 统计类型集合的哪个出现的次数 会得到一个字典
# [(1,2),(0,1)]
group = c.Counter(flaga)
# 获取到个数排序（从大到小） 值最大的前1个
# [(1,2)]  [0][0]获取到1 类型就是动作片罗
print(group.most_common(1)[0][0])
# 来个三目判断下 输出中文
result = ("动作片" if group.most_common(1)[0][0] == 1 else "爱情片")
print(result)
