import numpy as np
import matplotlib.pyplot as plot
import sklearn.linear_model as lm
import sklearn.datasets as ds
import sklearn.model_selection as ms
import pandas as pd


# Part1——加载数据集 #
Bos_path = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(Bos_path, sep="\s+", skiprows=22, header=None)
# 获取波士顿房价的所有特征数据↓
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
# 获取每行特征对应的房价↓
label = raw_df.values[1::2, 2]
nox = data[:, 5:6]
# 将数据拆分成80%的训练数据  20%的测试数据
xtrain, xtest, ytrain, ytest = ms.train_test_split(nox, label, test_size=0.2, random_state=42)
# print(ytest)
# 将 [[4],[2]]这样的特征矩阵转换成 [4,2]这样的向量 绘制散点图
plot.scatter(xtrain[:, -1], ytrain, c="green")
# plot.show()


# Part2——使用线程回归计算系数和截距 #
# 创建线程回归的类
lr = lm.LinearRegression().fit(xtrain, ytrain)
k = lr.score(xtrain,ytrain)
print(k)
# 系数也就是斜率
print(lr.coef_)
# 截距
# print(lr.intercept_)
xtrain = xtrain[ytrain < 50]
ytrain = ytrain[ytrain < 50]
plot.scatter(xtrain[:, -1], ytrain, c="red")
# 绘制 80%的真实数据
plot.plot(xtrain[:, -1], xtrain[:, -1] * lr.coef_ + lr.intercept_)
plot.show()