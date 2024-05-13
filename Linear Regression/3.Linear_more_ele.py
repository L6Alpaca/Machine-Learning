import numpy as np
import matplotlib.pyplot as plot
import sklearn.linear_model as lm
import sklearn.datasets as ds
import sklearn.model_selection as ms
import pandas as pd


#数据处理
Bos_path = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(Bos_path, sep="\s+", skiprows=22, header=None)
# 获取波士顿房价的所有特征数据↓
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
# 获取每行特征对应的房价↓
label = raw_df.values[1::2, 2]


# 将数据拆分成80%的训练数据  20%的测试数据
xtrain, xtest, ytrain, ytest = ms.train_test_split(data, label, test_size=0.2, random_state=10,shuffle = True)
# ra
xtrain = xtrain[ytrain < 50]
ytrain = ytrain[ytrain < 50]
lr = lm.LinearRegression()
lr.fit(xtrain, ytrain)
np.set_printoptions(suppress=True)  # 不使用科学计数法

# 每个自变量系数↓
print("所有的系数:")
print(lr.coef_)

# 截距↓
print(lr.intercept_)
