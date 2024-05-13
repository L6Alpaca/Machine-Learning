from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler, scale
from math import log


def sigmoid(X, w1, w2, b):
    z = w1 * X[0] + w2 * X[1] + b
    return 1 / (1 + np.exp(-z))


# 定义损失函数
def loss_function(X, y, w1, w2, b):
    loss = 0
    for x_i, y_i in zip(X, y):  # X,y同时遍历
        p = sigmoid(x_i, w1, w2, b)
        p = np.clip(p,0.000001,0.999999)  #clip——进行裁剪，使p中值保持在[a_min,a_max]大于化为a_max 小于化为a_min
        # print(np.log(p))
        loss += -(y_i * np.log(p) + (1 - y_i) * np.log(1 - p))
    return loss


# 1.加载数据
X, y = datasets.load_breast_cancer(return_X_y=True)
# return_X_y 表示是否分开X,y返回，False时返回字典，True时返回两数组
X = X[:, :2]  # 切片前两个特征
# print(X.shape)

# 2.加载数据
model = LogisticRegression()
model.fit(X, y)
# coef_表示线性系数，是二维数组
# print(model.coef_)
w1 = model.coef_[0, 0]
w2 = model.coef_[0, 1]
b = model.intercept_  # 获得截距b
# loss = loss_function(X,y,w1,w2,b)
# print(loss)

# 取w1,w2取值空间
w1_space = np.linspace(w1 - 2, w1 + 2, 100)
w2_space = np.linspace(w2 - 2, w2 + 2, 100)
loss1 = np.array([loss_function(X, y, i, w2, b) for i in w1_space])
loss2 = np.array([loss_function(X,y,w1,i,w2) for i in w2_space])
#print(loss1)    #[inf]表示无穷 1.分母为0 2.np.log(0)

plt.figure(figsize=(12,9))
plt.subplot(2,2,1)  #设置子视图 多个
#前两个数 2,2表示图例长宽比，第三个数1表示第i个图例
plt.plot(w1_space,loss1,color = "green")
plt.title("W1")

plt.subplot(2,2,2)
plt.plot(w2_space,loss2,color = "red")
plt.title("W2")
plt.show()
#观察到loss函数为下凸函数——存在最优解
#末尾水平部分表示裁剪后达到的一个固定值（因为超出部分裁剪后值相同）
