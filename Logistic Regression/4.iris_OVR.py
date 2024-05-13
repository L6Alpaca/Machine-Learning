import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



X, y = datasets.load_iris(return_X_y=True)  #含有 3种y——不同类型的iris 4个属性x
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#对multi_class这一项，Default = "auto"——即会在二分类时选择"ovr",多分类时选择"multinomial"使用"softmax"
model = LogisticRegression(multi_class="ovr")   #One vs Rest 可用于多分类
#ovr 同时还对每次预测的概率进行了归一化，使多项p之和为一
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
#print(y_pred)
#print(y_test)
proba = model.predict_proba(X_test)  #probability可能性（概率）
#y_pred 实际是 proba.argmax(axis=1) 即每一项中 最大的值 的位置（0 || 1 || 2）
#print(proba.argmax(axis=1))
print(proba[:5])    #取proba[:]中最大值位置为y_pred

accuracy = (y_test == y_pred).mean()
#print(accuracy)

'''
手写↓
def sigmoid(z):
    return 1/(1+np.exp(-z))
w = model.coef_     #三行——三个分类器，有三类结果  四列——四个属性在不同情况下的系数
b = model.intercept_
z = X_test.dot(w.T) + b
p = sigmoid(z)
p = p / p.sum(axis=1).reshape(-1,1)     #归一化处理
print(p[:5])
'''

#理解为一种特殊的二分类，结果有n类，每次取一种作为正类，其它作为负类，得到n个分类器，最后再整合
#视为两类 One 和 Rest

#优点：普适性、效率高、几种分类对应几个分类器
#缺点：训练集样本不平衡——数据多时易出现正类远不及负类，导致有偏向性