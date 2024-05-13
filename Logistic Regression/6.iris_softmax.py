import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

#1.数据准备
X, y = datasets.load_iris(return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=21)

#2.建模
model = LogisticRegression(multi_class="multinomial")
#"multinomial"使用"softmax"回归、归一化
#print(X_train)
#3.训练模型
model.fit(X_train,y_train)
#4.获得训练结果
score = model.score(X_test,y_test)  #score()对X_test进行predict后，与y_test对比计算accuracy
#print(score)

p = model.predict_proba(X_test)
print(p[:5])



'''
手写↓
def softmax(z):
    return np.exp(z)/np.exp(z).sum()
w = model.coef_
b = model.intercept_
#print(w)
z = X_test.dot(w.T) + b
proba = softmax((z))
proba = proba / proba.sum(axis=1).reshape(-1,1)
print(proba[:5])
'''