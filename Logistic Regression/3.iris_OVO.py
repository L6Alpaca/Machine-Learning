import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



X, y = datasets.load_iris(return_X_y=True)
cond = y!=2
X = X[cond]
y = y[cond]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=21)

model = LogisticRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print(y_pred)
print(y_test)
proba = model.predict_proba(X_test)  #probability可能性（概率）
#y_pred 实际是 proba.argmax(axis=1)即每一项中更大的值的位置（0 || 1）
#print(proba.argmax(axis=1))
print(proba)

#proba_实现过程
'''
def sigmoid(z):
    return 1/(1+np.exp(-z))
w = model.coef_[0]
b = model.intercept_
z = X_test.dot(w)+b
p = sigmoid(z)
print(p)
'''