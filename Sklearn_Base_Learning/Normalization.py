from sklearn.preprocessing import Normalizer,normalize
# Nomalizer 属于评估器方法
# normalize 属于函数方法
import numpy as np
# 1-范数——数据集里每个数/所有数的绝对值之和
# 2-范数——数据集里每个数/（所有数平方之和）的开方
# norm = "l1" or "l2" 表示范数选择
X = np.arange(15).reshape(5,3)
X_1 = Normalizer(norm="l1",copy=True).fit_transform(X)
print(X)
print(X_1)
'''
等同于
norm = Normalizer(norm='l1',copy=True)
norm.fit(X)
X_2 = norm.transform(X)
print(X_2)

或等同于 X_1 = normalize(X,norm='l1',axis=1,copy=True)

对numpy细节为
X = X / np.linalg.norm(X,ord=1,axis=1).reshape(5,1)
print(X)
'''


