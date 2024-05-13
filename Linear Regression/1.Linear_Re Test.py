# 从实现的角度来看，这只是普通的普通 最小二乘法 （scipy.linalg.lstsq） 或非负最小二乘法 （scipy.optimize.nnls） 包装为预测器对象。

import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = 1 * x_0 + 2 * x_1 + 3  ——(dot(x,y)x,y可以是数，可以是n维数组，数组可视为矩阵相乘)
y = np.dot(X, np.array([1, 2])) + 3
reg = LinearRegression().fit(X, y)
k = reg.score(X, y)
# score——得到回归曲线的斜率
print(k)
coefficent = reg.coef_
print(coefficent)
intercept = reg.intercept_
print(intercept)
# predict(x)给x得到预测值
prediction = reg.predict(np.array([[3, 5]]))
print(prediction)
