import numpy as np
from sklearn.model_selection import train_test_split

X = np.arange(12).reshape(6,2)
y = np.array([0,0,0,1,1,1])
times = 2
for i in range(2):
    train_set = train_test_split(X,y,random_state=24,shuffle=True)
    print()
    print(train_set)
    i+=1

# random_state的值可以理解为选择一种随机方式、对同一组数据的随机结果相同
# shuffle默认为True——超变量