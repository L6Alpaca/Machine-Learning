import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans  # 导入K均值聚类算法

# make_blobs：生成聚类的数据集
# n_samples：生成的样本点个数，n_features：样本特征数，centers：样本中心数
# cluster_std：聚类标准差，shuffle：是否打乱数据，random_state：随机种子
X, y = make_blobs(n_samples=150, n_features=2,centers=3, cluster_std=0.5,shuffle=True, random_state=0)

# 散点图
# c：点的颜色，marker：点的形状，edgecolor：点边缘的形状，s：点的大小
plt.scatter(X[:, 0], X[:, 1],c='white', marker='o',edgecolor='black', s=50)
plt.show()

# 定义模型
# n_clusters：要形成的簇数，即k均值的k，init：初始化方式，tot：Frobenius 范数收敛的阈值
model = KMeans(n_clusters=3, init='random',n_init=10, max_iter=300, tol=1e-04, random_state=0)
'''
1.init 参数提供了三种产生筷中心的方法：“K-means++”指定产生较大间距的筷中心(2.1.4节)；“random”指定随机产生簇中心；由用户通过一个ndarrav 数组指定初始筷中心。
2.n_init 参数指定了算法运行次数，它在不指定初始筷中心时，通过多次运行算法，最终选择最好的结果作为输出。
3.max iter 参数指定了一次运行中的最大迭代次数。在大规模数据集中，算法往往要耗费大量的时间，可通过指定迭代次数来折中耗时和效果。
4.tol 参数指定了算法收敛的國值。在大规模数据集中，算法往往难以完全收敛，即达到连续两次相同的分筷需要耗费很长时间,可通过指定國值来折中耗时和最优目标。
5.algorithm 参数指定了是否采用elkan k-means 算法来简化距离计算。该算法比经典的k-means 算法在迭代速度方面有很大的提高。但该算法不适用于稀疏的样本数据。值“full”指定采用经典k-means 算法。值“ellkan”指定采用 elkan k-means 算法。值“auto”自动选择，在稠密数据时采用 elkan k-means 算法，在稀疏数据时采用经典k-means 算法。
'''
# 训练加预测
y_pred = model.fit_predict(X)

# 画出预测的三个簇类
plt.scatter(
    X[y_pred == 0, 0], X[y_pred == 0, 1],
    s=50, c='lightgreen',
    marker='s', edgecolor='black',
    label='cluster 1'
)

plt.scatter(
    X[y_pred == 1, 0], X[y_pred == 1, 1],
    s=50, c='orange',
    marker='o', edgecolor='black',
    label='cluster 2'
)

plt.scatter(
    X[y_pred == 2, 0], X[y_pred == 2, 1],
    s=50, c='lightblue',
    marker='v', edgecolor='black',
    label='cluster 3'
)

# 画出聚类中心
plt.scatter(
    model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='centroids'
)
plt.legend(scatterpoints=1)
plt.grid()
plt.show()

# 计算inertia随着k变化的情况
distortions = []
for i in range(1, 10):
    model = KMeans(
        n_clusters=i, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    model.fit(X)
    distortions.append(model.inertia_)
# 画图可以看出k越大inertia越小，追求k越大对应用无益处
plt.plot(range(1, 10), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()
