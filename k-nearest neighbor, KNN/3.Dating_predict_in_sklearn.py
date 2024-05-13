import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sklearn.preprocessing as pre


def dataSet():
    arr = []
    with open("lovedata.txt", "r") as file:
        for line in file:
            arr.append(line.strip().split("\t"))  # 默认删除空白符(包括'\n','\r','\t',' ')
    arrnp = np.array(arr)
    # 将特征数据字符串转换成数字
    return arrnp[:, :3].astype(dtype=np.float32), arrnp[:, 3:].T[0]


# 调用sklearn测试k近邻


def testData(ratio, k):
    feature, result = dataSet()

    # 使用sklearn的预处理进行归一化 使用均值方差归一化（此处z_socre）
    feature = pre.StandardScaler().fit(feature).transform(feature)
    print(result)
    # train_test_split将数据拆分了 test_size的比例 传入0.1就是10%的测试集
    # random_state 随即种子 可能随即抽取10%的测试集 如果random_state是某个固定的数 下次传入 获取的是相同的测试集
    # 如果是0或者不填 每次获取的测试集都不是相同的数据
    train_X, test_X, train_y, test_y = train_test_split(feature, result, test_size=ratio, random_state=0)
    # 创建一个k临近 传入距离最近的k个值
    nei = KNeighborsClassifier(k)
    # 填充 训练数据 和 训练集结果
    nei.fit(train_X, train_y)
    # 预测所有的测试集 得到预测的结果
    predict_y = nei.predict(test_X)
    # 比较预测结果和实际结果 得到得分
    score = accuracy_score(test_y, predict_y)
    print(score)


ratio = 0.1
# test_data / all = ratio(0.1)
k = 5
testData(ratio, k)
