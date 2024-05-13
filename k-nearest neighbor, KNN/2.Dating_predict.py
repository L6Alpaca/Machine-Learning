import numpy as np
import matplotlib.pyplot as pl
import matplotlib.lines as mlines
import collections as coll

"""
  读取data.txt的所有数据集
  前三列：
    海伦收集的样本数据主要包含以下3种特征：
        每年获得的飞行常客里程数
        玩视频游戏所消耗时间百分比
        每周消费的冰淇淋公升数
  最后一列：
    didntLike 不喜欢的人
    smallDoses 魅力一般的人
    largeDoses 极具魅力的人

"""


def dataSet():
    arr = []
    with open("lovedata.txt", "r") as file:
        for line in file:
            arr.append(line.strip().split("\t"))  # 默认删除空白符(包括'\n','\r','\t',' ')
    arrnp = np.array(arr)
    # 将特征数据字符串转换成数字
    return arrnp[:, :3].astype(dtype=np.float32), arrnp[:, 3:].T[0]


'''
根据数据
绘制图形
'''


def graphDataSet(feature, result):
    pl.rcParams['font.family'] = ['STFangsong']
    # nrow=2,nclos=2时,代表fig画布被分为四个区域,axs[0][0]表示第一行第一个区域 figsize表示画布大小
    fig, axs = pl.subplots(nrows=2, ncols=2)  # """,figsize=(13,8)"""
    colorArray = ["black" if e == "didntLike" else ("orange" if e == "smallDoses" else "red") for e in result]
    drawSubPlot(axs, 0, 0, "每年获得的飞行常客里程数和玩视频游戏所消耗时间百分比占比"
                , "每年获得的飞行常客里程数",
                "玩视频游戏所消耗时间",
                feature[:, :1].T[0],
                feature[:, 1:2].T[0],
                colorArray
                )
    #####绘制 0,1这个subplot 上面代码用于学习
    drawSubPlot(axs, 0, 1, "玩视频游戏所消耗时间和每周消费的冰淇淋公升数占比"
                , "玩视频游戏所消耗时间",
                "每周消费的冰淇淋公升数",
                feature[:, 1:2].T[0],
                feature[:, 2:3].T[0],
                colorArray
                )
    drawSubPlot(axs, 1, 0, "每年获得的飞行常客里程数和每周消费的冰淇淋公升数占比"
                , "每年获得的飞行常客里程数",
                "每周消费的冰淇淋公升数",
                feature[:, 0:1].T[0],
                feature[:, 2:3].T[0],
                colorArray
                )
    pl.show()


"""
  绘制子plot的封装
"""


def drawSubPlot(axs, x, y, title, xlabel, ylabel, xdata, ydata, colorArray):
    axs[x][y].set_title(title)
    axs[x][y].set_xlabel(xlabel)
    axs[x][y].set_ylabel(ylabel)
    axs[x][y].scatter(x=xdata, y=ydata, color=colorArray, s=2)
    didntLike = mlines.Line2D([], [], color='black', marker='.',
                              markersize=2, label='不喜欢')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                               markersize=2, label='魅力一般')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',
                               markersize=2, label='极具魅力')
    axs[x][y].legend(handles=[didntLike, smallDoses, largeDoses])


'''
  归一化数据(MinMax_Standardization)
'''


def normalizing(feature):
    # graphDataSet(feature, result)
    # 对所有的数据进行归一化
    # 假设 feature=np.array([[1,2],[3,4],[1.3,2.3]])
    # 每一列上的最小值  [1,2]
    minVal = np.min(feature, axis=0)
    # 每一列上的最大值  [3,4]
    maxVal = np.max(feature, axis=0)
    # 当前数据集 -最小值 [[1,2],[3,4],[1.3,2.3]]-[1,2]是不行的 应该行和列一样
    # 第一列应该-1  第二列应该减去2
    # 模拟成数据  [[1,2],[3,4],[1.3,2.3]]-[[1,2],[1,2],[1,2]] 这样才行
    # 有几列 就有几个 [1,2]的最小值数组
    minArr = np.tile(minVal, (feature.shape[0], 1))
    maxArr = np.tile(maxVal, (feature.shape[0], 1))
    resultArr = (feature - minArr) / (maxArr - minArr)
    return resultArr


"""
  该函数用于返回预测当前data的label值 也就是knn算法
   data 用于预测结果的数据 比如 [1000,1.1,0.8]
   trainData 是训练集  [[40920	8.326976	0.953952],[14488	7.153469	1.673904]]
   k表示预测数据最近的k个数据
   labelData 表示训练集的对应的label数据
"""


def knn(data, trainData, labelData, k):
    testData = np.tile(data, (trainData.shape[0], 1))
    # print(testData)
    # 计算距离差的平方开根
    sqdata = np.sqrt(np.sum((testData - trainData) ** 2, axis=1))
    # 选取与当前点距离最小的k个点的下标；
    kindex = np.argsort(sqdata)[:k]
    # 取出所有的该距离位置最近的结果
    resultdata = [labelData[ki] for ki in kindex]
    # print(sqdata)
    # print(kindex)
    # print(resultdata)
    # 分组获取最大的那一个
    return (coll.Counter(resultdata).most_common(1)[0][0])


# knn(np.array([1000,1.1,0.8]),np.array([[40920,8.326976,0.953952],[14488,7.153469,1.673904],[35483,12.273169,1.508053]]),["不喜欢","喜欢","喜欢"],2)
"""
    将所有的数据按照ratio比例拆分
    90%数据用于训练  10%数据用于测试knn算法准确率
    ratio 表示拆分的比例  0.1表示训练集=1-0,1 测试集是0.1
    k表示knn的k
"""


def testData(ratio, k):
    feature, resultLabel = dataSet()
    feature = normalizing(feature)
    # 拿到10%的数据用户测试knn算法
    # 获取总行数
    rows = feature.shape[0]
    # 获取%90的实际个数 必须将float转换成int类型
    ratioCount = int(rows * (1 - ratio))
    trainData = feature[:ratioCount, ]
    testData = feature[ratioCount:, ]
    resultI = ratioCount
    # 统计正确率
    okCount = 0
    erroCount = 0
    for td in testData:
        realResult = resultLabel[resultI]
        calculateResult = knn(td, trainData, resultLabel, k)
        if realResult == calculateResult:
            okCount = okCount + 1
        else:
            erroCount = erroCount + 1
        print("真实结果:", realResult, "  预测结果:", calculateResult)
        resultI = resultI + 1
    print("正确率是:", (okCount / (okCount + erroCount)))


ratio = 0.1
k = 5
testData(ratio, k)


def classifyPerson():
    precentTats = float(input("每年获得的飞行常客里程数:"))
    ffMiles = float(input("玩视频游戏所耗时间百分比:"))
    iceCream = float(input("每周消费的冰激淋公升数:"))
    feature, resultLabel = dataSet()
    k = 10
    calculateResult = knn([precentTats, ffMiles, iceCream], feature, resultLabel, k)
    print("您可能 ", calculateResult, "这个人")


# classifyPerson()
feature, result = dataSet()
print(feature)
print(result)
graphDataSet(feature, result)
