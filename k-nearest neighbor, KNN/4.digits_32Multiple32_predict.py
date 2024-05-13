import sklearn.neighbors as skn
import numpy as np
import os

"""
获取训练集的数据
"""
def trainDataSet(dir):
    # 获取目录下所有的文件名
    files = os.listdir(dir)
    label = []
    tdata = []
    # 有多少个文件
    for i in range(len(files)):
        fl = files[i]
        with open(dir + "/" + fl) as file:
            tdata.append([])
            for line in file:
                line = line.strip()
                arr = [line[e] for e in range(len(line))]
                tdata[i].extend(np.array(arr).astype(np.int8))
        label.append(fl.split("_")[0])
    return tdata, label


def testData(k):
    # 训练数据
    dir = "trainingDigits"
    tdata, label = trainDataSet(dir)
    ne = skn.KNeighborsClassifier(k)
    ne.fit(tdata, label)
    # 测试数据
    dir1 = "testDigits"
    testdata, testlabel = trainDataSet(dir1)
    okCount = 0
    errCount = 0
    okCount = sum(ne.predict(testdata) == testlabel)
    print("正确个数:", okCount, " 错误个数：", len(testdata) - okCount)
    print("正确率：", okCount / len(testdata))


testData(3)
