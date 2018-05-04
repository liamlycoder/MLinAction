#!/usr/bin/python
# coding: utf-8
import csv
import time
import pandas as pd
from numpy import ravel
from sklearn.neighbors import KNeighborsClassifier


# 加载数据
def opencsv():
    # 使用 pandas 打开
    data = pd.read_csv(
        'dataSet/train.csv')
    data1 = pd.read_csv(
        'dataSet/test.csv')

    train_data = data.values[0:, 1:]  # 读入全部训练数据,  [行，列]
    train_label = data.values[0:, 0]  # 读取列表的第一列
    test_data = data1.values[0:, 0:]  # 测试全部测试个数据
    return train_data, train_label, test_data


def saveResult(result, csvName):
    with open(csvName, 'w') as myFile:  # 创建记录输出结果的文件（w 和 wb 使用的时候有问题）
        myWriter = csv.writer(myFile)  # 对文件执行写入
        myWriter.writerow(["ImageId", "Lable"])  # 设置表格的列名
        index = 0
        for i in result:
            tmp = []
            index = index + 1
            tmp.append(index)
            tmp.append(int(i))  # 测试集的标签值
            myWriter.writerow(tmp)


def knnClassify(trainData, trainLabel):
    knnClf = KNeighborsClassifier()   # 调用sklearn的knn算法，默认的k为5
    knnClf.fit(trainData, ravel(trainLabel))
    return knnClf


def dRecognition_knn():
    start_time = time.time()

    # 加载数据
    trainData, trainLabel, testData = opencsv()
    print("load data finish")
    stop_time_l = time.time()
    print('load data time used:%f' % (stop_time_l - start_time))

    # 模型训练
    knnClf = knnClassify(trainData, trainLabel)

    # 结果预测
    testLabel = knnClf.predict(testData)

    # 结果的输出
    saveResult(
        testLabel,
        'dataSet/Result_sklearn_knn.csv'
    )
    print("finish!")
    stop_time_r = time.time()
    print('classify time used:%f' % (stop_time_r - start_time))


if __name__ == '__main__':
    dRecognition_knn()
