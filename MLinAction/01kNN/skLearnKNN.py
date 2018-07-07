# !/usr/bin/python
# coding: utf8
# @Time    : 2018-06-04 21:11
# @Author  : Liam
# @Email   : luyu.real@qq.com
# @Software: PyCharm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_curve, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import ravel


# 加载数据
def file2matrix(filename):
    fr = open(filename)
    arrayAlines = fr.readlines()
    numberOfLines = len(arrayAlines)
    returnMat = np.zeros((numberOfLines, 2))
    classLabelVector = []
    index = 0
    for line in arrayAlines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:2]
        if listFromLine[-1] == "f" or listFromLine[-1] == 'F':
            classLabelVector.append(0)
        elif listFromLine[-1] == "m" or listFromLine[-1] == 'M':
            classLabelVector.append(1)
        index += 1
    return returnMat, classLabelVector


# 训练模型
def knnClassify(trainData, trainLabel):
    knnClf = KNeighborsClassifier()  # 调用sklearn的knn算法，默认的k为5
    knnClf.fit(trainData, ravel(trainLabel))
    return knnClf


# 模型评估
def showPrecisionRecall(x_test, y_test, clf):
    y_pre = clf.predict(x_test)
    precision, recall, thresholds = precision_recall_curve(y_test, y_pre)
    target_names = ['female', 'male']
    print(classification_report(y_test, y_pre, target_names=target_names))
    return precision, recall, thresholds


if __name__ == "__main__":
    # 解析数据
    testFileName = "Data.txt"
    testDataMat, testLableVector = file2matrix(testFileName)
    # 拆分训练数据与测试数据，这里是按照8:2的比例拆分
    x_train, x_test, y_train, y_test = train_test_split(testDataMat, testLableVector, test_size=0.2)

    # 训练模型
    clf = knnClassify(x_train, y_train)

    # 模型评估
    showPrecisionRecall(x_test, y_test, clf)
