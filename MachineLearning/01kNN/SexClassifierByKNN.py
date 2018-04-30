# -*- coding:utf-8 -*-
# __author__:Luyu-Liam
import numpy as np
import operator


def file2matrix(filename):
    '''
    函数描述：解析文件。将带有标签的数据解析为特征矩阵和标签向量
    :param filename:
        filename --文件名
    :return:
        returnMat - 特征矩阵
        classLabelVector - 分类Label向量
    '''

    # 打开文件
    fr = open(filename)
    # 读取文件所有内容
    arrayAlines = fr.readlines()
    # 得到文件行数
    numberOfLines = len(arrayAlines)
    # 构造一个numberOfLines行2列的全部为0的矩阵, 用于充当特征矩阵
    returnMat = np.zeros((numberOfLines, 2))
    # 分类标签
    classLabelVector = []

    # 行的索引值
    index = 0
    for line in arrayAlines:
        # 删除行首行尾的空白字符
        line = line.strip()
        # 按照制表符'\t'进行切片
        listFromLine = line.split('\t')
        # 取出前两列放入到特征矩阵中
        returnMat[index, :] = listFromLine[0:2]
        # 女性通过1存储，男性通过2存储
        if listFromLine[-1] == "f" or listFromLine[-1] == 'F':
            classLabelVector.append(1)
        elif listFromLine[-1] == "m" or listFromLine[-1] == 'M':
            classLabelVector.append(2)
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    '''
    函数描述：对数据进行归一化

    :param dataSet:
        dataSet - 特征矩阵
    :return:
        normDataSet - 归一化后的特征矩阵
        ranges - 数据范围
        minVals - 数据最小值
    '''

    # 分别获取特征矩阵中每一列的最小、最大值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    # 特征矩阵中每一列的数据范围
    ranges = maxVals - minVals
    # 获取特征矩阵的行数
    m = dataSet.shape[0]
    # 原始值减去最小值
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    # 再除以最大值和最小值的差，得到归一化特征矩阵
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def classifyByKNN(dataTest, dataSet, lables, k):
    '''
    函数描述：kNN分类器

    :param dataTest: 用于分类的数据（测试集）
    :param dataSet: 用于训练的数据（训练集）
    :param lables: 分类标签
    :param k: kNN算法参数，选择距离最近的k个点
    :return: 分类结果
    '''
    # 获取训练数据集的行数
    dataSetSize = dataSet.shape[0]

    diffMat = np.tile(dataTest, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    # 上面四行是通过相减、平方、求和、开方来求距离
    sortedDistIndex = distances.argsort()
    # 排序，返回的是索引值

    # 取出排序后的前k个标签，然后按照次数最多的标签进行分类
    classCount = {}
    for i in range(k):
        voteLables = lables[sortedDistIndex[i]]
        classCount[voteLables] = classCount.get(voteLables, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def classifyTest():
    # 输出结果列表
    resultList = ['女', '男']
    # 测试集
    testFileName = "dataSetTest.txt"
    testDataMat, testLableVector = file2matrix(testFileName)
    testLines = testDataMat.shape[0]
    #训练集
    exFileName = "dataSetEx.txt"
    dataMat, dataLable = file2matrix(exFileName)
    normMat, ranges, minVals = autoNorm(dataMat)
    # 错误的个数
    errorCount = 0
    for i in range(testLines):
        inArr = testDataMat[i, :]
        normInArr = (inArr - minVals) / ranges
        classifierResult = classifyByKNN(normInArr, normMat, dataLable, 5)
        print(resultList[classifierResult - 1])
        if classifierResult != testLableVector[i]:
            errorCount += 1
    # 计算错误率
    errorRate = errorCount / testLines
    print("错误率为：%f" %errorRate)


if __name__ == "__main__":
    classifyTest()
