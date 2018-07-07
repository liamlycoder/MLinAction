# !/usr/bin/python
# coding: utf8
# @Time    : 2018-07-07 23:06
# @Author  : Liam
# @Email   : luyu.real@qq.com
# @Software: PyCharm
import numpy as np
#                        .::::.
#                      .::::::::.
#                     :::::::::::
#                  ..:::::::::::'
#               '::::::::::::'
#                 .::::::::::
#            '::::::::::::::..
#                 ..::::::::::::.
#               ``::::::::::::::::
#                ::::``:::::::::'        .:::.
#               ::::'   ':::::'       .::::::::.
#             .::::'      ::::     .:::::::'::::.
#            .:::'       :::::  .:::::::::' ':::::.
#           .::'        :::::.:::::::::'      ':::::.
#          .::'         ::::::::::::::'         ``::::.
#      ...:::           ::::::::::::'              ``::.
#     ```` ':.          ':::::::::'                  ::::..
#                        '.:::::'                    ':'````..
#                     美女保佑 永无BUG
def kmeans(X, k, maxIt):
    """
    kmeans算法
    :param X: 特征矩阵。这里的矩阵是没有标签的。
    :param k: kmeans算法中的参数k
    :param maxIt: 最大迭代次数
    :return: 返回值是加好标签的特征矩阵，标签为矩阵的最后一列
    """
    # 获取行和列
    numPoints, numDim = X.shape

    # 构造出最后一列为标签列，初始化其值为0
    dataSet = np.zeros((numPoints, numDim + 1))
    dataSet[:, :-1] = X

    # 随机选择k个中心点。下面randint函数的意思是：从numPoints个值中随机选择k个
    centroids = dataSet[np.random.randint(numPoints, size=k), :]
    centroids[:, -1] = range(1, k+1)

    iterations = 0
    oldCentriods  = None

    while not shouldStop(oldCentriods, centroids, iterations, maxIt):
        print("iterations:", iterations)
        print("dataSet:", dataSet)
        print("centroids:", centroids)

        # 更新原来的中心点
        oldCentriods = np.copy(centroids)
        iterations += 1

        # 更新标签值
        updateLables(dataSet, centroids)

        # 更新中心点的值
        centroids = getCentriods(dataSet, k)

    return dataSet

def shouldStop(oldCentriods, centroid, iterations, maxIt):
    """
    根据中心点是否变化或者是否达到最大迭代次数来判断是否应该停止迭代
    :param oldCentriods: 旧的中心点
    :param centroid: 中心点
    :param iterations: 当前迭代次数
    :param maxIt: 最大迭代次数
    :return: 布尔值。True表示应该停止。
    """
    if iterations > maxIt:
        return True
    return np.array_equal(oldCentriods, centroid)

def updateLables(dataSet, centroids):
    """
    根据每次迭代后的特征矩阵，以及中心点，来更新标签值
    :param dataSet: 特征矩阵
    :param centroids: 中心点
    """
    numPoints, numDim = dataSet.shape
    for i in range(0, numPoints):
        dataSet[i, -1] = getLableFromClosestCentroid(dataSet[i, :-1], centroids)

def getCentriods(dataSet, k):
    """
    根据特征矩阵和k值，获取新的中心点
    :param dataSet: 特征矩阵
    :param k: 参数k值
    :return: 新的中心点矩阵
    """
    result = np.zeros((k, dataSet.shape[1]))
    # 遍历每个类别，计算每个类别的中心点
    for i in range(1, k+1):
        oneCluster = dataSet[dataSet[:, -1] == i, :-1]
        # mean函数可以用来求均值
        result[i - 1, :-1] = np.mean(oneCluster, axis=0)
        result[i - 1, -1] = i
    return result

def getLableFromClosestCentroid(dataSetRow, centroids):
    """
    计算离中心点最近的标签
    :param dataSetRow: 特征矩阵除去标签值部分
    :param centroids: 中心点
    :return: 距离最近的中心点的标签
    """
    lable = centroids[0, -1]
    minDist = np.linalg.norm(dataSetRow - centroids[0, -1])
    for i in range(1, centroids.shape[0]):
        dist = np.linalg.norm(dataSetRow - centroids[i, -1])
        if dist < minDist:
            minDist = dist
            lable = centroids[i, -1]
    print('MinDist:', minDist)
    return lable


if __name__ == "__main__":
    x1 = np.array([1, 1])
    x2 = np.array([2, 1])
    x3 = np.array([4, 3])
    x4 = np.array([5, 4])

    testX = np.vstack((x1, x2, x3, x4))

    res = kmeans(testX, 2, 10)
    print('result:\n', res)

    
