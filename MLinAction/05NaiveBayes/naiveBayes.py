# !/usr/bin/python
# coding: utf8
# @Time    : 2018/5/13 12:30
# @Author  : Liam
# @Email   : luyu.real@qq.com
# @Software: PyCharm
import re
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report


# 文本解析
def textParse(bigStr):
    """
    通过正则表达式将字符串进行切割。
    :param bigStr: 字符串
    :return: 切割后的单词列表，把长度小于2的单词自动忽略
    """
    tokenList = re.split(r'\W+', bigStr)
    return [tok.lower() for tok in tokenList if len(tok) > 3]


# 数据解析
def loadText():
    '''
    读取数据集，生成特征矩阵和标签向量，并构建词典
    :return: docList:特征矩阵，由文本中的单词构成. classList:标签  vocabSet:词典
    '''
    docList = []
    classList = []
    fullList = []
    for i in range(1, 26):
        try:
            words = textParse(open('input/spam/{}.txt'.format(i)).read())
        except:
            words = textParse(open('input/spam/{}.txt'.format(i), encoding='Windows 1252').read())
        docList.append(words)
        classList.append(1)
        fullList.extend(words)
        try:
            words = textParse(open('input/ham/{}.txt'.format(i)).read())
        except:
            words = textParse(open('input/ham/{}.txt'.format(i), encoding='Windows 1252').read())
        docList.append(words)
        classList.append(0)
        fullList.extend(words)
    vocabSet = set(fullList)
    vocabSet = list(vocabSet)
    return docList, classList, vocabSet


# 单词向量化
def word2Vec(docList, vocabSet):
    '''
    也就是将文本单词进行向量化
    :param docList: 特征矩阵，内容为单词
    :param vocabSet: 词典
    :return: 进行向量化以后的特征矩阵
    '''
    m = len(docList)
    n = len(vocabSet)
    xVec = np.zeros((m, n))
    for sample in range(m):
        for word in docList[sample]:
            if word in vocabSet:
                xVec[sample][vocabSet.index(word)] = xVec[sample][vocabSet.index(word)] + 1
    return xVec


# 训练数据
def trainNB(xTrain, yTraxin):
    '''
    通过sklearn中的高斯贝叶斯的方式进行训练
    :param xTrain: 特征矩阵
    :param yTraxin: 标签向量
    :return:
    '''
    gnb = GaussianNB()
    gnb.fit(xTrain, yTraxin)
    return gnb


# 模型评估
def showPrecisionRecall(xTest, yTest, gnb):
    '''
    计算准确率和召回率，并打印评估报告
    :param xTest: 测试集的输入变量x
    :param yTest: 测试集的输出变量y
    :param gnb: 训练好的高斯贝叶斯模型
    :return: precision: 准确率   recall：召回率
    '''
    yPred = gnb.predict(xTest)
    targetNames = ['ham', 'spam']
    precision, recall, thresholds = precision_recall_curve(yTest, yPred)
    print(classification_report(yTest, yPred, target_names=targetNames))
    return precision, recall


if __name__ == "__main__":
    # 加载数据
    doclist, y, vocabList = loadText()

    # 特征向量化
    x = word2Vec(doclist, vocabList)

    # 将数据集按照8：2的比例进行训练集、测试集的划分
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2)

    # 训练模型
    gnb = trainNB(xTrain, yTrain)

    # 模型评估
    showPrecisionRecall(xTest, yTest, gnb)
