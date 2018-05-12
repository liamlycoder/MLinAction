# !/usr/bin/python
# coding: utf8
# @Time    : 2018/5/12 13:47
# @Author  : Liam
# @Email   : luyu.real@qq.com
# @Software: PyCharm

import numpy as np
from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pydotplus
from sklearn.externals.six import StringIO


# 解析数据
def createDataSet():
    data = []
    lables = []
    fr = open("input/dataSet.txt")
    for line in fr.readlines():
        tokens = line.strip().split()
        data.append([float(tk) for tk in tokens[:-1]])
        lables.append(tokens[-1])
    x = np.array((data))
    lables = np.array(lables)
    y = np.zeros(lables.shape[0])

    # numpy的布尔型索引
    y[lables == 'm'] = 1
    y[lables == 'M'] = 1
    return x, y

# 训练数据
def predictTrain(x_train, y_train):
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf.fit(x_train, y_train)
    # feature_importances表示特征的影响力，值越大表示该特征在分类中的作用越大
    # print(clf.feature_importances_)
    y_pre = clf.predict(x_train)
    # print(x_train)
    # print(y_pre)
    # print(y_train)
    # print(y_pre == y_train)
    return y_pre, clf

# 模型评估
def showPrecisionRecall(x_test, y_test, clf):
    y_pre = clf.predict(x_test)
    precision, recall, thresholds = precision_recall_curve(y_test, y_pre)
    target_names = ['female', 'male']
    print(classification_report(y_test, y_pre, target_names=target_names))
    return precision, recall, thresholds

# 数据可视化
def showTree(clf):
    dotData = StringIO()
    tree.export_graphviz(clf, out_file=dotData)
    graph = pydotplus.graph_from_dot_data(dotData.getvalue())
    graph.write_pdf("output/result.pdf")

if __name__ == '__main__':
    x, y = createDataSet()
    # 拆分训练数据与测试数据，这里是按照8:2的比例拆分
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # 得到训练集的预测结果
    y_pre, clf = predictTrain(x_train, y_train)

    # 计算准确率和召回率，可以将计算结果返回，当然不需要的话可以选择不接受，因为在函数体内部我们已经打印出模型评估报告了
    showPrecisionRecall(x_test, y_test, clf)

    # 可视化
    showTree(clf)



