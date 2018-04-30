# -*- coding:utf-8 -*-
# __author__:Luyu-Liam
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
from SexClassifierByKNN import file2matrix

'''
用于数据可视化，画出训练集中身高与体重的散点图
'''


def showdatas(datingDataMat, datingLabels):
    # 设置汉字格式
    font = FontProperties(fname=r"simsun.ttc", size=14)
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('black')
        if i == 2:
            LabelsColors.append('orange')
    axs.scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 1], color=LabelsColors, s=20, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs0_title_text = axs.set_title(u'身高与体重散点图', FontProperties=font)
    axs0_xlabel_text = axs.set_xlabel(u'身高', FontProperties=font)
    axs0_ylabel_text = axs.set_ylabel(u'体重', FontProperties=font)
    plt.setp(axs0_title_text, size=16, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=11, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=11, weight='bold', color='black')

    # 设置图例
    Female = mlines.Line2D([], [], color='black', marker='.',
                              markersize=8, label='Female')
    Male = mlines.Line2D([], [], color='orange', marker='.',
                               markersize=8, label="Male")
    # 添加图例
    axs.legend(handles=[Female, Male])
    # 显示图片
    plt.show()


if __name__ == "__main__":
    # 打开的文件名
    filename = "dataSetEx.txt"
    # 打开并处理数据
    datingDataMat, datingLabels = file2matrix(filename)
    showdatas(datingDataMat, datingLabels)
