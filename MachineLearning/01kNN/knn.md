背景：通过KNN算法对性别进行分类
=========================
数据集包含了“身高”和“体重”两个属性。标签f表示女性，m表示男性。


####文件说明：
- dataSetEx.txt:训练集
- datSetTest.txt:测试集
- simsun.ttc:字体
- ShowDataSet.py:数据可视化文件。绘制出训练集的散点分布图
- SexClassifier.py:kNN分类算法文件




kNN算法中用到的API：
================

### 一、matplotlib库里面的：

    1)  plt.subplots(nrows, ncols, sharex, sharey, figszie=(x, y))
        用于画（多个）子图，其中nrows, ncols就表示子图以nrows*ncols矩阵的形式呈现，例如，2*2就表示四个子图。
        sharex和sharey表示是否共享x或y轴，可选参数：True, False, row(这一行共享), col(这一列共享).
        figsize(x, y)表示图的尺寸：x*y

    2)  scatter(x, y, color, s, alpha)
        画散点图，x, y分别表示两坐标轴的对应的数据，color为颜色值，s表示散点图形的大小，alpha表示散点图形的透明度
        关于scatter函数完整的有九个参数。
        详见https://matplotlib.org/api/_as_gen/matplotlib.pyplot.scatter.html?highlight=scatter#matplotlib.pyplot.scatter

    3)  set_title, set_xlable, set_ylable分别表示设置标题，设置x轴名称，y轴名称

    4)  setp(line, linestyle...)
        用来给指定的line设置样式(字体，颜色，大小......)

    5)  matplotlib.lines.Line2D()
        这个函数主要用来设置一个线条。参数非常多，官网给出了40个参数说明。
        详见https://matplotlib.org/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D

    6)  matplotlib.legend(handles)
        在位置loc上的轴上放置一个图例。这里的参数handles表示“要添加到图例中的艺术家列表(线条、补丁)”
        
### 二、numpy库里面的：
    1） numpy.tile(A, (x, y))
       A为一个矩阵，（x, y）为一个元祖。表示A矩阵在行方向上重复x次，在列方向上重复y次
       如果将参数（x, y）替换为一个数n，则表示在列方向上重复n次，默认行1次。
    2) A.min(x) / A.max(x)
        A为一个矩阵，当x为0时，返回的是一行数据，其中每个元素表示该列的最小值；
        当x为1时，返回的也是一行数据，其中每个元素表示该行的最小值。
        max函数亦然。
    3) numpy.argsort()
       排序函数，返回的是角标。排序方式默认为升序。
       用法很丰富，详见官网：https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html?highlight=argsort#numpy.argsort
       