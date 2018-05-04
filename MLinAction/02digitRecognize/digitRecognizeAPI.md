说明
=====

依然是knn算法，这次实现的是“手写数字识别”项目，由于数据集较大，请自行下载：[dataSet](https://www.kaggle.com/c/digit-recognizer/data)

这次的knn是直接调用的sklearn里面的，省去自己手写的麻烦。

训练集为已经做过预处理的文本文件，每一行的从第二列开始为每个像素点的灰度值（已经将二维图片转化为一维向量存储）。

数据集较大，程序大概运行20到30分钟。

运算结果输出到csv文件中，第一列代表序号，第二列为识别结果。


手写数字识别案例中用到的API
=========================

###一、有关对csv文件的操作：
    1.  pandas.read_csv() 
        直接传入csv文件名，读入文件数据，返回的是一个pandas对象。
    2.  pandasobject.values()
        读入矩阵中的指定数据，由一个pandas对象调起。
    3.  csv.writer(myFile).writerow()
        对文件执行写入操作，后面的writerow()表示写入一行。
        
        

###二、knn有关的操作：
    1.  KNeighborsClassifier()
        sklearn中的knn算法，默认k为5，可以传入n_neighbors值更换。
    2.  fit(X, Y)
        参数有两个， X为training data，Y为target values。这个函数的目的是做一个适配。具体用法见[官网](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier.fit ):
    3.  numpy.raver()
        这个一个扁平化函数，可以将二维矩阵降低为一维显示。
        它有一个兄弟函数flatten()，同样也是降维，但是不同的是，flatten()返回的是一个拷贝，而raver()返回的是视图，修改其值可以影响到原来的矩阵。
        
   
     
###三、时间函数：
    1. time.time()
        返回时间，可以用来测试程序运行时间。