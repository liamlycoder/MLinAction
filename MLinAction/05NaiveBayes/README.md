朴素贝叶斯
=========

一、理论
--------
参见[http://sklearn.apachecn.org/cn/0.19.0/modules/naive_bayes.html#gaussian-naive-bayes](http://sklearn.apachecn.org/cn/0.19.0/modules/naive_bayes.html#gaussian-naive-bayes)

二、案例
-----------
通过朴素贝叶斯的方式对垃圾邮件进行分类。<br>
训练集input中有两个文件集：
- ham：非垃圾邮件
- spam：垃圾邮件<br>

对邮件文本内容按单词进行切分，然后向量化得到特征矩阵。通过sklearn中的贝叶斯分类算法进行训练，并进行模型评估。
