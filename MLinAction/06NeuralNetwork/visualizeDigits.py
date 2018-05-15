# !/usr/bin/python
# coding: utf8
# @Time    : 2018/5/15 20:46
# @Author  : Liam
# @Email   : luyu.real@qq.com
# @Software: PyCharm
from sklearn.datasets import load_digits
import pylab as pl

def showImg(i):
    digits = load_digits()
    pl.gray()
    pl.matshow(digits.images[i])
    pl.show()

if __name__ == "__main__":
    for i in range(10):
        showImg(i)

    
