# !/usr/bin/python
# coding: utf8
# @Time    : 2018/5/15 20:50
# @Author  : Liam
# @Email   : luyu.real@qq.com
# @Software: PyCharm
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer
from neuralNetwork import NeuralNetwork
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    digits = load_digits()
    X = digits.data
    y = digits.target
    X -= X.min()
    X /= X.max()

    nn = NeuralNetwork([64, 100, 10], 'logistic')
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    lable_train = LabelBinarizer().fit_transform(y_train)
    lable_test = LabelBinarizer().fit_transform(y_test)
    print("start fitting")
    nn.fit(X_train, lable_train, epochs=3000)
    predictions = []
    for i in range(X_test.shape[0]):
        o = nn.predict(X_test[i])
        predictions.append(np.argmax(o))
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))

    
