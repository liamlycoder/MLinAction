# !/usr/bin/python
# coding: utf8
# @Time    : 2018-07-08 9:56
# @Author  : Liam
# @Email   : luyu.real@qq.com
# @Software: PyCharm
from numpy import *
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

# 定义结点
class cluster_node:
    def __init__(self, vec, left=None, right=None, distance=0.0, id=None, count=1):
        self.vec = vec
        self.left = left
        self.right = right
        self.distance = distance
        self.id = id
        self.count = count

# 欧式距离
def L2dist(v1, v2):
    return sqrt(sum((v1 - v2)**2))

# 绝对值距离
def L1dist(v1, v2):
    return sum(abs(v1 - v2))

# 层次聚类算法
def hcluster(features, distance=L2dist):
    distances = {}
    currentclusterid = 1

    clust = [cluster_node(array(features[i]), id=i) for i in range(len(features))]

    while len(clust) > 1:
        lowestpair = (0, 1)
        closest = distance(clust[0].vec, clust[1].vec)

        for i in range(len(clust)):
            for j in range(i+1, len(clust)):
                if (clust[i].id, clust[j].id) not in distances:
                    distances[(clust[i].id, clust[j].id)] = distance(clust[i].vec, clust[j].vec)
                d = distances[(clust[i].id, clust[j].id)]
                if d < closest:
                    closest = d
                    lowestpair = (i, j)
        mergevec = [(lowestpair[0].vec[i] + lowestpair[1].vec[i])/2.0 for i in range(len(clust[0].vec))]
        newcluster = cluster_node(array(mergevec), left=clust[lowestpair[0]], right=clust[lowestpair[1]], distance=closest, id=currentclusterid)

        currentclusterid -= 1
        del clust[lowestpair[0]]
        del clust[lowestpair[1]]
        clust.append(newcluster)
    return clust[0]

# 提取聚类
def extract_clusters(clust, dist):
    clusters = {}
    if clust.distance < dist:
        return [clust]
    else:
        cl = []
        cr = []
        if clust.left != None:
            cl = extract_clusters(clust.left, dist=dist)
        if clust.right != None:
            cr = extract_clusters(clust.right, dist=dist)
        return cl+cr

# 获取聚类的元素
def get_cluster_elements(clust):
    if clust.id >= 0:
        return [clust.id]
    else:
        cl = []
        cr = []
        if clust.left != None:
            cl = get_cluster_elements(clust.left)
        if clust.right != None:
            cr = get_cluster_elements(clust.right)
        return cl+cr

# 打印结点
def printclust(clust, labels=None, n=0):
    for i in range(n):
        print(' ')
    if clust.id < 0:
        print('-')
    else:
        if labels == None:
            print(clust.id)
        else:
            print(labels[clust.id])
    if clust.left != None:
        printclust(clust.left, labels=labels, n=n+1)
    if clust.right != None:
        printclust(clust.right, labels=labels, n=n+1)

# 获得高度
def getheight(clust):
    if clust.left == None and clust.right == None:
        return 1
    return getheight(clust.left) + getheight(clust.right)

# 获得深度
def getdepth(clust):
    if clust.left == None and clust.right == None:
        return 0
    return max(getdepth(clust.left), getdepth(clust.right)) + clust.distance

    
