# -*- coding:utf-8 -*-  
import numpy as np


class Vector(object):
    def __init__(self, vector, cluNum):
        self.vector = np.array(vector, dtype='float64')
        self.cluster = np.random.choice(cluNum, 1)[0]
        self.centerD = [0 for i in range(cluNum)]

    def CalculationDistance(self, centerVs):
        for i in range(len(centerVs)):
            self.centerD[i] = np.sqrt(np.sum(np.square(self.vector - centerVs[i])))
        self.cluster = self.centerD.index(min(self.centerD))


def ComputingCenterVector(vList):
    clusterDic = {}
    centerVs = []
    for i in vList:
        if i.cluster not in clusterDic.keys():
            clusterDic[i.cluster] = []
        clusterDic[i.cluster].append(i)
    keys = sorted(clusterDic.keys())
    for key in keys:
        m = clusterDic[key][0].vector
        for x in range(1, len(clusterDic[key])):
            m = m + clusterDic[key][x].vector
        centerVs.append(m / len(clusterDic[key]))
    for i in vList:
        i.CalculationDistance(centerVs)
    return centerVs


def main():
    data = []
    with open('iris.data.txt') as f:
        for i in f.readlines():
            data.append(i.strip().split(',')[:-1])
    vList = []
    for i in data:
        vList.append(Vector(i, 3))
    oldCenterVs = ComputingCenterVector(vList)
    while True:
        m = ComputingCenterVector(vList)
        count = 0
        for i in range(len(m)):
            if (m[i] == oldCenterVs[i]).all():
                count += 1
            if count == len(m):
                for i in vList:
                    print('向量：', i.vector, '聚类：', i.cluster)
                return 1
        oldCenterVs = m


main()
