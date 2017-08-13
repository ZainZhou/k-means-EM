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

def printResult(vList):
    clusterDic = {}
    for v in vList:
        if v.cluster not in clusterDic.keys():
            clusterDic[v.cluster] = []
        clusterDic[v.cluster].append(v.vector)
    for key in clusterDic:
        print('\n'+'聚类'+str(key)+'(数目:'+str(len(clusterDic[key]))+')'+':\n')
        for i in clusterDic[key]:
            print(str(i))

def main():
    data = []
    epsilon = 0.0001 # 当两次计算后每个中心向量误差小于千分之一时结束
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
            if sum(abs(m[i]-oldCenterVs[i])) < epsilon:
                count += 1
            if count == len(m):
                printResult(vList)
                return 1
        oldCenterVs = m
main()