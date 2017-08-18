# -*- coding: utf-8 -*-

import numpy as np
import math  
import copy  
import matplotlib.pyplot as plt  
import matplotlib.mlab as mlab  

isdebug = True

def InitData(Sigma,mu_r,k,N):  
	global X  
	global Mu  
	global Expectations  
	global guass_real
	global Sigmas
	guass_real = [[] for i in range(k)]
	X = np.zeros((1,N))  
	Mu = np.random.random(k)
	Expectations = np.zeros((N,k))  
	for i in range(0,N):  
		if np.random.random(1) > 0.5:  
			X[0,i] = np.random.normal(mu_r[0], Sigma[0])
			guass_real[0].append(X[0,i])
		else:  
			X[0,i] = np.random.normal(mu_r[1], Sigma[1])
			guass_real[1].append(X[0,i])
	sum1 = X.sum()
	X2 = X*X
	sum2 = X2.sum()
	mean = sum1/N
	var = sum2/N-mean**2
	Sigmas = (np.array([var,var]))**0.5
	if isdebug:  
		print("***********")
		print("初始观测数据X：")
		print(X)
		
# EM算法：步骤1，计算E[zij]  
def Estep(Sigmas, k, N):  
	global Expectations  
	global Mu  
	global X  
	for i in range(0,N):  
		Denom = 0 
		Numer = [0.0] * k
		for j in range(0,k):  
			Numer[j] = math.exp((-1/(2*(float(Sigmas[j]**2))))*(float(X[0,i]-Mu[j]))**2)  
			Denom += Numer[j]
		for j in range(0,k): 
			Expectations[i,j] = Numer[j] / Denom  
	if isdebug:  
		print("***********")
		print("隐藏变量E（Z）：")
		print(Expectations)
		
# EM算法：步骤2，求最大化E[zij]的参数Mu  
def Mstep(k,N):  
	global Expectations  
	global X  
	for j in range(0,k):  
		Numer1 = 0   
		Numer2 = 0  
		Denom = 0 
		for i in range(0,N):
			Numer1 += Expectations[i,j]*X[0,i]  
			Denom += Expectations[i,j]
			Mu[j] = Numer1 / Denom
			Numer2 += Expectations[i,j]*(X[0,i]-Mu[j])**2
			Sigmas[j] = (Numer2 / Denom)**0.5
		
# 算法迭代iter_num次，或达到精度Epsilon停止迭代  
def run(Sigma,mu_r,k,N,iter_num,Epsilon):  
	InitData(Sigma,mu_r,k,N)  
	print("初始<u1,u2>:", Mu)
	print("初始<Sigma1,Sigma2>",Sigmas)
	for i in range(iter_num):  
		Old_Mu = copy.deepcopy(Mu)  
		Old_Sigmas = copy.deepcopy(Sigmas)
		Estep(Sigmas,k,N)  
		Mstep(k, N)
		print('迭代次数:',i,'μ:',Mu,'Sigma:',Sigmas)
		if sum(abs(Mu - Old_Mu)) < Epsilon and sum(abs(Sigmas - Old_Sigmas)) < Epsilon:  
			break  

if __name__ == '__main__':
	sigma = [8,10]   # 生成各高斯分布的标准差
	mu_r = [40,20]    # 高斯分布的均值 用于产生样本
	k = 2       # 高斯分布的个数
	N = 5000    # 样本个数
	iter_num = 3000 # 最大迭代次数
	epsilon = 0.0001    # 当两次误差小于这个时退出
	run(sigma,mu_r,k,N,iter_num,epsilon)  
	guass_pre = [[] for i in range(k)]
	g = []
	for i in range(len(Expectations)):
		g_pre = np.where(Expectations[i,:] == max(Expectations[i]))[0][0]
		if abs(Expectations[i][0]-Expectations[i][1]) < 0.5:
			r = np.random.uniform(0,1,1)
			if max(Expectations[i]) < r :
				g_pre = np.where(Expectations[i,:] == min(Expectations[i]))[0][0]	
		guass_pre[g_pre].append(X[0,i])
	fig,ax = plt.subplots(2,2)
	for i in range(len(guass_pre)):
		ax = plt.subplot(2,2,i+3)
		ax.set_title("Prediction\n"+'μ='+str(round(Mu[i],4))+'    sigma='+str(round(Sigmas[i],4)))
		ax.hist(guass_pre[i],50,facecolor='yellowgreen',alpha=0.75,normed=1)
		ax.set_xticks(np.linspace(0,60,7))
	for i in range(len(guass_real)):
		ax = plt.subplot(2,2,i+1)
		ax.set_title("Real\n"+'μ='+str(round(mu_r[i],4))+'    sigma='+str(round(sigma[i],4)))
		ax.hist(guass_real[i],50,facecolor='red',alpha=0.75,normed=1)
		ax.set_xticks(np.linspace(0,60,7))
	fig.subplots_adjust(wspace=0.6,hspace=0.8)
	plt.show()
