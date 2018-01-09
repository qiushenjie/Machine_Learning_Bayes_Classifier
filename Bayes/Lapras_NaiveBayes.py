#《机器学习》周志华 习题7.3 拉普拉斯修正的朴素贝叶斯分类器
from numpy import *
import numpy as np
import pandas as pd

#读取文件格式为xlsx的数据
def dataLoad(filename):
	df = pd.read_excel(fliename, header = None) #这里为获取属性列表将header设置成None
	propLabelSet = df.values[0:1, 1:-1] #属性列表
	dataSet = df.values[1:,1:-1] #样本数据
	labelSet = df.values[1:,-1:] #样本分类结果数据
	return propLabelSet, dataSet, labelSet

# #计算训练样本加上预测样本后，每个属性类别的个数，并保存在一个列表中用于在函数trainNB()中进行拉普拉斯修正
def getPropNum(TTdataSet):
	n = shape(TTdataSet)[1]
	newPropNum = zeros(n)
	for i in range(n):
		propCate = list(set(TTdataSet[:,i]))
		newPropNum[i] = len(propCate)
	return (newPropNum)


#训练朴素贝叶斯，将每个属性标签的后验概率并存储在模型字典里，对测试样本分类时可直接调用
def trainNB(propLabelSet, trainSet, allLabelSet, numDataProp):
	numTrain, numProp = shape(trainSet)
	modelDict = {} #创建保存模型的字典
	priorProb = {} #创建用于保存先验概率的字典
	numYandN = {}
	tempCate = allLabelSet.flatten() 
	labelSet = list(set(tempCate))
	N = len(labelSet)
	for i in range(0,len(labelSet)):
		priorProb[labelSet[i]] = float((tempCate.tolist().count(labelSet[i]) + 1)/(numTrain + N))  #统计好、坏瓜个数并计算其先验概率，写入字典priorProb中。这里count无法对array进行计数，因此只能转为list
		numYandN[labelSet[i]] = tempCate.tolist().count(labelSet[i])
	for i in range(numProp):
		discretePropDict = {} #存储每个离散属性标签的字典
		#属性标签为离散型时
		if type(trainSet[0][i]).__name__=='str':
			Ni = numDataProp[i]
			discretePropSet = list(set(trainSet[:,i]))
			for item in discretePropSet: #对每个属性类别的分类结果进行计数
				Y_and_N = {}  #存储每个属性标签下已知好坏瓜后此标签的概率
				for result in labelSet:
					countAll = tempCate.tolist().count(result)
					countData = 0
					for j in range(numTrain): #遍历每个数据
						if (trainSet[j][i] == item) and (allLabelSet[j] == result):
							countData += 1
					Y_and_N[result] = float((countData + 1)/float(countAll + Ni))
				discretePropDict[item] = Y_and_N
			modelDict[propLabelSet[0][i]] = discretePropDict

		#属性标签为连续性时，用极大似然估计求概率分布的均值和方差
		else:
			Y_and_N = {}  #存储每个属性下已知好坏瓜后此属性的均值或方差的大小
			for result in labelSet:
				exp_and_varroot = {}
				countAll = tempCate.tolist().count(result)
				expec = float(sum([trainSet[j,i] for j in range(numTrain) if allLabelSet[j] == result]))/countAll #求均值
				exp_and_varroot['均值'] = expec
				var = float(sum([float((trainSet[j,i]-expec))**2 for j in range(numTrain) if labelData[j] == result])/countAll) #求方差
				var_root = sqrt(var) #求标准差
				exp_and_varroot['标准差'] = var_root
				Y_and_N[result] = exp_and_varroot
			modelDict[propLabelSet[0][i]] = Y_and_N
	return modelDict, priorProb, numYandN

#计算连续型属性的概率
def calContinuiousProb(data, expec, varroot):
	return float((1/(sqrt(2*pi)*varroot))*exp(-(data - expec)**2/(2*varroot**2)))


#对测试数据进行预测
def testDataPredict(testSet, trainModel, priorProba, propLabel, numDataPro, numYandN):
	m, n = shape(testSet)
	for i in range(m):
		prob = {}
		for item in priorProba.keys():
			prob[item] = 1.0*priorProba[item]
			for j in range(n):
				if type(testSet[0][j]).__name__ != 'str':
					prob[item] *= calContinuiousProb(testSet[i][j], trainModel[propLabel[0][j]][item]['均值'], trainModel[propLabel[0][j]][item]['标准差'])
				else:
					if testSet[i][j] in trainModel[propLabel[0][j]].keys():
						prob[item] *= trainModel[propLabel[0][j]][testSet[i][j]][item]
					else:
						print('%s不在训练样本的%s属性中出现过，是新的属性类别' % (testSet[i][j], propLabel[0][j]))
						prob[item] *= float(1/(numYandN[item] + numDataPro[j]))
		print(prob)
		if prob['是'] >prob['否']:
			print('测试样本%d是好瓜' %(i))
		else:
			print('测试样本%d是坏瓜' %(i))


if __name__=="__main__": 
	fliename = 'watermelon_4.3.xlsx'
	propLabel, trainData, labelData = dataLoad(fliename)
	df = pd.read_excel('watermelon_4.3_test.xlsx')
	testData = df.values[:,1:-1]
	TTdata = np.concatenate((trainData,testData))
	numPropData = getPropNum(TTdata)
	modelDict, priorProb, numYesandNo = trainNB(propLabel, trainData, labelData, numPropData)
	testDataPredict(testData, modelDict, priorProb, propLabel, numPropData, numYesandNo)
