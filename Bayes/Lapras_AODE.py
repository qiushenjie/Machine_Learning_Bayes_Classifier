from numpy import *
import numpy as np
import pandas as pd
from collections import Counter

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
def trainNB(propLabelSet, trainSet, allLabelSet, numDataProp, TTdataSet):
	numTrain, numProp = shape(trainSet)
	prob_c_xi = {}
	D_c_xi = {}
	prob_xj_c_xi = {}
	D_xj_c_xi = {}
	tempCate = allLabelSet.flatten() 
	labelSet = list(set(tempCate))
	for i in range(numProp):
		if type(trainSet[0][i]).__name__ == 'str':
			Ni = numDataProp[i]
			D_c_i_j = {}
			prob_j_c_i = {}
			prop_i_set = list(set(TTdataSet[:,i]))
			#求prob_c_xi的字典，其格式为{'是':{i属性:{j属性类别1:概率, i属性类别2:概率},...},'否':...},三层字典
			#求P(xj|c, xi)的字典，格式为 {i属性:{i属性类别1:{'是':{j属性:{j属性类别1:概率, j属性类别2:概率},...},...},...},...},五层字典
			for itemI in prop_i_set:
				D_c_ij = {}
				prob_c_ij = {}
				for catetem in labelSet:
					D_c_xjj = {}
					prob_c_xjj = {}
					D_c_xii = {}
					prob_c_xii = {}
					for j in range(numProp):
						if type(trainSet[0][j]).__name__ != 'str': continue
						else:
							Ni = numDataProp[j]
							D_cjj = {}
							prob_cjj = {}
							D_cii = {}
							prob_cii = {}
							prop_j_set = list(set(TTdataSet[:,j]))
							for itemJ in prop_j_set:
								numcount = 0								
								numcountj = 0								
								numcountcij = 0
								for n in range(numTrain):
									if (allLabelSet[n] == catetem) and (trainSet[n, j] == itemJ): numcount += 1
									if (allLabelSet[n] == catetem) and (trainSet[n, j] == itemJ) and (j != i): numcountj += 1  #本属性不能作为本属性的超父
									if (allLabelSet[n] == catetem) and (trainSet[n, j] == itemJ) and (trainSet[n, i] == itemI) and (j != i): numcountcij += 1#本属性不能作为本属性的超父

								D_cjj[itemJ] = numcount #累计个数字典
								prob_cjj[itemJ] = (numcount+1)/(numTrain+Ni*len(labelSet)) #计算概率

								D_cii[itemJ] = numcountcij
								prob_cii[itemJ] = float((numcountcij+1)/(numcountj  + Ni))

							D_c_xjj[propLabelSet[0][j]] = D_cjj
							prob_c_xjj[propLabelSet[0][j]] = prob_cjj

							D_c_xii[propLabelSet[0][j]] = D_cii
							prob_c_xii[propLabelSet[0][j]] = prob_cii	

					D_c_xi[catetem] = D_c_xjj
					prob_c_xi[catetem] = prob_c_xjj

					D_c_ij[catetem] = D_c_xii
					prob_c_ij[catetem] = prob_c_xii

				D_c_i_j[itemI] = D_c_ij
				prob_j_c_i[itemI] = prob_c_ij
			D_xj_c_xi[propLabelSet[0][i]] = D_c_i_j
			prob_xj_c_xi[propLabelSet[0][i]] = prob_j_c_i

		#对连续性属性，用的是朴素贝叶斯对连续型属性的处理方式
		else:   
			Y_and_N = {}  #存储每个属性下已知好、坏瓜后此属性的均值或方差的大小
			for result in labelSet:
				exp_and_varroot = {}
				countAll = tempCate.tolist().count(result)
				expec = float(sum([trainSet[k,i] for k in range(numTrain) if allLabelSet[k] == result]))/countAll #求均值
				exp_and_varroot['均值'] = expec
				var = float(sum([float((trainSet[k,i]-expec))**2 for k in range(numTrain) if labelData[k] == result])/countAll) #求方差
				var_root = sqrt(var) #求标准差
				exp_and_varroot['标准差'] = var_root
				Y_and_N[result] = exp_and_varroot
			prob_xj_c_xi[propLabelSet[0][i]] = Y_and_N	

	return prob_c_xi, D_c_xi, prob_xj_c_xi, D_xj_c_xi #最后得到概率和个数的字典

#计算连续型属性的概率
def calContinuiousProb(data, expec, varroot):
	return float((1/(sqrt(2*pi)*varroot))*exp(-(data - expec)**2/(2*varroot**2)))

#对测试数据进行预测
#计算样本分别是好瓜和坏瓜的概率，对每个样本先固定一个属性i，其对应的值为xi，求此时xi下 “P(c,xi)*(分类为好瓜或者坏瓜的情况下每一个j(j!=i)属性下样本属性类别取值为xj的概率的乘积” 的值，遍历样本所有属性i，并求和
def testDataPredict(trainSet, testSet, Proba_xj_c_xi, Proba_c_xi, propLabel, numDataPro, allLabelSet):
	m, n = shape(testSet)
	for k in range(m): #遍历每个测试样本
		p_c_x = {} #为每个样本创建一个字典，用来存储此样本为好瓜、坏瓜的概率 
		tempCate = allLabelSet.flatten() 
		labelSet = list(set(tempCate))
		for itemYorN in labelSet:
			for i in range(n):
				if type(testSet[0][i]).__name__ == 'str':
					count_c_xi = 0.0
					
					count_xj_c_xi = 1.0
					for j in range(n):
						if type(testSet[0][j]).__name__ != 'str' or i == j: continue
						else:
							count_xj_c_xi *= Proba_xj_c_xi[propLabel[0][i]][testSet[k][i]][itemYorN][propLabel[0][j]][testSet[k][j]]
					count_c_xi += Proba_c_xi[itemYorN][propLabel[0][i]][testSet[k][i]] * count_xj_c_xi
				elif type(testSet[0][i]).__name__ != 'str': 
					continue #因为书中未介绍AODE分类器对连续性数据的处理，这里只能跳过
					# count_c_xi +=  calContinuiousProb(testSet[k][i], Proba_xj_c_xi[propLabel[0][i]][itemYorN]['均值'], Proba_xj_c_xi[propLabel[0][i]][itemYorN]['标准差'])

			p_c_x[itemYorN] = count_c_xi

		print(p_c_x)
		if p_c_x['是'] >p_c_x['否']:
			print('测试样本%d是好瓜' %(k))
		else:
			print('测试样本%d是坏瓜' %(k))


if __name__=="__main__": 
	fliename = 'watermelon_4.3.xlsx'
	propLabel, trainData, labelData = dataLoad(fliename)
	df = pd.read_excel('watermelon_4.3_test.xlsx')
	testData = df.values[:,1:-1]
	TTdata = np.concatenate((trainData,testData))
	numPropData = getPropNum(TTdata)
	dictProb_c_xi, dictD_c_xi, dictProb_xj_c_xi, dictD_xj_c_xi = trainNB(propLabel, trainData, labelData, numPropData, TTdata)
	testDataPredict(trainData, testData, dictProb_xj_c_xi, dictProb_c_xi, propLabel, numPropData, labelData)

