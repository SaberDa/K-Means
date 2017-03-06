# -*- coding: utf-8 -*-

from numpy import *
import operator

# 从os模块中拿出listdir，其作用为可以列出给定目录名

from os import listdir

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group,labels

'''伪代码：
对未知类别属性的数据集中的每个点依次执行以下操作：
（1）计算一直类别数据集中的每个点与当前点之间的距离
（2）按照距离递增次序排序
（3）选取与当前点距离最小的k个点
（4）确定前k个点所在类别出现的频率
（5）返回前k个点出现频率最高的类别作为当前点的预测分类
'''

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # 距离计算
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    # 选择距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    # 排序
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    # 得到文件行数
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    # 创建返回的numpy库
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    # 解析文件数据到列表
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector


# 归一化
# newValue = (oldValue -min) / (max - min)
def autoNorm(dataset):
    minvals = dataset.min(0)
    maxvals = dataset.max(0)
    ranges = maxvals - minvals
    normDataSet = zeros(shape(dataset))
    m = dataset.shape[0]
    normDataSet = dataset - tile(minvals, (m,1))
    # 特征值相除
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet,ranges,minvals

def datingClassTest():
    hoRatio = 0.50      #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    print(errorCount)

    # 使用k-临近算法识别手写识别系统
    '''
    （1）收集数据：提供文本文件
    （2）准备数据：编写函数img2vector()，将图像格式转换为分类器可识别的向量格式
    （3）分析数据：检查数据，确保其符合要求
    （4）训练算法
    （5）测试算法：编写函数使用提供的部分数据集作为测试样本。测试样本与非测试赝本的主要区别在于测试样本是已经分类的数据,
    如果预测分类与实际分类不同，则标记为一个错误
    （6）使用算法
    '''
# 将图像转换为测试向量

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    # 获取目录内容
    trainingFileList = listdir('digits/trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    # 从文件名解析分类内容
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('digits/trainingDigits/%s' % fileNameStr)
    testFileList = listdir('digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f"  % (errorCount/float(mTest)))





