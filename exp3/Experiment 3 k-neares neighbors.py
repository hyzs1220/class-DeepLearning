# coding=utf-8

import pandas as pd
import random 
import numpy as np
import matplotlib.pyplot as plt
import operator
from tqdm import tqdm

N = 11

# 读取数据
data_pd = pd.read_csv('winequality-red.csv', header=0, sep=';')
data = []
#  quality (score between 0 and 10)
labels = []
# 把输出分割出来
for _, row in data_pd.iterrows():
    data.append([row['fixed acidity'], row['volatile acidity'], row['citric acid'],
              row['residual sugar'], row['chlorides'], row['free sulfur dioxide'],
              row['total sulfur dioxide'], row['density'], row['pH'],
              row['sulphates'], row['alcohol']])
    labels.append(row['quality'])
# print(data[1])


# 打乱顺序进行训练和测试
# random.shuffle(data)
# print(len(data),len(data[0]))
# print(data)

# 划分训练数据和测试数据
count = len(data)
ratio = 0.8
split_point = int(count * ratio)
train_data = data[:split_point]
train_lables = labels[:split_point]
test_data = data[split_point:]
test_lables = labels[split_point:]
print('There {} datas in total, {} datas used for train, {} used for test'.format(count, split_point, count - split_point))



# K-nearest neighbors for classification
# test_data, train_data, train_lables, k
def classify(inX, dataSet, labels, k):

    # 计算欧式距离
    dataSetSize = len(dataSet)
    diffMat = np.tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5


    sortedDistIndicies=distances.argsort() # 排序并返回index
    # 选择距离最近的k个值
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    # 排序
    # print(classCount)
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


total = 0
for i in tqdm(range(len(test_data))):
# for i in range(2):
    out = classify(test_data[i], train_data, labels, 15)
    # print(out,test_lables[i])
    if out == test_lables[i]:
        total += 1

print('There {} test datas in total, accuracy is {} %'.format(len(test_data), (total / len(test_data))*100 ))