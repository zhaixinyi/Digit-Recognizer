import csv
import math
import random
import numpy as np
from tqdm import tqdm

def loadTrainData():
	l = []
	with open('train.csv', 'r') as f:
		lines = csv.reader(f)
		for line in lines:
			l.append(line)
		l.remove(l[0])
		l = np.array(l)
		random.shuffle(l)
		label = l[:,0]
		data = l[:,1:]
		label = np.int32(label)
		data = np.int32(data)
		return data, label

def loadTestData():
	l = []
	with open('test.csv', 'r') as f:
		lines = csv.reader(f)
		for line in lines:
			l.append(line)
		l.remove(l[0])
		data = np.array(l)
		return data

def KNN(Target, traindata, trainlabel, k = 1):
	target = np.mat(Target)
	traindata = np.mat(traindata)
	trainlabel = np.mat(trainlabel)

	size = traindata.shape[0]
	sub = np.array(abs(traindata - np.tile(target, (size, 1))))
	distances = sub.sum(axis = 1)
	distancesorted = distances.argsort()

	for i in range(k):
		Targetlabel = trainlabel[0, distancesorted[i]]
	return Targetlabel

def saveResult(result):
	with open('res.csv', 'w') as f:
		csv_writer = csv.writer(f)
		csv_writer.writerow(['Imageid', 'Label'])
		for item in result:
			csv_writer.writerow(item)

if __name__ == '__main__':
	train_data, train_label = loadTrainData()
	test_data = loadTestData()
	# err = 0

	result = []
	for i in tqdm(range(len(test_data))):
		x = test_data[i]
		# print(x.dtype)
		# print(train_data.dtype)
		x = x.astype('int32') 
		image_id, label = i + 1, KNN(x, train_data, train_label, 1)
		result.append([i, label])
	saveResult(result)


