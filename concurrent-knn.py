import csv
import math
import random
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import time

train_data = []
train_label = []
val_data = []
val_label = []
test_data = []
test_label = []

N = 100
color = 100/255

def loadTrainData():
	l = []
	with open('train.csv', 'r') as f:
		lines = csv.reader(f)
		for line in lines:
			l.append(line)
		l.remove(l[0])
		l = np.array(l)
		# random.shuffle(l)
		label = l[:,0]
		data = l[:,1:]
		label = np.int32(label)
		data = np.int32(data)

		img = CutPicture(data)
		img = StretchPicture(img).reshape(N**2)


		#data = np.mat(data)
		#label = np.mat(label)
		return img, label

def loadTestData():
	l = []
	with open('test.csv', 'r') as f:
		lines = csv.reader(f)
		for line in lines:
			l.append(line)
		l.remove(l[0])
		data = np.array(l)
		data = np.int32(data)
		img = CutPicture(data)
		img = StretchPicture(img).reshape(N**2)
		return img

#切割图象
def CutPicture(img):
    size = JudgeEdge(img)
	return img[size[0]:size[1]+1, size[2]:size[3]+1]

def JudgeEdge(img):
	x1, y1, x2, y2 = 28, 28, 0, 0
	for i in range(28):
		for j in range(28):
			if img[i, j] > 0:
				x1 = min(x1, i)
				y1 = min(y1, j)
				x2 = max(x2, i)
				y2 = max(y2, j)
	return [x1, y1, x2, y2]

#拉伸图像
def StretchPicture(img):
	newImg = np.ones(N**2).reshape(N, N)
	newImg1 = np.ones(N ** 2).reshape(N, N)

	step1 = len(img[0])/N

	step2 = len(img)/N

	for i in range(len(img)):
		for j in range(N):
			newImg[i, j] = img[i, int(np.floor(j*step1))]

	for i in range(N):
		for j in range(N):
			newImg1[j, i] = newImg[int(np.floor(j*step2)), i]
	return newImg1



def KNN(target, traindata, trainlabel, k = 1):
	size = traindata.shape[0]
	distances = np.abs(traindata - np.tile(target, (size, 1))).sum(axis = 1)

	distancesorted = distances.argsort()
	for i in range(k):
		Targetlabel = trainlabel[distancesorted[i]]
	return Targetlabel

def saveResult(result):
	with open('res.csv', 'w') as f:
		csv_writer = csv.writer(f)
		csv_writer.writerow(['ImageId', 'Label'])
		for item in result:
			csv_writer.writerow(item)

def getTarget():
	targetList = []
	test_data = loadTestData()
	for i in range(len(test_data)):
		x = test_data[i]
		x = x.astype('int32')
		targetList.append(x)
	return targetList

def knn(Imageid):
	return KNN(test_data[Imageid], train_data, train_label, 1)


def main():
	global train_data, train_label, test_data
	train_data, train_label = loadTrainData()
	test_data = loadTestData()

	print('algorithm begin')
	tic = time.time()
	Imageid = range(len(test_data))
	
	pool = Pool(8)
	test_label = pool.map(knn, Imageid)
	pool.close()
	pool.join()
	print('time=', time.time() - tic)

	result = []
	print(len(test_label))
	for i in range(len(test_label)):
		result.append([i+1, test_label[i]])
	saveResult(result)

if __name__ == '__main__':
	#import profile
	#profile.run("main()", "prof.txt")
	#import pstats
	#p = pstats.Stats("prof.txt")
	#p.sort_stats("time").print_stats()
	main()


