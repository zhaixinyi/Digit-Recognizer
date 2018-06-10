import os
import csv
import json
import math
import random
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import time
from IPython import embed
from sklearn import svm
from sklearn.externals import joblib
from sklearn.decomposition import PCA

train_data = []
train_label = []
val_data = []
val_label = []
test_data = []
test_label = []

N = 28

def loadTrainData():
  l = []
  result = []
  if not os.path.exists("data.npz"):
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
      data = data / 255.0
      # print('data:', data.shape) # 42000 * 784 <del>28 * 27</del>
      for i in tqdm(range(len(data))):
        # imshow(data[i].reshape(28, 28))
        img1 = CutPicture(data[i].reshape(28, 28))
        # imshow(img1)
        img = StretchPicture(img1)
        # imshow(img)
        img = img.reshape(-1)
        result.append(img)
      res = np.array(result)
      # return res, label
    np.savez("data.npz", res, label)

  r = np.load("data.npz")
  # print(r["arr_0"].shape, r["arr_1"].shape)
  return r["arr_0"], r["arr_1"]


def loadTestData():
  l = []
  res = []
  if not os.path.exists("test.npz"):
    with open('test.csv', 'r') as f:
      lines = csv.reader(f)
      for line in lines:
        l.append(line)
      l.remove(l[0])
      data = np.array(l)
      data = np.int32(data)
      for i in tqdm(range(len(data))):
        img2 = CutPicture(data[i].reshape(28, 28))
        img = StretchPicture(img2)
        # imshow(img)
        img = img.reshape(-1)
        res.append(img)
      res_test = np.int32(res)
    np.savez("test.npz", res_test)
 
  r = np.load("test.npz")
  # print(res_test.shape)
  return r["arr_0"]




#切割图象
def CutPicture(img):
  x1, y1, x2, y2 = JudgeEdge(img)
  return img[x1 : x2 + 1, y1 : y2 + 1]

def JudgeEdge(img):
  x1, y1, x2, y2 = img.shape[0], img.shape[1], 0, 0
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      if img[i, j] > 0:
        x1 = min(x1, i)
        y1 = min(y1, j)
        x2 = max(x2, i)
        y2 = max(y2, j)
  return x1, y1, x2, y2

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



