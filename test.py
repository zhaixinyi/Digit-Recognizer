import os
import csv
import math
import random
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import time
from IPython import embed
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.externals import joblib
# from loadData import *

train_data = []
train_label = []
val_data = []
val_label = []
test_data = []
test_label = []

N = 28


def imshow(img):
  print('image size:', img.shape)
  assert len(img.shape) == 2
  print('-' * img.shape[1])
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      if img[i, j] > 0:
        print('x', end='')
      else:
        print(' ', end='')
    print('')
  print('-' * img.shape[1])

def loadTrainData():
  l = []
  result = []
  with open('/home/shaoshuai/zxy/mnist/knn_cut/train.csv', 'r') as f:
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

    #data = np.mat(data)
    #label = np.mat(label)
    return res, label

def loadTestData():
  l = []
  res = []
  with open('/home/shaoshuai/zxy/mnist/knn_cut/test.csv', 'r') as f:
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
    res_test = np.array(res)

    return res_test

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

def SVM(target, traindata, trainlabel):
  svc = svm.SVC(kernel = 'rbf', C = 10)
  svc.fit(traindata, trainlabel)
  pre = svc.predict(target)
  return pre

def saveResult(result):
  with open('/home/shaoshuai/zxy/svm/svm.csv', 'w') as f:
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

def argmax(L):
  maxindex = 0
  i = 0
  for tmp in L:
    if tmp > L[maxindex]:
      maxindex = i
    i += 1

  return maxindex


def main():
  global train_data, train_label, test_data
  train_data, train_label = loadTrainData()
  test_data = loadTestData()
  pca = PCA(n_components = 0.8, whiten = True)
  train_x = pca.fit_transform(train_data)
  test_x = pca.transform(test_data)

  print('algorithm begin')
  tic = time.time()
  Imageid = range(len(test_data))

  if not os.path.exists("/home/shaoshuai/zxy/svm/train_model.m"):
    svc = svm.SVC(kernel = 'rbf', C = 10)
    svc.fit(train_x, train_label)
    joblib.dump(svc, "train_model.m")

  if not os.path.exists("/home/shaoshuai/zxy/svm/train_svm.m"):
    model = svm.SVC(C = 11, gamma = 0.025, kernel = 'rbf')
    model.fit(train_x, train_label)
    joblib.dump(model, "train_svm.m")

  if not os.path.exists("/home/shaoshuai/zxy/svm/svm.m"):
    pattern = svm.SVC(C = 10.5, gamma = 0.04, kernel = 'rbf')
    pattern.fit(train_x, train_label)
    joblib.dump(pattern, "svm.m")

  if not os.path.exists("/home/shaoshuai/zxy/svm/svm1.m"):
    clf1 = svm.SVC(C = 12, gamma = 0.045, kernel = 'rbf')
    clf1.fit(train_x, train_label)
    joblib.dump(clf1, "svm1.m")

  if not os.path.exists("/home/shaoshuai/zxy/svm/svm2.m"):
    clf2 = svm.SVC(C = 8, gamma = 0.01, kernel = 'rbf')
    clf2.fit(train_x, train_label)
    joblib.dump(clf2, "svm2.m")

  if not os.path.exists("/home/shaoshuai/zxy/svm/svm3.m"):
    clf3 = svm.SVC(C = 10, gamma = 0.03, kernel = 'rbf')
    clf3.fit(train_x, train_label)
    joblib.dump(clf3, "svm3.m")

  if not os.path.exists("/home/shaoshuai/zxy/svm/svm4.m"):
    clf4 = svm.SVC(C = 6, gamma = 0.02, kernel = 'rbf')
    clf4.fit(train_x, train_label)
    joblib.dump(clf4, "svm4.m")

  if not os.path.exists("/home/shaoshuai/zxy/svm/svm5.m"):
    clf5 = svm.SVC(C = 10, gamma = 0.025, kernel = 'rbf')
    clf5.fit(train_x, train_label)
    joblib.dump(clf5, "svm5.m")

  if not os.path.exists("/home/shaoshuai/zxy/svm/svm6.m"):
    clf6 = svm.SVC(C = 10.5, gamma = 0.035, kernel = 'rbf')
    clf6.fit(train_x, train_label)
    joblib.dump(clf6, "svm6.m")

  if not os.path.exists("//home/shaoshuai/zxy/svm/svm7.m"):
    clf7 = svm.SVC(C = 4, gamma = 0.01, kernel = 'rbf')
    clf7.fit(train_x, train_label)
    joblib.dump(clf7, "svm7.m")

  if not os.path.exists("/home/shaoshuai/zxy/svm/svm8.m"):
    clf8 = svm.SVC(C = 8, gamma = 0.03, kernel = 'rbf')
    clf8.fit(train_x, train_label)
    joblib.dump(clf8, "svm8.m")

  if not os.path.exists("/home/shaoshuai/zxy/svm/svm9.m"):
    clf9 = svm.SVC(C = 2, gamma = 0.1, kernel = 'rbf')
    clf9.fit(train_x, train_label)
    joblib.dump(clf9, "svm9.m")

  if not os.path.exists("/home/shaoshuai/zxy/svm/svm10.m"):
    clf10 = svm.SVC(C = 4, gamma = 0.025, kernel = 'rbf')
    clf10.fit(train_x, train_label)
    joblib.dump(clf10, "svm10.m")

  if not os.path.exists("/home/shaoshuai/zxy/svm/svm11.m"):
    clf11 = svm.SVC(C = 6, gamma = 0.015, kernel = 'rbf')
    clf11.fit(train_x, train_label)
    joblib.dump(clf11, "svm11.m")

  if not os.path.exists("/home/shaoshuai/zxy/svm/svm12.m"):
    clf12 = svm.SVC(C = 8, gamma = 0.06, kernel = 'rbf')
    clf12.fit(train_x, train_label)
    joblib.dump(clf12, "svm12.m")

  if not os.path.exists("/home/shaoshuai/zxy/svm/svm13.m"):
    clf13 = svm.SVC(C = 3, gamma = 0.1, kernel = 'rbf')
    clf13.fit(train_x, train_label)
    joblib.dump(clf13, "svm13.m")

  if not os.path.exists("/home/shaoshuai/zxy/svm/svm14.m"):
    clf14 = svm.SVC(C = 7, gamma = 0.03, kernel = 'rbf')
    clf14.fit(train_x, train_label)
    joblib.dump(clf14, "svm14.m")

  if not os.path.exists("/home/shaoshuai/zxy/svm/svm15.m"):
    clf15 = svm.SVC(C = 9, gamma = 0.035, kernel = 'rbf')
    clf15.fit(train_x, train_label)
    joblib.dump(clf15, "svm15.m")

  if not os.path.exists("/home/shaoshuai/zxy/svm/svm16.m"):
    clf16 = svm.SVC(C = 11, gamma = 0.04, kernel = 'rbf')
    clf16.fit(train_x, train_label)
    joblib.dump(clf16, "svm16.m")

  if not os.path.exists("/home/shaoshuai/zxy/svm/svm17.m"):
    clf17 = svm.SVC(C = 5, gamma = 0.1, kernel = 'rbf')
    clf17.fit(train_x, train_label)
    joblib.dump(clf17, "svm17.m")

  svc = joblib.load("/home/shaoshuai/zxy/svm/train_model.m")
  model = joblib.load("/home/shaoshuai/zxy/svm/train_svm.m")
  pattern = joblib.load("/home/shaoshuai/zxy/svm/svm.m")
  clf1 = joblib.load("/home/shaoshuai/zxy/svm/svm1.m")
  clf2 = joblib.load("/home/shaoshuai/zxy/svm/svm2.m")
  clf3 = joblib.load("/home/shaoshuai/zxy/svm/svm3.m")
  clf4 = joblib.load("/home/shaoshuai/zxy/svm/svm4.m")
  clf5 = joblib.load("/home/shaoshuai/zxy/svm/svm5.m")
  clf6 = joblib.load("/home/shaoshuai/zxy/svm/svm6.m")
  clf7 = joblib.load("/home/shaoshuai/zxy/svm/svm7.m")
  clf8 = joblib.load("/home/shaoshuai/zxy/svm/svm8.m")
  clf9 = joblib.load("/home/shaoshuai/zxy/svm/svm9.m")
  clf10 = joblib.load("/home/shaoshuai/zxy/svm/svm10.m")
  clf11 = joblib.load("/home/shaoshuai/zxy/svm/svm11.m")
  clf12 = joblib.load("/home/shaoshuai/zxy/svm/svm12.m")
  clf13 = joblib.load("/home/shaoshuai/zxy/svm/svm13.m")
  clf14 = joblib.load("/home/shaoshuai/zxy/svm/svm14.m")
  clf15 = joblib.load("/home/shaoshuai/zxy/svm/svm15.m")
  clf16 = joblib.load("/home/shaoshuai/zxy/svm/svm16.m")
  clf17 = joblib.load("/home/shaoshuai/zxy/svm/svm17.m")

  result = []
  # svc = svm.SVC(kernel = 'rbf', C = 10)
  # svc.fit(train_x, train_label)
  for i in tqdm(range(len(test_x))):
    L = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    pre = svc.predict([test_x[i]])[0]
    ret = model.predict([test_x[i]])[0]
    res = pattern.predict([test_x[i]])[0]
    res1 = clf1.predict([test_x[i]])[0]
    res2 = clf2.predict([test_x[i]])[0]
    res3 = clf3.predict([test_x[i]])[0]
    res4 = clf4.predict([test_x[i]])[0]
    res5 = clf5.predict([test_x[i]])[0]
    res6 = clf6.predict([test_x[i]])[0]
    res7 = clf7.predict([test_x[i]])[0]
    res8 = clf8.predict([test_x[i]])[0]
    res9 = clf9.predict([test_x[i]])[0]
    res10 = clf10.predict([test_x[i]])[0]
    res11 = clf11.predict([test_x[i]])[0]
    res12 = clf12.predict([test_x[i]])[0]
    res13 = clf13.predict([test_x[i]])[0]
    res14 = clf14.predict([test_x[i]])[0]
    res15 = clf15.predict([test_x[i]])[0]
    res16 = clf16.predict([test_x[i]])[0]
    res17 = clf17.predict([test_x[i]])[0]

    L[pre] += 1
    L[ret] += 1
    L[res] += 1
    L[res1] += 1
    L[res2] += 1
    L[res3] += 1
    L[res4] += 1
    L[res5] += 1
    L[res6] += 1
    L[res7] += 1
    L[res8] += 1
    L[res9] += 1
    L[res10] += 1
    L[res11] += 1
    L[res12] += 1
    L[res13] += 1
    L[res14] += 1
    L[res15] += 1
    L[res16] += 1
    L[res17] += 1
    pred = argmax(L)
    result.append([i+1, pred])
  saveResult(result)
  print('time=', time.time() - tic)


if __name__ == '__main__':
  #import profile
  #profile.run("main()", "prof.txt")
  #import pstats
  #p = pstats.Stats("prof.txt")
  #p.sort_stats("time").print_stats()
  main()
