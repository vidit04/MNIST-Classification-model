import numpy as np
import os
import gzip
import cv2
import matplotlib.pyplot as plt


#os.chdir("C:\Users\user\Desktop\MNIST")

data_image_train = gzip.open('train-images-idx3-ubyte.gz','r')
data_label_train = gzip.open('train-labels-idx1-ubyte.gz','r')
         

data_image_test = gzip.open('t10k-images-idx3-ubyte.gz','r')
data_label_test = gzip.open('t10k-labels-idx1-ubyte.gz','r')

data_image_train.read(16)
data_label_train.read(8)

data_image_test.read(16)
data_label_test.read(8)

image_train = data_image_train.read(28*28*60000)
label_train = data_label_train.read(60000)

image_test = data_image_test.read(28*28*10000)
label_test = data_label_test.read(10000)

image_train = np.frombuffer(image_train, dtype=np.uint8)
label_train = np.frombuffer(label_train, dtype=np.uint8)

image_test = np.frombuffer(image_test, dtype=np.uint8)
label_test = np.frombuffer(label_test, dtype=np.uint8)

image_train = image_train.reshape(60000,28,28)
image_train = image_train[59999,:,:]
label_train = label_train.reshape(60000)
label_train = label_train[59999]
print(label_train)
#cv2.imshow('window_name', image_train)
#cv2.waitKey(0)
plt.imshow(image_train)
plt.show()


image_test = image_test.reshape(10000,28,28)
image_test = image_test[9999,:,:]
label_test = label_test.reshape(10000)
label_test = label_test[9999]
print(label_test)
#cv2.imshow('window_name', image_test)
#cv2.waitKey(0)
plt.imshow(image_test)
plt.show()


