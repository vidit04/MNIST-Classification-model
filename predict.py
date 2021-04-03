import numpy as np 
import os
import gzip
import matplotlib.pyplot as plt

from train import accuracy


data_image_test = gzip.open('t10k-images-idx3-ubyte.gz','r')
data_label_test = gzip.open('t10k-labels-idx1-ubyte.gz','r')

data_image_test.read(16)
data_label_test.read(8)

image_test = data_image_test.read(28*28*10000)
label_test = data_label_test.read(10000)

image_test = np.frombuffer(image_test, dtype=np.uint8)
label_test = np.frombuffer(label_test, dtype=np.uint8)

image_test = image_test.reshape(10000,784)
label_test = label_test.reshape(10000,1)
#label_test1 = label_test.reshape(4000:,1)

one_hot_label_test = np.zeros((10000,10), dtype = np.float32)
for i in range(10000):
    position_test = label_test[i,:]
    one_hot_label_test[i,position_test] = 1.0

image_test = image_test[4000:,:]
label_test = one_hot_label_test[4000:,:]

weights_1 = np.loadtxt('saved files/Weights_1_for_one_layer_relu_activation_normal_Normalization_xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')
baises_1 = np.loadtxt('saved files/Baises_1_for_one_layer_relu_activation_normal_Normalization_xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')
weights_2 = np.loadtxt('saved files/Weights_2_for_one_layer_relu_activation_normal_Normalization_xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')
baises_2 = np.loadtxt('saved files/Baises_2_for_one_layer_relu_activation_normal_Normalization_xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')

baises_1 = np.reshape(baises_1,(64,1))
baises_2 = np.reshape(baises_2,(10,1))
weights_3 = 0
baises_3 = 0
num=1
activation_1 ="relu"
accuracy = accuracy(weights_1,baises_1,weights_2, baises_2,weights_3,baises_3, image_test, label_test,activation_1, num)
accuracy = accuracy *100
print("Accuracy of the Test set for the defined model is ",accuracy,"%")

for i in range(len(image_test)):
    length =1
    image_arr = image_test[i,:]
    image_arr = np.reshape(image_arr,(1,784))
    labels_arr = label_test[i,:]
    labels_arr = np.reshape(labels_arr,(1,10))
    Z1 = np.zeros((64,length),dtype = np.float32)
    Z2 = np.zeros((10,length),dtype = np.float32)
    A1 = np.zeros((64,length),dtype = np.float32)
    true_array = np.zeros((length,1),dtype=np.float32)

    Z1 = np.matmul(weights_1.T,image_arr.T) + baises_1
    #print(Z1.shape)

    if activation_1 == "relu" or activation_1 == "Relu":
        #A1 = relu_activation(Z1)
        A1 = np.maximum(0,Z1)
    if activation_1 == "sigmoid" or activation_1 == "Sigmoid":
        A1 = sigmoid_activation(Z1)
    #A1 = np.maximum(0,Z1)

    Z2 = np.matmul(weights_2.T,A1) + baises_2
    #print(Z2.shape)

    Z2_max = np.max(Z2, axis=0)
    Z2_max = np.reshape(Z2_max,(1, length))
    Z2 = np.exp(Z2-Z2_max)
             
    ####################################################
    #Z1 = Z1.transpose()

    A2 = Z2/np.sum(Z2,axis=0)
    
    #total_loss = Loss_function(A2,labels_arr)

    pred_y = np.argmax(A2,axis = 0)
    pred_y = np.reshape(pred_y,(1))
    true_y = np.argmax(labels_arr, axis = 1)
    true_y = np.reshape(true_y,(1))
    image = np.reshape(image_arr,(28,28))
    #label = one_hot_label[x[i],:]
    #digit = np.argmax(label, axis =0)

    fig = plt.figure("              Pridicted label = " + str(pred_y) + "                                                                              True label = " + str(true_y))
    imgplot = plt.imshow(image)
    plt.show()

