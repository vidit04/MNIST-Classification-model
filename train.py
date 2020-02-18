import numpy as np
import os
import gzip
import cv2
import matplotlib.pyplot as plt



#os.chdir("C:\Users\user\Desktop\MNIST")
def Data_pre_processing():

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

    #image_train = image_train.reshape(60000,28,28)
    #image_train = image_train[59999,:,:]
    image_train = image_train.reshape(60000,784)
    label_train = label_train.reshape(60000,1)
    #label_train = label_train[59999]
    #print(label_train)
    #cv2.imshow('window_name', image_train)
    #cv2.waitKey(0)
    #plt.imshow(image_train)
    #plt.show()
    one_hot_label_train = np.zeros((60000,10), dtype = np.float32)
    for i in range(60000):
        position_train = label_train[i,:]
        one_hot_label_train[i,position_train] = 1.0

    #print(one_hot_label_train[59999,:])


    #image_test = image_test.reshape(10000,28,28)
    #image_test = image_test[9999,:,:]
    image_test = image_test.reshape(10000,784)
    label_test = label_test.reshape(10000,1)
    #label_test = label_test[9999]

    #print(label_test)
    #cv2.imshow('window_name', image_test)
    #cv2.waitKey(0)
    #plt.imshow(image_test)
    #plt.show()

    one_hot_label_test = np.zeros((10000,10), dtype = np.float32)
    for i in range(10000):
        position_test = label_test[i,:]
        one_hot_label_test[i,position_test] = 1.0

    #print(one_hot_label_test[9999,:])   

    image_valid1 = image_train[58000:,:]
    image_valid2 = image_test[:4000,:]
    image_valid = np.concatenate((image_valid1, image_valid2), axis=0)
    image_valid = image_valid.astype('float32')
    image_valid = image_valid/255
    #print(len(image_valid))
    image_train = image_train[:58000,:]
    image_train = image_train.astype('float32')
    image_train = image_train/255
    #print(len(image_train))
    image_test = image_test[4000:,:]
    image_test = image_test.astype('float32')
    image_test = image_test/255
    #print(len(image_test))

    one_hot_label_valid1 = one_hot_label_train[58000:,:]
    one_hot_label_valid2 = one_hot_label_test[:4000,:]
    one_hot_label_valid = np.concatenate((one_hot_label_valid1, one_hot_label_valid2), axis=0)
    #print(len(one_hot_label_valid))
    one_hot_label_train = one_hot_label_train[:58000,:]
    #print(len(one_hot_label_train))
    one_hot_label_test = one_hot_label_test[4000:,:]
    #print(len(one_hot_label_test))

    return image_train, one_hot_label_train, image_valid, one_hot_label_valid, image_test, one_hot_label_test


a,b,c,d,e,f = Data_pre_processing()
cost = []
weights = np.zeros((784,10),dtype = np.float32)
baises = np.zeros((10,1), dtype = np.float32)

dloss_dweights = np.zeros((784,10),dtype = np.float32)
dloss_dbaises  = np.zeros((10,1),dtype = np.float32)

Z1 = np.zeros((10,4),dtype = np.float32)
Z1_back = np.zeros((10,4),dtype = np.float32)
learning_rate =0.001

for i in range(784):
    for j in range(10):
        weights[i,j] = 0.01 * np.random.randn()

for i in range(10):

    n=0
    
    for j in range(1500):

        x_train = a[n:n+4,:]
        y_train = b[n:n+4,:]

        batch_size = x_train.shape[0]

        for k in range(10):
            for l in range(4):
                for m in range(784):
                    Z1[k,l] = Z1[k,l] + weights[m,k]* x_train[l,m]


        for k in range(10):
            for l in range(4):
                Z1[k,l] = Z1[k,l] + baises[k,:]


        Z1_max = np.max(Z1, axis=0)
        Z1_max = np.reshape(Z1_max,(1, batch_size))
        
        for k in range(10):
            for l in range(4):
                Z1[k,l] = np.exp(Z1[k,l]-Z1_max[:,l])

                
####################################################
        #Z1 = Z1.transpose()

        A1 = Z1/np.sum(Z1,axis=0)

##############################################
        ###### Loss
        loss = np.zeros((10,4), dtype = np.float32)
        for k in range(10):
            for l in range(4):
                loss[k,l] = - y_train[l,k]*(np.log(A1[k,l]))
        loss_per_sample = np.sum(loss,axis=1)
        total_loss = np.sum(loss_per_sample,axis= 0)
        total_loss = total_loss/batch_size

        #####  Back propogation

        for k in range(10):
            for l in range(4):
                Z1_back[k,l] = A1[k,l] - y_train[l,k]

        for k in range(784):
            for l in range(10):
                for m in range(4):
                    dloss_dweights[k,l] = dloss_dweights[k,l] + x_train[m,k]* Z1_back[l,m]

        dloss_dbaises = np.sum(Z1_back, axis =1)
        dloss_dbaises = np.reshape(dloss_dbaises,(10,1))
##########################################################
        dloss_dweights =  dloss_dweights/batch_size
        dloss_dbaises = dloss_dbaises/batch_size
#########################################################

        for k in range(784):
            for l in range(10):
                weights[k,l] = weights[k,l] - learning_rate*dloss_dweights[k,l]

        for k in range(10):
            baises[k,:] = baises[k,:] - learning_rate*dloss_dbaises[k,:]


        n= n + 4

    cost.append(total_loss)

    print("Final Cost :",total_loss, " Epoch : ", i)

x = np.arange(0,10, 1)
y = cost

plt.xlabel('Epochs')  
plt.ylabel('Training_Loss') 
 
plt.title('Loss_Graph') 
plt.plot(x, y)
plt.show()


