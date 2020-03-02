import numpy as np
import os
import gzip
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


#os.chdir("C:\Users\user\Desktop\MNIST")
def accuracy(weights_1,weights_2,baises_1, baises_2 , image_arr, labels_arr ):



    length = len(image_arr)
    Z1 = np.zeros((64,length),dtype = np.float32)
    Z2 = np.zeros((10,length),dtype = np.float32)
    A1 = np.zeros((64,length),dtype = np.float32)
    true_array = np.zeros((length,1),dtype=np.float32)

    for k in range(64):
        for l in range(length):
            for m in range(784):
                Z1[k,l] = Z1[k,l] + weights_1[m,k]* image_arr[l,m]


    for k in range(64):
        for l in range(length):
            Z1[k,l] = Z1[k,l] + baises_1[k,:]


    ################### Relu activation
    for k in range(64):
        for l in range(length):
            if Z1[k,l] <=0:
                A1[k,l] = 0
            if Z1[k,l] > 0:
                A1[k,l] = Z1[k,l]

        #A1= Z1
        #A1 = A1.transpose()

    for k in range(10):
        for l in range(length):
            for m in range(64):
                Z2[k,l] = Z2[k,l] + weights_2[m,k]* A1[m,l]

    for k in range(10):
        for l in range(length):
            Z2[k,l] = Z2[k,l] + baises_2[k,:]

    Z2_max = np.max(Z2, axis=0)
    Z2_max = np.reshape(Z2_max,(1, length))
        
    for k in range(10):
        for l in range(length):
            Z2[k,l] = np.exp(Z2[k,l]-Z2_max[:,l])

                
    ####################################################
    #Z1 = Z1.transpose()

    A2 = Z2/np.sum(Z2,axis=0)

    pred_y = np.argmax(A2,axis = 0)
    pred_y = np.reshape(pred_y,(length,1))
    true_y = np.argmax(labels_arr, axis = 1)
    true_y = np.reshape(true_y,(length,1))

    s = 0

    for r in range(length):
        if pred_y[r,:] == true_y[r,:]:
            true_array[r,:] = True
            s = s + 1

    acc  = s/length

    return acc


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


batch_size = 4
hidden_layer_1 = 64

a,b,c,d,e,f = Data_pre_processing()
cost = []
acc = []
weights_1 = np.zeros((784,64),dtype = np.float32)
baises_1 = np.zeros((64,1), dtype = np.float32)

weights_2 = np.zeros((64,10),dtype = np.float32)
baises_2 = np.zeros((10,1), dtype = np.float32)

dloss_dweights_1 = np.zeros((784,64),dtype = np.float32)
dloss_dbaises_1  = np.zeros((64,1),dtype = np.float32)

dloss_dweights_2 = np.zeros((64,10),dtype = np.float32)
dloss_dbaises_2  = np.zeros((10,1),dtype = np.float32)

dloss_dA1 = np.zeros((64,4),dtype = np.float32)

Z1 = np.zeros((64,4),dtype = np.float32)
Z2 = np.zeros((10,4),dtype = np.float32)

Z1_back = np.zeros((64,4),dtype = np.float32)
Z2_back = np.zeros((10,4),dtype = np.float32)

learning_rate = 0.1

for i in range(784):
    for j in range(64):
        weights_1[i,j] = 0.01 * np.random.randn()

for i in range(64):
    for j in range(10):
        weights_2[i,j] = 0.01 * np.random.randn()

#for i in tqdm(range(10), total=10 ,desc = "First Loop", unit='Epochs', unit_scale=True):
for i in range(10):
    n=0
    
    for j in tqdm(range(150), total=14500 ,desc = "Second Loop", unit='Iterations', unit_scale=True):

        
        x_train = a[n:n+batch_size,:]
        y_train = b[n:n+batch_size,:]


        #batch_size = x_train.shape[0]

        dloss_dweights_1 = np.zeros((784,64),dtype = np.float32)
        dloss_dbaises_1  = np.zeros((64,1),dtype = np.float32)

        dloss_dweights_2 = np.zeros((64,10),dtype = np.float32)
        dloss_dbaises_2  = np.zeros((10,1),dtype = np.float32)

        dloss_dA1 = np.zeros((64,4),dtype = np.float32)

        Z1 = np.zeros((64,4),dtype = np.float32)
        Z2 = np.zeros((10,4),dtype = np.float32)

        A1 = np.zeros((64,4),dtype = np.float32)

        Z1_back = np.zeros((hidden_layer_1,batch_size),dtype = np.float32)
        Z2_back = np.zeros((10,batch_size),dtype = np.float32)
        
        for k in range(64):
            for l in range(4):
                for m in range(784):
                    Z1[k,l] = Z1[k,l] + weights_1[m,k]* x_train[l,m]


        for k in range(64):
            for l in range(4):
                Z1[k,l] = Z1[k,l] + baises_1[k,:]


        ################### Relu activation
        for k in range(64):
            for l in range(4):
                if Z1[k,l] <=0:
                    A1[k,l] = 0
                if Z1[k,l] > 0:
                    A1[k,l] = Z1[k,l]

        #A1= Z1
        #A1 = A1.transpose()

        for k in range(10):
            for l in range(4):
                for m in range(64):
                    Z2[k,l] = Z2[k,l] + weights_2[m,k]* A1[m,l]

        for k in range(10):
            for l in range(4):
                Z2[k,l] = Z2[k,l] + baises_2[k,:]

        Z2_max = np.max(Z2, axis=0)
        Z2_max = np.reshape(Z2_max,(1, 4))
        
        for k in range(10):
            for l in range(4):
                Z2[k,l] = np.exp(Z2[k,l]-Z2_max[:,l])

                
        ####################################################
        #Z1 = Z1.transpose()

        A2 = Z2/np.sum(Z2,axis=0)

        ##############################################
        ###### Loss
        loss = np.zeros((10,4), dtype = np.float32)
        for k in range(10):
            for l in range(4):
                loss[k,l] = - y_train[l,k]*(np.log(A2[k,l]))
        loss_per_sample = np.sum(loss,axis=1)
        total_loss = np.sum(loss_per_sample,axis= 0)
        total_loss = total_loss/batch_size

        #####  Back propogation

        for k in range(10):
            for l in range(4):
                Z2_back[k,l] = A2[k,l] - y_train[l,k]

        for k in range(64):
            for l in range(10):
                for m in range(4):
                    dloss_dweights_2[k,l] = dloss_dweights_2[k,l] + A1[k,m]* Z2_back[l,m]

        dloss_dbaises_2 = np.sum(Z2_back, axis = 1)
        dloss_dbaises_2 = np.reshape(dloss_dbaises_2,(10,1))
        ##########################################################
        dloss_dweights_2 =  dloss_dweights_2/batch_size
        dloss_dbaises_2 = dloss_dbaises_2/batch_size
        #########################################################

        
        for k in range(64):
            for l in range(4):
                for m in range(10):
                    dloss_dA1[k,l] = dloss_dA1[k,l] + weights_2[k,m]* Z2_back[m,l]

        #Z1_back = dloss_dA1

        ####### Relu Backward

        for k in range(64):
            for l in range(4):
                if Z1[k,l] <=0:
                    Z1_back[k,l] = 0
                if Z1[k,l] > 0:
                    Z1_back[k,l] = dloss_dA1[k,l]

        for k in range(784):
            for l in range(64):
                for m in range(4):
                    dloss_dweights_1[k,l] = dloss_dweights_1[k,l] + x_train[m,k]* Z1_back[l,m]

        dloss_dbaises_1 = np.sum(Z1_back, axis =1)
        dloss_dbaises_1 = np.reshape(dloss_dbaises_1,(64,1))
        ##########################################################
        dloss_dweights_1 =  dloss_dweights_1/batch_size
        dloss_dbaises_1 = dloss_dbaises_1/batch_size
        #########################################################

        for k in range(784):
            for l in range(64):
                weights_1[k,l] = weights_1[k,l] - learning_rate*dloss_dweights_1[k,l]

        for k in range(64):
            baises_1[k,:] = baises_1[k,:] - learning_rate*dloss_dbaises_1[k,:]


        for k in range(64):
            for l in range(10):
                weights_2[k,l] = weights_2[k,l] - learning_rate*dloss_dweights_2[k,l]

        for k in range(10):
            baises_2[k,:] = baises_2[k,:] - learning_rate*dloss_dbaises_2[k,:]


        n = n + 4
        
    acc_epoch = accuracy(weights_1,weights_2,baises_1, baises_2 , c, d )
    cost.append(total_loss)
    acc.append(acc_epoch)

    print("Final Cost :",total_loss, " Epoch : ", i)
    print("Final Accuracy :",acc_epoch, " Epoch : ", i)   
            


x = np.arange(0,10, 1)
y = cost

plt.xlabel('Epochs')  
plt.ylabel('Loss') 
 
plt.title('Training_Loss_Graph') 
plt.plot(x, y)
plt.show()

z= acc
plt.xlabel('Epochs')  
plt.ylabel('Accuracy')

plt.title('Training_Acc_Graph') 
plt.plot(x, z)
plt.show()



