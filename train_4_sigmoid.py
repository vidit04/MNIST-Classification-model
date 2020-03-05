import numpy as np
import os
import gzip
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


#os.chdir("C:\Users\user\Desktop\MNIST")
#os.chdir("C:\Users\user\Desktop\MNIST")
def sigmoid_activation(array_Z_sigmoid):
    array_z_row = len(array_Z_sigmoid)
    array_z_col = len(array_Z_sigmoid[1])
    array_A_sig = np.zeros((array_z_row,array_z_col),dtype = np.float32)
    
    for k in range(array_z_row):
        for l in range(array_z_col):
            array_A_sig[k,l] = 1/(1+np.exp(-array_Z_sigmoid[k,l]))
            
    return array_A_sig

def sigmoid_activation_back(array_dloss_dA_sigmoid, array_Z_sigmoid_back):
    array_dloss_dA_row = len(array_dloss_dA_sigmoid)
    array_dloss_dA_col = len(array_dloss_dA_sigmoid[1])
    array_Z_back_sig = np.zeros((array_dloss_dA_row,array_dloss_dA_col),dtype = np.float32)
    array_Z_back_sigmoid_dummy = np.zeros((array_dloss_dA_row,array_dloss_dA_col),dtype = np.float32)

    for k in range(array_dloss_dA_row):
        for l in range(array_dloss_dA_col):
            array_Z_back_sigmoid_dummy[k,l] = 1/(1+np.exp(-array_Z_sigmoid_back[k,l]))

    for k in range(array_dloss_dA_row):
        for l in range(array_dloss_dA_col):
            array_Z_back_sig[k,l] = array_dloss_dA_sigmoid[k,l]* array_Z_back_sigmoid_dummy[k,l]*(1-array_Z_back_sigmoid_dummy[k,l])

    return array_Z_back_sig


def relu_activation(array_Z):
    array_z_row = len(array_Z)
    array_z_col = len(array_Z[1])
    array_A = np.zeros((array_z_row,array_z_col),dtype = np.float32)
    
    for k in range(array_z_row):
        for l in range(array_z_col):
            if array_Z[k,l] <=0:
                array_A[k,l] = 0
            if array_Z[k,l] > 0:
                array_A[k,l] = array_Z[k,l]

    return array_A

def relu_activation_back(array_dloss_dA, array_Z_relu_back):
    array_dloss_dA_row = len(array_dloss_dA)
    array_dloss_dA_col = len(array_dloss_dA[1])
    array_Z_back = np.zeros((array_dloss_dA_row,array_dloss_dA_col),dtype = np.float32)
    for k in range(array_dloss_dA_row):
        for l in range(array_dloss_dA_col):
            if array_Z_relu_back[k,l] <=0:
                array_Z_back[k,l] = 0
            if array_Z_relu_back[k,l] > 0:
                array_Z_back[k,l] = array_dloss_dA[k,l]

    return array_Z_back
    
def Loss_function(pred_y,true_y):
    ##############################################
    ###### Loss
    length_loss = len(pred_y[1])
    loss = np.zeros((10,length_loss), dtype = np.float32)
    for k in range(10):
        for l in range(length_loss):
            loss[k,l] = - true_y[l,k]*(np.log(pred_y[k,l]))
    loss_per_sample = np.sum(loss,axis=1)
    total_loss = np.sum(loss_per_sample,axis= 0)
    total_loss = total_loss/length_loss
    return total_loss

    
def accuracy(weights_1,weights_2,baises_1, baises_2,weights_3,baises_3, image_arr, labels_arr ):


    length = len(image_arr)
    true_array = np.zeros((length,1),dtype=np.float32)

    Z1 = np.matmul(weights_1.T,image_arr.T) + baises_1

    A1 = np.maximum(0,Z1)

    Z2 = np.matmul(weights_2.T,A1) + baises_2

    A2 = np.maximum(0,Z2)
    #A2= Z2
    Z3 = np.matmul(weights_3.T,A2) + baises_3

    Z3_max = np.max(Z3, axis=0)
    Z3_max = np.reshape(Z3_max,(1, length))
    Z3 = np.exp(Z3-Z3_max)

                
    ####################################################
    #Z1 = Z1.transpose()

    A3 = Z3/np.sum(Z3,axis=0)

    total_loss = Loss_function(A3,labels_arr)

    pred_y = np.argmax(A3,axis = 0)
    pred_y = np.reshape(pred_y,(length,1))
    true_y = np.argmax(labels_arr, axis = 1)
    true_y = np.reshape(true_y,(length,1))

    s = 0
    #true_array = (pred_y==true_y).all()
    #print(true_array.shape)

    for r in range(length):
        if pred_y[r,:] == true_y[r,:]:
            true_array[r,:] = 1
            s = s + 1

    acc  = s/length

    return acc,total_loss
    #true_array = (pred_y==true_y).all()
    #s = np.count_nonzero(true_array)
    #acc = s/length 
    #return acc

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
    total_image  = np.concatenate((image_train, image_test), axis=0)
    total_labels = np.concatenate((one_hot_label_train,one_hot_label_test), axis=0)

    perm = np.arange(total_image.shape[0])
    np.random.shuffle(perm)
    total_image = total_image[perm]
    total_labels = total_labels[perm]

    image_train = total_image[:60000,:]
    image_test = total_image[60000:,:]
    one_hot_label_train = total_labels[:60000,:]
    one_hot_label_test = total_labels[60000:,:]

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
hadden_layer_2 = 16

a,b,c,d,e,f = Data_pre_processing()
cost_train = []
cost_valid = []
cost_test = []
acc_training = []
acc_validation = []
acc_test = []
weights_1 = np.zeros((784,64),dtype = np.float32)
baises_1 = np.zeros((64,1), dtype = np.float32)

weights_2 = np.zeros((64,64),dtype = np.float32)
baises_2 = np.zeros((64,1), dtype = np.float32)

weights_3 = np.zeros((64,10),dtype = np.float32)
baises_3 = np.zeros((10,1), dtype = np.float32)

dloss_dweights_1 = np.zeros((784,64),dtype = np.float32)
dloss_dbaises_1  = np.zeros((64,1),dtype = np.float32)

dloss_dweights_2 = np.zeros((64,64),dtype = np.float32)
dloss_dbaises_2  = np.zeros((64,1),dtype = np.float32)

dloss_dweights_3 = np.zeros((64,10),dtype = np.float32)
dloss_dbaises_3  = np.zeros((10,1),dtype = np.float32)

dloss_dA1 = np.zeros((64,4),dtype = np.float32)
dloss_dA2 = np.zeros((64,4),dtype = np.float32)

Z1 = np.zeros((64,4),dtype = np.float32)
Z2 = np.zeros((64,4),dtype = np.float32)
Z3 = np.zeros((10,4),dtype = np.float32)

Z1_back = np.zeros((64,4),dtype = np.float32)
Z2_back = np.zeros((64,4),dtype = np.float32)
Z3_back = np.zeros((10,4),dtype = np.float32)

learning_rate = 1

for i in range(784):
    for j in range(64):
        weights_1[i,j] = 0.01 * np.random.randn()

for i in range(64):
    for j in range(64):
        weights_2[i,j] = 0.01 * np.random.randn()

for i in range(64):
    for j in range(10):
        weights_3[i,j] = 0.01 * np.random.randn()

#for i in tqdm(range(10), total=10 ,desc = "First Loop", unit='Epochs', unit_scale=True):
for i in range(10):
    n=0
    
    for j in tqdm(range(14500), total=14500 ,desc = "Second Loop", unit='Iterations', unit_scale=True):

        
        x_train = a[n:n+batch_size,:]
        y_train = b[n:n+batch_size,:]

        dloss_dweights_1 = np.zeros((784,64),dtype = np.float32)
        dloss_dbaises_1  = np.zeros((64,1),dtype = np.float32)

        dloss_dweights_2 = np.zeros((64,64),dtype = np.float32)
        dloss_dbaises_2  = np.zeros((64,1),dtype = np.float32)

        dloss_dweights_3 = np.zeros((64,10),dtype = np.float32)
        dloss_dbaises_3  = np.zeros((10,1),dtype = np.float32)

        dloss_dA1 = np.zeros((64,4),dtype = np.float32)
        dloss_dA2 = np.zeros((64,4),dtype = np.float32)

        Z1 = np.zeros((64,4),dtype = np.float32)
        Z2 = np.zeros((64,4),dtype = np.float32)
        Z3 = np.zeros((10,4),dtype = np.float32)

        Z1_back = np.zeros((64,4),dtype = np.float32)
        Z2_back = np.zeros((64,4),dtype = np.float32)
        Z3_back = np.zeros((10,4),dtype = np.float32)


        Z1 = np.matmul(weights_1.T,x_train.T) + baises_1

        #A1 = np.maximum(0,Z1)
        #A1 = relu_activation(Z1)
        A1 = sigmoid_activation(Z1)

        Z2 = np.matmul(weights_2.T,A1) + baises_2

        #A2 = np.maximum(0,Z2)
        #A2 = relu_activation(Z2)
        A2 = sigmoid_activation(Z2)

        #A2= Z2
        Z3 = np.matmul(weights_3.T,A2) + baises_3

        Z3_max = np.max(Z3, axis=0)
        Z3_max = np.reshape(Z3_max,(1, 4))
        Z3 = np.exp(Z3-Z3_max)

                
        ####################################################
        #Z1 = Z1.transpose()

        A3 = Z3/np.sum(Z3,axis=0)

        ##############################################
        ###### Loss
        #loss = np.zeros((10,4), dtype = np.float32)
        #for k in range(10):
        #    for l in range(4):
        #        loss[k,l] = - y_train[l,k]*(np.log(A3[k,l]))
        #loss_per_sample = np.sum(loss,axis=1)
        #total_loss = np.sum(loss_per_sample,axis= 0)
        #total_loss = total_loss/batch_size

        #####  Back propogation

        Z3_back = A3 - y_train.T
        dloss_dweights_3 = (1./batch_size) * np.matmul(A2,Z3_back.T)
        dloss_dbaises_3 = (1./batch_size) * np.sum(Z3_back, axis = 1, keepdims= True)
        
        dloss_dA2 = np.matmul(weights_3, Z3_back)


        #Z1_back = dloss_dA1

        ####### Relu Backward

        #for k in range(64):
        #    for l in range(4):
        #        if Z2[k,l] <=0:
        #            Z2_back[k,l] = 0
        #        if Z2[k,l] > 0:
        #            Z2_back[k,l] = dloss_dA2[k,l]

        #Z2_back = relu_activation_back(dloss_dA2, Z2)
        Z2_back = sigmoid_activation_back(dloss_dA2, Z2)

        dloss_dweights_2 = (1./batch_size) * np.matmul(A1,Z2_back.T)
        dloss_dbaises_2 = (1./batch_size) * np.sum(Z2_back, axis = 1, keepdims= True)

        dloss_dA1 = np.matmul(weights_2,Z2_back)

        

        #Z1_back = dloss_dA1

        ####### Relu Backward

        #for k in range(64):
        #    for l in range(4):
        #        if Z1[k,l] <=0:
        #            Z1_back[k,l] = 0
        #        if Z1[k,l] > 0:
        #            Z1_back[k,l] = dloss_dA1[k,l]

        #Z1_back = relu_activation_back(dloss_dA1, Z1)
        Z1_back = sigmoid_activation_back(dloss_dA1, Z1)

        ##################################################################################
        dloss_dweights_1 = (1./batch_size) * np.matmul(x_train.T,Z1_back.T)
        dloss_dbaises_1 = (1./batch_size) * np.sum(Z1_back, axis = 1, keepdims= True)




        weights_1= weights_1- learning_rate*dloss_dweights_1
        baises_1 = baises_1 - learning_rate*dloss_dbaises_1
        
        weights_2= weights_2- learning_rate*dloss_dweights_2
        baises_2 = baises_2 - learning_rate*dloss_dbaises_2

        weights_3= weights_3- learning_rate*dloss_dweights_3
        baises_3 = baises_3 - learning_rate*dloss_dbaises_3



        n = n + 4

    acc_epoch_train, loss_train = accuracy(weights_1,weights_2,baises_1, baises_2,weights_3,baises_3 , a, b )
    acc_epoch_valid, loss_valid = accuracy(weights_1,weights_2,baises_1, baises_2 ,weights_3,baises_3, c, d )
    acc_epoch_test, loss_test = accuracy(weights_1,weights_2,baises_1, baises_2 ,weights_3,baises_3, e, f )
    cost_train.append(loss_train)
    cost_valid.append(loss_valid)
    cost_test.append(loss_test)
    acc_training.append(acc_epoch_train)
    acc_validation.append(acc_epoch_valid)
    acc_test.append(acc_epoch_test)

    print("Final Cost :",loss_train, " Epoch : ", i)
    print("Final Accuracy :",acc_epoch_train, " Epoch : ", i)   
            


x = np.arange(0,10, 1)
y = cost_train
y1 = cost_valid
y2 = cost_test

plt.xlabel('Epochs')  
plt.ylabel('Loss') 
 
plt.title('Training_Loss_Graph') 
plt.plot(x, y,"b", x, y1,"g", x, y2,"r")
plt.show()

z= acc_training
z1 = acc_validation
z2 = acc_test
plt.xlabel('Epochs')  
plt.ylabel('Accuracy')

plt.title('Training_Acc_Graph') 
plt.plot(x, z,"b", x, z1,"g", x, z2,"r")
plt.show()



