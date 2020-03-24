import numpy as np 
import os
import gzip
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv




def Data_pre_processing(Normal):

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

    if (Normal == "simple" or Normal == "Simple"):
        image_valid = Simple_normalization(image_valid)
        print("I am in simple")
    if (Normal == "normal" or Normal == "Normal"):
        image_valid = Normal_normalization(image_valid)
        print("I am in normal")

    #print(len(image_valid))
    image_train = image_train[:58000,:]
    image_train = image_train.astype('float32')
    
    if (Normal == "simple" or Normal == "Simple"):
        image_train = Simple_normalization(image_train)
        print("I am in simple")
    if (Normal == "normal" or Normal == "Normal"):
        image_train = Normal_normalization(image_train)
        print("I am in normal")

    #print(len(image_train))
    image_test = image_test[4000:,:]
    image_test = image_test.astype('float32')

    if (Normal == "simple" or Normal == "Simple"):
        image_test = Simple_normalization(image_test)
        print("I am in simple")
    if (Normal == "normal" or Normal == "Normal"):
        image_test = Normal_normalization(image_test)
        print("I am in normal")

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

def Normal_normalization(image_array):
    mean  = np.mean(image_array)
    #print(mean)
    std = np.std(image_array, ddof = 1)
    #print(std)
    image_array = (image_array - mean)/std
    return image_array

def Simple_normalization(image_array):
    image_array = image_array / 255
    return image_array

def Gaussian_initialization(weights):
    row_weights = weights.shape[0]
    width_weights = weights.shape[1]
    weights = 0.01 * np.random.randn(row_weights,width_weights)

    ## only uncomment during testing
    #weights = 0.01 * weights
    return weights

def Xavier_initialization(weights):
    row_weights = weights.shape[0]
    width_weights = weights.shape[1]
    weights = np.random.randn(row_weights,width_weights) * np.sqrt(1./row_weights)
    
    ## only uncomment during testing
    #weights = weights * np.sqrt(1./row_weights)
    return weights

def learning_rate_decay(learning_rate, decay_rate):
    learning_rate = learning_rate * decay_rate
    return learning_rate

def save_fun(save_array, ext):
    np.savetxt( ext + '.csv', save_array, delimiter=',')
    #return None

def save_fun_list(save_list, ext):
    with open( ext + ".csv", 'w',newline = "") as f:
         riter= csv.writer(f)
         riter.writerow(save_list)
    #return None

def dropout_forward(array_Act, prob):

    mask = np.zeros((len(array_Act),len(array_Act[1])),dtype = np.float32)
    mask = np.random.rand(len(mask),len(mask[1]))
    
    # only uncomment for testing
    #mask = np.ones(len(mask),len(mask[1]),dtype = np.float32) 
    mask = mask < prob
    array_Act = np.multiply(array_Act, mask)
    array_Act = array_Act/prob
    #print("I am in dropout")
    return array_Act

def Momentum_optimizer(weights_1,baises_1,weights_2,baises_2,weights_3,baises_3,mov_weights_1,mov_baises_1,mov_weights_2,mov_baises_2,mov_weights_3,mov_baises_3,dloss_dweights_1,dloss_dbaises_1,dloss_dweights_2,dloss_dbaises_2,dloss_dweights_3, dloss_dbaises_3, learning_rate, beta, num):

    #print(dloss_dbaises_3)
    #print(mov_baises_3)
    if num ==2:
        mov_weights_1 = (beta*mov_weights_1) + ((1. - beta)* dloss_dweights_1)
        mov_baises_1 = (beta*mov_baises_1) + ((1. - beta)* dloss_dbaises_1)

        mov_weights_2 = (beta*mov_weights_2) + ((1. - beta)* dloss_dweights_2)
        mov_baises_2 = (beta*mov_baises_2) + ((1. - beta)* dloss_dbaises_2)


        mov_weights_3 = (beta*mov_weights_3) + ((1. - beta)* dloss_dweights_3)
        mov_baises_3 = (beta*mov_baises_3) + ((1. - beta)* dloss_dbaises_3)

        #print(baises_3)
        #print(mov_baises_3)

        weights_1 = weights_1- learning_rate*mov_weights_1
        baises_1 = baises_1 - learning_rate*mov_baises_1
        
        weights_2= weights_2- learning_rate*mov_weights_2
        baises_2 = baises_2 - learning_rate*mov_baises_2

        weights_3= weights_3- learning_rate*mov_weights_3
        baises_3 = baises_3 - learning_rate*mov_baises_3

        return weights_1,baises_1,weights_2,baises_2,weights_3,baises_3,mov_weights_1,mov_baises_1,mov_weights_2,mov_baises_2,mov_weights_3,mov_baises_3

    if num ==1:
        mov_weights_1 = beta*mov_weights_1 + (1. - beta)* dloss_dweights_1
        mov_baises_1 = beta*mov_baises_1 + (1. - beta)* dloss_dbaises_1

        mov_weights_2 = beta*mov_weights_2 + (1. - beta)* dloss_dweights_2
        mov_baises_2 = beta*mov_baises_2+(1. - beta)* dloss_dbaises_2
    
        weights_1 = weights_1- learning_rate*mov_weights_1
        baises_1 = baises_1 - learning_rate*mov_baises_1
        
        weights_2= weights_2- learning_rate*mov_weights_2
        baises_2 = baises_2 - learning_rate*mov_baises_2

        return weights_1,baises_1,weights_2,baises_2,mov_weights_1,mov_baises_1,mov_weights_2,mov_baises_2

    if num ==0:
        mov_weights_1 = beta*mov_weights_1 + (1. - beta)* dloss_dweights_1
        mov_baises_1 = beta*mov_baises_1 + (1. - beta)* dloss_dbaises_1
    
        weights_1 = weights_1- learning_rate*mov_weights_1
        baises_1 = baises_1 - learning_rate*mov_baises_1
        return weights_1,baises_1,mov_weights_1,mov_baises_1


def SGD_optimizer(weights_1,baises_1,weights_2,baises_2,weights_3,baises_3,dloss_dweights_1,dloss_dbaises_1,dloss_dweights_2,dloss_dbaises_2,dloss_dweights_3, dloss_dbaises_3, learning_rate,num):

    if num ==2:    
        #print(baises_3)
        #print(dloss_dbaises_3)
        weights_1= weights_1- (learning_rate*dloss_dweights_1)
        baises_1 = baises_1 - (learning_rate*dloss_dbaises_1)
        
        weights_2= weights_2- (learning_rate*dloss_dweights_2)
        baises_2 = baises_2 - (learning_rate*dloss_dbaises_2)

        weights_3= weights_3- (learning_rate*dloss_dweights_3)
        baises_3 = baises_3 - (learning_rate*dloss_dbaises_3)

        return weights_1,baises_1,weights_2,baises_2,weights_3,baises_3

    if num ==1:
        weights_1= weights_1- learning_rate*dloss_dweights_1
        baises_1 = baises_1 - learning_rate*dloss_dbaises_1
        
        weights_2= weights_2- learning_rate*dloss_dweights_2
        baises_2 = baises_2 - learning_rate*dloss_dbaises_2

        return weights_1,baises_1,weights_2,baises_2

    if num ==0:
        weights_1= weights_1- learning_rate*dloss_dweights_1
        baises_1 = baises_1 - learning_rate*dloss_dbaises_1
        return weights_1,baises_1


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


def reg_loss(weights_1,weights_2,weights_3, alpha,num):

    if num == 2:
        reg_loss_2 = (0.5 * alpha * np.sum(weights_1*weights_1)) + (0.5 * alpha * np.sum(weights_2*weights_2)) + (0.5 * alpha * np.sum(weights_3 * weights_3))
        return reg_loss_2
    if num == 1:
        reg_loss_1 = (0.5 * alpha * np.sum(weights_1*weights_1)) + (0.5 * alpha * np.sum(weights_2*weights_2))
        return reg_loss_1
    if num == 0:
        reg_loss_0 = 0.5 * alpha * np.sum(weights_1*weights_1)
        return reg_loss_0
    
def Loss_function(pred_y,true_y,weights_1,weights_2,weights_3 ,alpha, reg , num):
    ##############################################
    ###### Loss
    length_loss = len(pred_y[1])
    eps=1e-12
    # Comment the next line for testing
    pred_y = np.clip(pred_y, eps, 1. - eps)
    loss = np.zeros((10,length_loss), dtype = np.float32)
    for k in range(10):
        for l in range(length_loss):
            loss[k,l] = - true_y[l,k]*(np.log(pred_y[k,l]))
    #print(loss)
    loss_per_sample = np.sum(loss,axis=1)
    total_loss = np.sum(loss_per_sample,axis= 0)
    total_loss = total_loss/length_loss

    if (reg == "Yes" or reg == "yes" or reg == "y" or reg == "Y" or reg == "YES"):
        #print("Yes, I am in regression")

        reg_loss_ = reg_loss(weights_1,weights_2,weights_3, alpha,num)
        #print(reg_loss)
        total_loss = total_loss + reg_loss_

    return total_loss

def forward_prop_for_loss(weights_1,baises_1,weights_2, baises_2,weights_3,baises_3, image_arr, labels_arr, alpha ,reg, prob,activation_1,dropout, num):

    if num == 2:
        length = len(image_arr)
        #true_array = np.zeros((length,1),dtype=np.float32)

        Z1 = np.matmul(weights_1.T,image_arr.T) + baises_1

        if activation_1 == "relu" or activation_1 == "Relu":
            #A1 = relu_activation(Z1)
            A1 = np.maximum(0,Z1)
        if activation_1 == "sigmoid" or activation_1 == "Sigmoid":
            A1 = sigmoid_activation(Z1)

        ##### Dropout
        if dropout == "Yes" or dropout == "yes" or dropout == "y" or dropout == "Y" or dropout == "YES":
            A1 = dropout_forward(A1, prob)

        Z2 = np.matmul(weights_2.T,A1) + baises_2

        if activation_1 == "relu" or activation_1 == "Relu":
            #A1 = relu_activation(Z1)
            A2 = np.maximum(0,Z2)
        if activation_1 == "sigmoid" or activation_1 == "Sigmoid":
            A2 = sigmoid_activation(Z2)

        ##### Dropout
        if dropout == "Yes" or dropout == "yes" or dropout == "y" or dropout == "Y" or dropout == "YES":
            A2 = dropout_forward(A2, prob)
    
        #A2= Z2
        Z3 = np.matmul(weights_3.T,A2) + baises_3

        Z3_max = np.max(Z3, axis=0)
        Z3_max = np.reshape(Z3_max,(1, length))
        Z3 = np.exp(Z3-Z3_max)

                
        ####################################################
        #Z1 = Z1.transpose()

        A3 = Z3/np.sum(Z3,axis=0)
        #print(A3)

        ##############################################
        ###### Loss
        total_loss = Loss_function(A3,labels_arr,weights_1,weights_2,weights_3, alpha, reg, num)
        return total_loss

    if num ==1:
        length = len(image_arr)
        #true_array = np.zeros((length,1),dtype=np.float32)

        Z1 = np.matmul(weights_1.T,image_arr.T) + baises_1

        if activation_1 == "relu" or activation_1 == "Relu":
            #A1 = relu_activation(Z1)
            A1 = np.maximum(0,Z1)
        if activation_1 == "sigmoid" or activation_1 == "Sigmoid":
            A1 = sigmoid_activation(Z1)
        
        #A1 = np.maximum(0,Z1)

        ##### Dropout
        if dropout == "Yes" or dropout == "yes" or dropout == "y" or dropout == "Y" or dropout == "YES":
            A1 = dropout_forward(A1, prob)

        Z2 = np.matmul(weights_2.T,A1) + baises_2

        #A2 = np.maximum(0,Z2)

        ##### Dropout
        #if dropout == "Yes" or dropout == "yes" or dropout == "y" or dropout == "Y" or dropout == "YES":
        #    A2 = dropout_forward(A2, prob)
    
        #A2= Z2
        #Z3 = np.matmul(weights_3.T,A2) + baises_3

        Z2_max = np.max(Z2, axis=0)
        Z2_max = np.reshape(Z2_max,(1, length))
        Z2 = np.exp(Z2-Z2_max)

                
        ####################################################
        #Z1 = Z1.transpose()

        A2 = Z2/np.sum(Z2,axis=0)

        ##############################################
        ###### Loss
        total_loss = Loss_function(A2,labels_arr,weights_1,weights_2,weights_3, alpha, reg, num)

        return total_loss

    if num ==0:
        length = len(image_arr)
        #true_array = np.zeros((length,1),dtype=np.float32)

        Z1 = np.matmul(weights_1.T,image_arr.T) + baises_1

        Z1_max = np.max(Z1, axis=0)
        Z1_max = np.reshape(Z1_max,(1, length))
        Z1 = np.exp(Z1-Z1_max)
    
        ####################################################
        #Z1 = Z1.transpose()

        A1 = Z1/np.sum(Z1,axis=0)

        ##############################################
        ###### Loss
        total_loss = Loss_function(A1,labels_arr,weights_1,weights_2,weights_3, alpha, reg, num)
        return total_loss
        
    
def accuracy(weights_1,baises_1,weights_2, baises_2,weights_3,baises_3, image_arr, labels_arr,activation_1, num):

    if num==2:
        
        length = len(image_arr)
        true_array = np.zeros((length,1),dtype=np.float32)

        Z1 = np.matmul(weights_1.T,image_arr.T) + baises_1

        if activation_1 == "relu" or activation_1 == "Relu":
            #A1 = relu_activation(Z1)
            A1 = np.maximum(0,Z1)
        if activation_1 == "sigmoid" or activation_1 == "Sigmoid":
            A1 = sigmoid_activation(Z1)

        Z2 = np.matmul(weights_2.T,A1) + baises_2

        if activation_1 == "relu" or activation_1 == "Relu":
            #A1 = relu_activation(Z1)
            A2 = np.maximum(0,Z2)
        if activation_1 == "sigmoid" or activation_1 == "Sigmoid":
            A2 = sigmoid_activation(Z2)

        #A2= Z2
        Z3 = np.matmul(weights_3.T,A2) + baises_3

        Z3_max = np.max(Z3, axis=0)
        Z3_max = np.reshape(Z3_max,(1, length))
        Z3 = np.exp(Z3-Z3_max)

                
        ####################################################
        #Z1 = Z1.transpose()

        A3 = Z3/np.sum(Z3,axis=0)

        #total_loss = Loss_function(A3,labels_arr,weights_1,weights_2,weights_3, alpha, reg)

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

        return acc #, total_loss
        #true_array = (pred_y==true_y).all()
        #s = np.count_nonzero(true_array)
        #acc = s/length 
        #return acc
    if num ==1:
        
        length = len(image_arr)
        Z1 = np.zeros((64,length),dtype = np.float32)
        Z2 = np.zeros((10,length),dtype = np.float32)
        A1 = np.zeros((64,length),dtype = np.float32)
        true_array = np.zeros((length,1),dtype=np.float32)

        Z1 = np.matmul(weights_1.T,image_arr.T) + baises_1

        if activation_1 == "relu" or activation_1 == "Relu":
            #A1 = relu_activation(Z1)
            A1 = np.maximum(0,Z1)
        if activation_1 == "sigmoid" or activation_1 == "Sigmoid":
            A1 = sigmoid_activation(Z1)
        #A1 = np.maximum(0,Z1)

        Z2 = np.matmul(weights_2.T,A1) + baises_2

        Z2_max = np.max(Z2, axis=0)
        Z2_max = np.reshape(Z2_max,(1, length))
        Z2 = np.exp(Z2-Z2_max)
             
        ####################################################
        #Z1 = Z1.transpose()

        A2 = Z2/np.sum(Z2,axis=0)
    
        #total_loss = Loss_function(A2,labels_arr)

        pred_y = np.argmax(A2,axis = 0)
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

        return acc #,total_loss

    if num==0:
            
        length = len(image_arr)
        true_array = np.zeros((length,1),dtype=np.float32)

        Z1 = np.matmul(weights_1.T,image_arr.T) + baises_1

        Z1_max = np.max(Z1, axis=0)
        Z1_max = np.reshape(Z1_max,(1, length))
        Z1 = np.exp(Z1-Z1_max)

                
        ####################################################
        #Z1 = Z1.transpose()

        A1 = Z1/np.sum(Z1,axis=0)

        #total_loss = Loss_function(A3,labels_arr,weights_1,weights_2,weights_3, alpha, reg)

        pred_y = np.argmax(A1,axis = 0)
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

        return acc #, total_loss
        

def NN_2_layers_test(weights_1,baises_1,weights_2,baises_2,weights_3,baises_3,a,b):

    batch_size = 4
    hidden_layer_1 = 64
    hidden_layer_2 = 64

    #layers = input("Number of Hidden layers in Model: ")
    activation_1 = input("Activation Function for Hidden layers: ")
    #activation_2 = input("Activation Function for 2nd Hidden layer: ")
     
    #Normal = input("Want to implement simple normalization or Normalizied Normalization : ")
    #initial = input("Want to implement Gaussioan distribution or Xavier Initialization : ")
    
    num = 2
    reg = input("Want to implement L1 Regression: ")
    decay = input("Want to implement Decay learning rate: ")
    #dropout = input("Want to implement Dropout: ")
    dropout = "n"

    optimizer = input("Type of Optimizer want to use: ")

    #a,b,c,d,e,f = Data_pre_processing(Normal)
    cost_train = []
    #cost_valid = []
    #cost_test = []
    acc_training = []
    #acc_validation = []
    #acc_test = []
    learning_rate_list = []

    #weights_1 = np.zeros((784,64),dtype = np.float32)
    #baises_1 = np.zeros((64,1), dtype = np.float32)

    #weights_2 = np.zeros((64,64),dtype = np.float32)
    #baises_2 = np.zeros((64,1), dtype = np.float32)

    #weights_3 = np.zeros((64,10),dtype = np.float32)
    #baises_3 = np.zeros((10,1), dtype = np.float32)

    mov_weights_1 = np.zeros((10,10),dtype = np.float32)
    mov_baises_1 = np.zeros((10,1), dtype = np.float32)

    mov_weights_2 = np.zeros((10,10),dtype = np.float32)
    mov_baises_2 = np.zeros((10,1), dtype = np.float32)

    mov_weights_3 = np.zeros((10,10),dtype = np.float32)
    mov_baises_3 = np.zeros((10,1), dtype = np.float32)

    dloss_dweights_1 = np.zeros((10,10),dtype = np.float32)
    dloss_dbaises_1  = np.zeros((10,1),dtype = np.float32)

    dloss_dweights_2 = np.zeros((10,10),dtype = np.float32)
    dloss_dbaises_2  = np.zeros((10,1),dtype = np.float32)

    dloss_dweights_3 = np.zeros((10,10),dtype = np.float32)
    dloss_dbaises_3  = np.zeros((10,1),dtype = np.float32)

    dloss_dA1 = np.zeros((10,4),dtype = np.float32)
    dloss_dA2 = np.zeros((10,4),dtype = np.float32)

    Z1 = np.zeros((10,4),dtype = np.float32)
    Z2 = np.zeros((10,4),dtype = np.float32)
    Z3 = np.zeros((10,4),dtype = np.float32)

    Z1_back = np.zeros((10,4),dtype = np.float32)
    Z2_back = np.zeros((10,4),dtype = np.float32)
    Z3_back = np.zeros((10,4),dtype = np.float32)

    learning_rate = 0.001
    beta = 0.9
    alpha = 0.00001
    decay_rate = 0.98
    prob = 0.8


    #if (initial == "gauss" or initial == "Gauss"):
    #    weights_1 = Gaussian_initialization(weights_1)
    #    print("I am in Gauss")
    #if (initial == "xavier" or initial == "Xavier"):
    #    weights_1 = Xavier_initialization(weights_1)
    #    print("I am in Xavier")

    #if (initial == "gauss" or initial == "Gauss"):
    #    weights_2 = Gaussian_initialization(weights_2)
    #    print("I am in Gauss")
    #if (initial == "xavier" or initial == "Xavier"):
    #    weights_2 = Xavier_initialization(weights_2)
    #    print("I am in Xavier")

    #if (initial == "gauss" or initial == "Gauss"):
    #    weights_3 = Gaussian_initialization(weights_3)
    #    print("I am in Gauss")

    #if (initial == "xavier" or initial == "Xavier"):
    #    weights_3 = Xavier_initialization(weights_3)
    #    print("I am in Xavier")



    #for i in range(784):
    #    for j in range(64):
    #        weights_1[i,j] = 0.01 * np.random.randn()

    #for i in range(64):
    #    for j in range(64):
    #        weights_2[i,j] = 0.01 * np.random.randn()

    #for i in range(64):
    #    for j in range(10):
    #        weights_3[i,j] = 0.01 * np.random.randn()


    #for i in tqdm(range(10), total=10 ,desc = "First Loop", unit='Epochs', unit_scale=True):
    for i in range(5):
        n=0
        learning_rate_list.append(learning_rate)
        for j in tqdm(range(1), total=1 ,desc = "Second Loop", unit='Iterations', unit_scale=True):

            x_train = a[n:n+batch_size,:]
            y_train = b[n:n+batch_size,:]

            dloss_dweights_1 = np.zeros((10,10),dtype = np.float32)
            dloss_dbaises_1  = np.zeros((10,1),dtype = np.float32)

            dloss_dweights_2 = np.zeros((10,10),dtype = np.float32)
            dloss_dbaises_2  = np.zeros((10,1),dtype = np.float32)

            dloss_dweights_3 = np.zeros((10,10),dtype = np.float32)
            dloss_dbaises_3  = np.zeros((10,1),dtype = np.float32)

            dloss_dA1 = np.zeros((10,4),dtype = np.float32)
            dloss_dA2 = np.zeros((10,4),dtype = np.float32)

            Z1 = np.zeros((10,4),dtype = np.float32)
            Z2 = np.zeros((10,4),dtype = np.float32)
            Z3 = np.zeros((10,4),dtype = np.float32)

            A1 = np.zeros((10,4),dtype = np.float32)
            A2 = np.zeros((10,4),dtype = np.float32)

            Z1_back = np.zeros((10,4),dtype = np.float32)
            Z2_back = np.zeros((10,4),dtype = np.float32)
            Z3_back = np.zeros((10,4),dtype = np.float32)


            Z1 = np.matmul(weights_1.T,x_train.T) + baises_1

            #A1 = np.maximum(0,Z1)
            #A1 = relu_activation(Z1)
            if activation_1 == "relu" or activation_1 == "Relu":
                A1 = relu_activation(Z1)
            if activation_1 == "sigmoid" or activation_1 == "Sigmoid":
                A1 = sigmoid_activation(Z1)

            ##### Dropout
            if dropout == "Yes" or dropout == "yes" or dropout == "y" or dropout == "Y" or dropout == "YES":
                mask_1 = np.zeros((len(A1),len(A1[1])),dtype = np.float32)
                mask_1 = np.random.rand(len(mask_1),len(mask_1[1]))
                mask_1 = mask_1 < prob

                A1 = np.multiply(A1, mask_1)
                A1 = A1/prob

             #if dropout == "Yes" or dropout == "yes" or dropout == "y" or dropout == "Y" or dropout == "YES":
             #    A1 = dropout_forward(A1, prob)
            

            Z2 = np.matmul(weights_2.T,A1) + baises_2

            #A2 = np.maximum(0,Z2)
            #A2 = relu_activation(Z2)
            if activation_1 == "relu" or activation_1 == "Relu":
                A2 = relu_activation(Z2)
            if activation_1 == "sigmoid" or activation_1 == "Sigmoid":
                A2 = sigmoid_activation(Z2)

            ##### Dropout

            if dropout == "Yes" or dropout == "yes" or dropout == "y" or dropout == "Y" or dropout == "YES":
            #       A2 = dropout_forward(A2, prob)
                mask_2 = np.zeros((len(A2),len(A2[1])),dtype = np.float32)
                mask_2 = np.random.rand(len(mask_2),len(mask_2[1]))
                mask_2 = mask_2 < prob

                A2 = np.multiply(A2, mask_2)
                A2 = A2/prob
            
            #A2= Z2
            Z3 = np.matmul(weights_3.T,A2) + baises_3

            Z3_max = np.max(Z3, axis=0)
            Z3_max = np.reshape(Z3_max,(1, 4))
            Z3 = np.exp(Z3-Z3_max)

                
            ####################################################
            #Z1 = Z1.transpose()

            A3 = Z3/np.sum(Z3,axis=0)

            #print(A3)

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

            ### Dropout_backward

            if dropout == "Yes" or dropout == "yes" or dropout == "y" or dropout == "Y" or dropout == "YES":
                dloss_dA2 =  np.multiply(dloss_dA2, mask_2)
                dloss_dA2 = dloss_dA2/prob


            #Z1_back = dloss_dA1

            ####### Relu Backward

            #for k in range(64):
            #    for l in range(4):
            #        if Z2[k,l] <=0:
            #            Z2_back[k,l] = 0
            #        if Z2[k,l] > 0:
            #            Z2_back[k,l] = dloss_dA2[k,l]

            if activation_1 == "relu" or activation_1 == "Relu":
                Z2_back = relu_activation_back(dloss_dA2, Z2)
            if activation_1 == "sigmoid" or activation_1 == "Sigmoid":
                Z2_back = sigmoid_activation_back(dloss_dA2, Z2)

            #Z2_back = relu_activation_back(dloss_dA2, Z2)

            dloss_dweights_2 = (1./batch_size) * np.matmul(A1,Z2_back.T)
            dloss_dbaises_2 = (1./batch_size) * np.sum(Z2_back, axis = 1, keepdims= True)

            dloss_dA1 = np.matmul(weights_2,Z2_back)

            if dropout == "Yes" or dropout == "yes" or dropout == "y" or dropout == "Y" or dropout == "YES":
                dloss_dA1 =  np.multiply(dloss_dA1, mask_1)
                dloss_dA1 = dloss_dA1/prob

        

            #Z1_back = dloss_dA1

            ####### Relu Backward

            #for k in range(64):
            #    for l in range(4):
            #        if Z1[k,l] <=0:
            #            Z1_back[k,l] = 0
            #        if Z1[k,l] > 0:
            #            Z1_back[k,l] = dloss_dA1[k,l]

            if activation_1 == "relu" or activation_1 == "Relu":
                Z1_back = relu_activation_back(dloss_dA1, Z1)
            if activation_1 == "sigmoid" or activation_1 == "Sigmoid":
                Z1_back = sigmoid_activation_back(dloss_dA1, Z1)

            #Z1_back = relu_activation_back(dloss_dA1, Z1)

            ##################################################################################
            dloss_dweights_1 = (1./batch_size) * np.matmul(x_train.T,Z1_back.T)
            dloss_dbaises_1 = (1./batch_size) * np.sum(Z1_back, axis = 1, keepdims= True)

            if reg == "Yes" or reg == "yes" or reg == "y" or reg == "Y" or reg == "YES":
                dloss_dweights_1 = dloss_dweights_1 + alpha * weights_1
                dloss_dweights_2 = dloss_dweights_2 + alpha * weights_2
                dloss_dweights_3 = dloss_dweights_3 + alpha * weights_3
            

            #weights_1= weights_1- learning_rate*dloss_dweights_1
            #baises_1 = baises_1 - learning_rate*dloss_dbaises_1
        
            #weights_2= weights_2- learning_rate*dloss_dweights_2
            #baises_2 = baises_2 - learning_rate*dloss_dbaises_2

            #weights_3= weights_3- learning_rate*dloss_dweights_3
            #baises_3 = baises_3 - learning_rate*dloss_dbaises_3
            

            if optimizer == "SGD" or optimizer =="sgd":
                weights_1,baises_1,weights_2,baises_2,weights_3,baises_3 = SGD_optimizer(weights_1,baises_1,weights_2,baises_2,weights_3,baises_3,dloss_dweights_1,dloss_dbaises_1,dloss_dweights_2,dloss_dbaises_2,dloss_dweights_3, dloss_dbaises_3, learning_rate, num)

            #print(mov_weights_1)
            #print(mov_weights_2)
            #print(mov_weights_3)
            #print(mov_baises_1)
            #print(mov_baises_2)
            #print(mov_baises_3)
            
            #print(weights_1)
            #print(weights_2)
            #print(weights_3)
            #print(baises_1)
            #print(baises_2)
            #print(baises_3)
            
            
            if optimizer == "Momentum" or optimizer =="momentum":
                weights_1,baises_1,weights_2,baises_2,weights_3,baises_3,mov_weights_1,mov_baises_1,mov_weights_2,mov_baises_2,mov_weights_3,mov_baises_3 = Momentum_optimizer(weights_1,baises_1,weights_2,baises_2,weights_3,baises_3,mov_weights_1,mov_baises_1,mov_weights_2,mov_baises_2,mov_weights_3,mov_baises_3,dloss_dweights_1,dloss_dbaises_1,dloss_dweights_2,dloss_dbaises_2,dloss_dweights_3, dloss_dbaises_3, learning_rate, beta,num)
        
            n = n + 4

        if decay == "Yes" or decay == "yes" or decay == "y" or decay == "Y" or decay == "YES":
            learning_rate = learning_rate_decay(learning_rate,decay_rate)
        #if learning_rate < 0.001:
        #    learning_rate = 0.001
        #print(dloss_dA2)
        #print(dloss_dbaises_2)
        
    

        acc_epoch_train = accuracy(weights_1,baises_1,weights_2, baises_2,weights_3,baises_3,a,b, activation_1,num )
        #acc_epoch_valid = accuracy(weights_1,baises_1,weights_2, baises_2,weights_3,baises_3,c,d, activation_1,num )
        #acc_epoch_test = accuracy(weights_1,baises_1,weights_2, baises_2,weights_3,baises_3,e,f, activation_1,num )
        #print(acc_epoch_train)

        loss_train = forward_prop_for_loss(weights_1,baises_1,weights_2, baises_2,weights_3,baises_3, a, b , alpha ,reg, prob, activation_1, dropout,num)
        #print(loss_train)
        #loss_valid = forward_prop_for_loss(weights_1,baises_1,weights_2, baises_2,weights_3,baises_3, c, d , alpha ,reg, prob, activation_1, dropout,num)
        #loss_test = forward_prop_for_loss(weights_1,baises_1,weights_2, baises_2,weights_3,baises_3, e, f , alpha ,reg, prob, activation_1, dropout,num)
    
        #cost_train.append(loss_train)
        #cost_valid.append(loss_valid)
        #cost_test.append(loss_test)
        #acc_training.append(acc_epoch_train)
        #acc_validation.append(acc_epoch_valid)
        #acc_test.append(acc_epoch_test)
    

        #print("Training Cost for Epoch:",loss_train, " Epoch : ", i)
        #print("Training Accuracy for Epoch:",acc_epoch_train, " Epoch : ", i)  


    #string = "_for_two_layer_" + activation_1 + "_activation_" + Normal + "_Normalization_" + initial+ "_initalization_with_" + reg + "_regression_with_" + decay + "_learning_decay_with_" + dropout + "_dropout_with_" + optimizer + "_optimizer"

    #save_fun( weights_1, "Weights_1" + string)
    #print("Weight_1 file saved for 2 Layer Neural Network.")
    #save_fun( baises_1, "Baises_1"  + string)
    #print("Baises_1 file saved for 2 Layer Neural Network.")
    #save_fun( weights_2, "Weights_2" + string)
    #print("Weight_2 file saved for 2 Layer Neural Network.")
    #save_fun( baises_2, "Baises_2" + string)
    #print("Baises_2 file saved for 2 Layer Neural Network.")
    #save_fun(weights_3 , "Weights_3" + string)
    #print("Weight_3 file saved for 2 Layer Neural Network.")
    #save_fun( baises_3, "Baises_3" + string)
    #print("Baises_3 file saved for 2 Layer Neural Network.")

    return weights_1,baises_1,weights_2, baises_2,weights_3,baises_3,loss_train, acc_epoch_train


def NN_2_layers():

    batch_size = 4
    hidden_layer_1 = 64
    hidden_layer_2 = 64

    #layers = input("Number of Hidden layers in Model: ")

    repeat1 =0
    while(repeat1 ==0):
        Normal = input("Enter type of normalization for Data [Simple or Normal]: ")
        print("Normalization entered : " + Normal)
        if (Normal=="Simple" or Normal=="simple" or Normal=="Normal" or Normal=="normal"):
            repeat1 = 1
    repeat1 = 0

    while(repeat1 ==0):
        initial = input("Enter function for Weights value initialization [Gauss or Xavier]: ")
        print("Weight Intialixation entered : " + initial)
        if (initial == "Gauss" or initial == "gauss" or initial == "Xavier" or initial == "xavier"):
            repeat1 = 1
    repeat1 =0
    while(repeat1 ==0):
        activation_1 = input("Enter Activation function for hidden layers [Relu or Sigmoid]: ")
        print("Activation Function for hidden layers entered : " + activation_1)
        if (activation_1 == "Relu" or activation_1 == "relu" or activation_1 == "Sigmoid" or activation_1 == "sigmoid"):
            repeat1 = 1
    repeat1 = 0
    while (repeat1 ==0):
        reg = input("Want to implement L1 Regression in the Model [y/n]: ")
        print("String entered for L1 Regression : " + reg)
        if (reg == "Y" or reg =="N" or reg =="No" or reg =="Yes" or reg =="n" or reg =="y" or reg =="no" or reg =="yes"):
            repeat1 =1

    repeat1 = 0
    while (repeat1 ==0):
        decay = input("Want to implement decaying learning rate [y/n]: ")
        print("String entered for decaying learning rate : " + decay)
        if (decay == "Y" or decay =="N" or decay =="No" or decay =="Yes" or decay =="n" or decay =="y" or decay =="no" or decay =="yes"):
            repeat1 =1

    #repeat1 = 0
    #while (repeat1 ==0):
    #    dropout = input("Want to implement Dropout[y/n]: ")
    #    print("String entered for dropout : " + dropout)
    #    if (dropout == "Y" or dropout =="N" or dropout =="No" or dropout =="Yes" or dropout =="n" or dropout =="y" or dropout =="no" or dropout =="yes"):
    #        repeat1 =1
    dropout ="n"

    repeat1 = 0
    while (repeat1 ==0):
        optimizer = input("Enter type of Optimizer want to use to train the model [SGD or Momentum]: ")
        print("Optimizer Entered : " + optimizer)
        if (optimizer =="sgd" or optimizer =="SGD" or optimizer =="Momentum" or optimizer =="momentum"  ):
            repeat1 =1

    epoch = input("Enter Number of Epochs for training the model: ")
    print("Epoch value Entered : ",epoch)
    epoch = int(epoch)


    print("\n      Model with 2 hidden layer with 64 Neuron each is under training\n")

    print("Value of hyper parameters used in the model during training")
    print("Learning rate = 0.001 ")
    print("Momentum parameter [beta] = 0.9 ")
    print("Regression parameter [alpha] = 0.00001 ")
    print("Decay rate = 0.98")

    num=2

    a,b,c,d,e,f = Data_pre_processing(Normal)
    cost_train = []
    cost_valid = []
    cost_test = []
    acc_training = []
    acc_validation = []
    acc_test = []
    learning_rate_list = []

    weights_1 = np.zeros((784,64),dtype = np.float32)
    baises_1 = np.zeros((64,1), dtype = np.float32)

    weights_2 = np.zeros((64,64),dtype = np.float32)
    baises_2 = np.zeros((64,1), dtype = np.float32)

    weights_3 = np.zeros((64,10),dtype = np.float32)
    baises_3 = np.zeros((10,1), dtype = np.float32)

    mov_weights_1 = np.zeros((784,64),dtype = np.float32)
    mov_baises_1 = np.zeros((64,1), dtype = np.float32)

    mov_weights_2 = np.zeros((64,64),dtype = np.float32)
    mov_baises_2 = np.zeros((64,1), dtype = np.float32)

    mov_weights_3 = np.zeros((64,10),dtype = np.float32)
    mov_baises_3 = np.zeros((10,1), dtype = np.float32)

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

    learning_rate = 0.001
    beta = 0.9
    alpha = 0.00001
    decay_rate = 0.98
    prob = 0.8


    if (initial == "gauss" or initial == "Gauss"):
        weights_1 = Gaussian_initialization(weights_1)
        print("I am in Gauss")
    if (initial == "xavier" or initial == "Xavier"):
        weights_1 = Xavier_initialization(weights_1)
        print("I am in Xavier")

    if (initial == "gauss" or initial == "Gauss"):
        weights_2 = Gaussian_initialization(weights_2)
        print("I am in Gauss")
    if (initial == "xavier" or initial == "Xavier"):
        weights_2 = Xavier_initialization(weights_2)
        print("I am in Xavier")

    if (initial == "gauss" or initial == "Gauss"):
        weights_3 = Gaussian_initialization(weights_3)
        print("I am in Gauss")

    if (initial == "xavier" or initial == "Xavier"):
        weights_3 = Xavier_initialization(weights_3)
        print("I am in Xavier")



    #for i in range(784):
    #    for j in range(64):
    #        weights_1[i,j] = 0.01 * np.random.randn()

    #for i in range(64):
    #    for j in range(64):
    #        weights_2[i,j] = 0.01 * np.random.randn()

    #for i in range(64):
    #    for j in range(10):
    #        weights_3[i,j] = 0.01 * np.random.randn()


    #for i in tqdm(range(10), total=10 ,desc = "First Loop", unit='Epochs', unit_scale=True):
    for i in range(epoch):
        n=0
        learning_rate_list.append(learning_rate)
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

            A1 = np.zeros((64,4),dtype = np.float32)
            A2 = np.zeros((64,4),dtype = np.float32)

            Z1_back = np.zeros((64,4),dtype = np.float32)
            Z2_back = np.zeros((64,4),dtype = np.float32)
            Z3_back = np.zeros((10,4),dtype = np.float32)


            Z1 = np.matmul(weights_1.T,x_train.T) + baises_1

            #A1 = np.maximum(0,Z1)
            #A1 = relu_activation(Z1)
            if activation_1 == "relu" or activation_1 == "Relu":
                A1 = relu_activation(Z1)
            if activation_1 == "sigmoid" or activation_1 == "Sigmoid":
                A1 = sigmoid_activation(Z1)

            ##### Dropout
            if dropout == "Yes" or dropout == "yes" or dropout == "y" or dropout == "Y" or dropout == "YES":
                mask_1 = np.zeros((len(A1),len(A1[1])),dtype = np.float32)
                mask_1 = np.random.rand(len(mask_1),len(mask_1[1]))
                mask_1 = mask_1 < prob

                A1 = np.multiply(A1, mask_1)
                A1 = A1/prob

             #if dropout == "Yes" or dropout == "yes" or dropout == "y" or dropout == "Y" or dropout == "YES":
             #    A1 = dropout_forward(A1, prob)
            

            Z2 = np.matmul(weights_2.T,A1) + baises_2

            #A2 = np.maximum(0,Z2)
            #A2 = relu_activation(Z2)
            if activation_1 == "relu" or activation_1 == "Relu":
                A2 = relu_activation(Z2)
            if activation_1 == "sigmoid" or activation_1 == "Sigmoid":
                A2 = sigmoid_activation(Z2)

            ##### Dropout

            if dropout == "Yes" or dropout == "yes" or dropout == "y" or dropout == "Y" or dropout == "YES":
            #       A2 = dropout_forward(A2, prob)
                mask_2 = np.zeros((len(A2),len(A2[1])),dtype = np.float32)
                mask_2 = np.random.rand(len(mask_2),len(mask_2[1]))
                mask_2 = mask_2 < prob

                A2 = np.multiply(A2, mask_2)
                A2 = A2/prob
            
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

            ### Dropout_backward

            if dropout == "Yes" or dropout == "yes" or dropout == "y" or dropout == "Y" or dropout == "YES":
                dloss_dA2 =  np.multiply(dloss_dA2, mask_2)
                dloss_dA2 = dloss_dA2/prob


            #Z1_back = dloss_dA1

            ####### Relu Backward

            #for k in range(64):
            #    for l in range(4):
            #        if Z2[k,l] <=0:
            #            Z2_back[k,l] = 0
            #        if Z2[k,l] > 0:
            #            Z2_back[k,l] = dloss_dA2[k,l]

            if activation_1 == "relu" or activation_1 == "Relu":
                Z2_back = relu_activation_back(dloss_dA2, Z2)
            if activation_1 == "sigmoid" or activation_1 == "Sigmoid":
                Z2_back = sigmoid_activation_back(dloss_dA2, Z2)

            #Z2_back = relu_activation_back(dloss_dA2, Z2)

            dloss_dweights_2 = (1./batch_size) * np.matmul(A1,Z2_back.T)
            dloss_dbaises_2 = (1./batch_size) * np.sum(Z2_back, axis = 1, keepdims= True)

            dloss_dA1 = np.matmul(weights_2,Z2_back)

            if dropout == "Yes" or dropout == "yes" or dropout == "y" or dropout == "Y" or dropout == "YES":
                dloss_dA1 =  np.multiply(dloss_dA1, mask_1)
                dloss_dA1 = dloss_dA1/prob

        

            #Z1_back = dloss_dA1

            ####### Relu Backward

            #for k in range(64):
            #    for l in range(4):
            #        if Z1[k,l] <=0:
            #            Z1_back[k,l] = 0
            #        if Z1[k,l] > 0:
            #            Z1_back[k,l] = dloss_dA1[k,l]

            if activation_1 == "relu" or activation_1 == "Relu":
                Z1_back = relu_activation_back(dloss_dA1, Z1)
            if activation_1 == "sigmoid" or activation_1 == "Sigmoid":
                Z1_back = sigmoid_activation_back(dloss_dA1, Z1)

            #Z1_back = relu_activation_back(dloss_dA1, Z1)

            ##################################################################################
            dloss_dweights_1 = (1./batch_size) * np.matmul(x_train.T,Z1_back.T)
            dloss_dbaises_1 = (1./batch_size) * np.sum(Z1_back, axis = 1, keepdims= True)

            if reg == "Yes" or reg == "yes" or reg == "y" or reg == "Y" or reg == "YES":
                dloss_dweights_1 = dloss_dweights_1 + alpha * weights_1
                dloss_dweights_2 = dloss_dweights_2 + alpha * weights_2
                dloss_dweights_3 = dloss_dweights_3 + alpha * weights_3
            

            #weights_1= weights_1- learning_rate*dloss_dweights_1
            #baises_1 = baises_1 - learning_rate*dloss_dbaises_1
        
            #weights_2= weights_2- learning_rate*dloss_dweights_2
            #baises_2 = baises_2 - learning_rate*dloss_dbaises_2

            #weights_3= weights_3- learning_rate*dloss_dweights_3
            #baises_3 = baises_3 - learning_rate*dloss_dbaises_3

            if optimizer == "SGD" or optimizer =="sgd":
                weights_1,baises_1,weights_2,baises_2,weights_3,baises_3 = SGD_optimizer(weights_1,baises_1,weights_2,baises_2,weights_3,baises_3,dloss_dweights_1,dloss_dbaises_1,dloss_dweights_2,dloss_dbaises_2,dloss_dweights_3, dloss_dbaises_3, learning_rate, num)

            if optimizer == "Momentum" or optimizer =="momentum":
                weights_1,baises_1,weights_2,baises_2,weights_3,baises_3,mov_weights_1,mov_baises_1,mov_weights_2,mov_baises_2,mov_weights_3,mov_baises_3 = Momentum_optimizer(weights_1,baises_1,weights_2,baises_2,weights_3,baises_3,mov_weights_1,mov_baises_1,mov_weights_2,mov_baises_2,mov_weights_3,mov_baises_3,dloss_dweights_1,dloss_dbaises_1,dloss_dweights_2,dloss_dbaises_2,dloss_dweights_3, dloss_dbaises_3, learning_rate, beta,num)
        
            n = n + 4

        if decay == "Yes" or decay == "yes" or decay == "y" or decay == "Y" or decay == "YES":
            learning_rate = learning_rate_decay(learning_rate,decay_rate)
        #if learning_rate < 0.001:
        #    learning_rate = 0.001
        
    

        acc_epoch_train = accuracy(weights_1,baises_1,weights_2, baises_2,weights_3,baises_3,a,b, activation_1,num )
        acc_epoch_valid = accuracy(weights_1,baises_1,weights_2, baises_2,weights_3,baises_3,c,d, activation_1,num )
        acc_epoch_test = accuracy(weights_1,baises_1,weights_2, baises_2,weights_3,baises_3,e,f, activation_1,num )

        loss_train = forward_prop_for_loss(weights_1,baises_1,weights_2, baises_2,weights_3,baises_3, a, b , alpha ,reg, prob, activation_1, dropout,num)
        loss_valid = forward_prop_for_loss(weights_1,baises_1,weights_2, baises_2,weights_3,baises_3, c, d , alpha ,reg, prob, activation_1, dropout,num)
        loss_test = forward_prop_for_loss(weights_1,baises_1,weights_2, baises_2,weights_3,baises_3, e, f , alpha ,reg, prob, activation_1, dropout,num)
    
        cost_train.append(loss_train)
        cost_valid.append(loss_valid)
        cost_test.append(loss_test)
        acc_training.append(acc_epoch_train)
        acc_validation.append(acc_epoch_valid)
        acc_test.append(acc_epoch_test)
    

        print("Training Cost for Epoch:",loss_train, " Epoch : ", i)
        print("Training Accuracy for Epoch:",acc_epoch_train, " Epoch : ", i)  


    string = "_for_two_layer_" + activation_1 + "_activation_" + Normal + "_Normalization_" + initial+ "_initalization_with_" + reg + "_regression_with_" + decay + "_learning_decay_with_" + dropout + "_dropout_with_" + optimizer + "_optimizer"

    save_fun( weights_1, "Weights_1" + string)
    print("Weight_1 file saved for 2 Layer Neural Network.")
    save_fun( baises_1, "Baises_1"  + string)
    print("Baises_1 file saved for 2 Layer Neural Network.")
    save_fun( weights_2, "Weights_2" + string)
    print("Weight_2 file saved for 2 Layer Neural Network.")
    save_fun( baises_2, "Baises_2" + string)
    print("Baises_2 file saved for 2 Layer Neural Network.")
    save_fun(weights_3 , "Weights_3" + string)
    print("Weight_3 file saved for 2 Layer Neural Network.")
    save_fun( baises_3, "Baises_3" + string)
    print("Baises_3 file saved for 2 Layer Neural Network.")

    return cost_train, cost_valid , cost_test, acc_training,acc_validation,acc_test ,learning_rate_list,string, epoch 


def NN_1_layers_test(weights_1,baises_1,weights_2,baises_2,a,b):

    batch_size = 4
    hidden_layer_1 = 64
    #hidden_layer_2 = 16
    num = 1
    #layers = input("Number of Hidden layers in Model: ")
    activation_1 = input("Activation Function for Hidden layer: ")

    #Normal = input("Want to implement simple normalization or Normalizied Normalization : ")
    #initial = input("Want to implement Gaussioan distribution or Xavier Initialization : ")

    reg = input("Want to implement L1 Regression: ")
    decay = input("Want to implement Decay learning rate: ")
    #dropout = input("Want to implement Dropout: ")
    dropout = "n"

    optimizer = input("Type of Optimizer want to use: ")

    #a,b,c,d,e,f = Data_pre_processing(Normal)
    #cost_train = []
    #cost_valid = []
    #cost_test = []
    #acc_training = []
    #acc_validation = []
    #acc_test = []
    #learning_rate_list = []

    #weights_1 = np.zeros((10,10),dtype = np.float32)
    #baises_1 = np.zeros((10,1), dtype = np.float32)

    #weights_2 = np.zeros((10,10),dtype = np.float32)
    #baises_2 = np.zeros((10,1), dtype = np.float32)

    weights_3 = 0
    baises_3 = 0
    
    mov_weights_1 = np.zeros((10,10),dtype = np.float32)
    mov_baises_1 = np.zeros((10,1), dtype = np.float32)

    mov_weights_2 = np.zeros((10,10),dtype = np.float32)
    mov_baises_2 = np.zeros((10,1), dtype = np.float32)

    mov_weights_3 = 0
    mov_baises_3 = 0

    dloss_dweights_1 = np.zeros((10,10),dtype = np.float32)
    dloss_dbaises_1  = np.zeros((10,1),dtype = np.float32)

    dloss_dweights_2 = np.zeros((10,10),dtype = np.float32)
    dloss_dbaises_2  = np.zeros((10,1),dtype = np.float32)

    dloss_dweights_3 = 0
    dloss_dbaises_3 = 0

    dloss_dA1 = np.zeros((10,4),dtype = np.float32)

    Z1 = np.zeros((10,4),dtype = np.float32)
    Z2 = np.zeros((10,4),dtype = np.float32)

    Z1_back = np.zeros((10,4),dtype = np.float32)
    Z2_back = np.zeros((10,4),dtype = np.float32)

    learning_rate = 0.001
    beta = 0.9
    alpha = 0.00001
    decay_rate = 0.98
    prob = 0.8

    #if (initial == "gauss" or initial == "Gauss"):
    #    weights_1 = Gaussian_initialization(weights_1)
    #    print("I am in Gauss")
    #if (initial == "xavier" or initial == "Xavier"):
    #    weights_1 = Xavier_initialization(weights_1)
    #    print("I am in Xavier")

    #if (initial == "gauss" or initial == "Gauss"):
    #    weights_2 = Gaussian_initialization(weights_2)
    #    print("I am in Gauss")
    #if (initial == "xavier" or initial == "Xavier"):
    #    weights_2 = Xavier_initialization(weights_2)
    #    print("I am in Xavier")


    #for i in range(784):
    #    for j in range(64):
    #        weights_1[i,j] = 0.01 * np.random.randn()

    #for i in range(64):
    #    for j in range(10):
    #        weights_2[i,j] = 0.01 * np.random.randn()

    #for i in tqdm(range(10), total=10 ,desc = "First Loop", unit='Epochs', unit_scale=True):
    for i in range(5):
        n=0
        #learning_rate_list.append(learning_rate)
        for j in tqdm(range(1), total=1 ,desc = "Second Loop", unit='Iterations', unit_scale=True):
        
            x_train = a[n:n+batch_size,:]
            y_train = b[n:n+batch_size,:]


            #batch_size = x_train.shape[0]

            dloss_dweights_1 = np.zeros((10,10),dtype = np.float32)
            dloss_dbaises_1  = np.zeros((10,1),dtype = np.float32)

            dloss_dweights_2 = np.zeros((10,10),dtype = np.float32)
            dloss_dbaises_2  = np.zeros((10,1),dtype = np.float32)

            dloss_dA1 = np.zeros((10,4),dtype = np.float32)

            Z1 = np.zeros((10,4),dtype = np.float32)
            Z2 = np.zeros((10,4),dtype = np.float32)

            A1 = np.zeros((10,4),dtype = np.float32)

            Z1_back = np.zeros((10,batch_size),dtype = np.float32)
            Z2_back = np.zeros((10,batch_size),dtype = np.float32)


            Z1 = np.matmul(weights_1.T,x_train.T) + baises_1

            #A1 = np.maximum(0,Z1)
            if activation_1 == "relu" or activation_1 == "Relu":
                A1 = relu_activation(Z1)
            if activation_1 == "sigmoid" or activation_1 == "Sigmoid":
                A1 = sigmoid_activation(Z1)

            ##### Dropout
            if dropout == "Yes" or dropout == "yes" or dropout == "y" or dropout == "Y" or dropout == "YES":
                mask_1 = np.zeros((len(A1),len(A1[1])),dtype = np.float32)
                mask_1 = np.random.rand(len(mask_1),len(mask_1[1]))
                mask_1 = mask_1 < prob

                A1 = np.multiply(A1, mask_1)
                A1 = A1/prob

            #if dropout == "Yes" or dropout == "yes" or dropout == "y" or dropout == "Y" or dropout == "YES":
            #    A1 = dropout_forward(A1, prob)
            

            Z2 = np.matmul(weights_2.T,A1) + baises_2

            #A2 = np.maximum(0,Z2)
            #A2 = relu_activation(Z2)

            ##### Dropout

            #if dropout == "Yes" or dropout == "yes" or dropout == "y" or dropout == "Y" or dropout == "YES":
            #       A2 = dropout_forward(A2, prob)
            #    mask_2 = np.zeros((len(A2),len(A2[1])),dtype = np.float32)
            #    mask_2 = np.random.rand(len(mask_2),len(mask_2[1]))
            #    mask_2 = mask_2 < prob

            #    A2 = np.multiply(A2, mask_2)
            #    A2 = A2/prob
            
            #A2= Z2
            #Z3 = np.matmul(weights_3.T,A2) + baises_3

            Z2_max = np.max(Z2, axis=0)
            Z2_max = np.reshape(Z2_max,(1, 4))
            Z2 = np.exp(Z2-Z2_max)

                
            ####################################################
            #Z1 = Z1.transpose()

            A2 = Z2/np.sum(Z2,axis=0)
            #print(A2)

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

            Z2_back = A2 - y_train.T
            #dloss_dweights_3 = (1./batch_size) * np.matmul(A2,Z3_back.T)
            #dloss_dbaises_3 = (1./batch_size) * np.sum(Z3_back, axis = 1, keepdims= True)
        
            #dloss_dA2 = np.matmul(weights_3, Z3_back)

            ### Dropout_backward

            #if dropout == "Yes" or dropout == "yes" or dropout == "y" or dropout == "Y" or dropout == "YES":
            #    dloss_dA2 =  np.multiply(dloss_dA2, mask_2)
            #    dloss_dA2 = dloss_dA2/prob


            #Z1_back = dloss_dA1

            ####### Relu Backward

            #for k in range(64):
            #    for l in range(4):
            #        if Z2[k,l] <=0:
            #            Z2_back[k,l] = 0
            #        if Z2[k,l] > 0:
            #            Z2_back[k,l] = dloss_dA2[k,l]

            #Z2_back = relu_activation_back(dloss_dA2, Z2)

            dloss_dweights_2 = (1./batch_size) * np.matmul(A1,Z2_back.T)
            dloss_dbaises_2 = (1./batch_size) * np.sum(Z2_back, axis = 1, keepdims= True)

            dloss_dA1 = np.matmul(weights_2,Z2_back)

            if dropout == "Yes" or dropout == "yes" or dropout == "y" or dropout == "Y" or dropout == "YES":
                dloss_dA1 =  np.multiply(dloss_dA1, mask_1)
                dloss_dA1 = dloss_dA1/prob
        

            #Z1_back = dloss_dA1

            ####### Relu Backward

            #for k in range(64):
            #    for l in range(4):
            #        if Z1[k,l] <=0:
            #            Z1_back[k,l] = 0
            #        if Z1[k,l] > 0:
            #            Z1_back[k,l] = dloss_dA1[k,l]

            if activation_1 == "relu" or activation_1 == "Relu":
                Z1_back = relu_activation_back(dloss_dA1, Z1)
            if activation_1 == "sigmoid" or activation_1 == "Sigmoid":
                Z1_back = sigmoid_activation_back(dloss_dA1, Z1)

            #Z1_back = relu_activation_back(dloss_dA1, Z1)

            ##################################################################################
            dloss_dweights_1 = (1./batch_size) * np.matmul(x_train.T,Z1_back.T)
            dloss_dbaises_1 = (1./batch_size) * np.sum(Z1_back, axis = 1, keepdims= True)

            if reg == "Yes" or reg == "yes" or reg == "y" or reg == "Y" or reg == "YES":
                dloss_dweights_1 = dloss_dweights_1 + alpha * weights_1
                dloss_dweights_2 = dloss_dweights_2 + alpha * weights_2
                #dloss_dweights_3 = dloss_dweights_3 + alpha * dloss_dweights_3
            

            #weights_1= weights_1- learning_rate*dloss_dweights_1
            #baises_1 = baises_1 - learning_rate*dloss_dbaises_1
        
            #weights_2= weights_2- learning_rate*dloss_dweights_2
            #baises_2 = baises_2 - learning_rate*dloss_dbaises_2

            #weights_3= weights_3- learning_rate*dloss_dweights_3
            #baises_3 = baises_3 - learning_rate*dloss_dbaises_3

            if optimizer == "SGD" or optimizer =="sgd":
                weights_1,baises_1,weights_2,baises_2 = SGD_optimizer(weights_1,baises_1,weights_2,baises_2,weights_3,baises_3,dloss_dweights_1,dloss_dbaises_1,dloss_dweights_2,dloss_dbaises_2,dloss_dweights_3, dloss_dbaises_3, learning_rate,num)


            if optimizer == "Momentum" or optimizer =="momentum":
                weights_1,baises_1,weights_2,baises_2,mov_weights_1,mov_baises_1,mov_weights_2,mov_baises_2 = Momentum_optimizer(weights_1,baises_1,weights_2,baises_2,weights_3,baises_3,mov_weights_1,mov_baises_1,mov_weights_2,mov_baises_2,mov_weights_3,mov_baises_3,dloss_dweights_1,dloss_dbaises_1,dloss_dweights_2,dloss_dbaises_2,dloss_dweights_3, dloss_dbaises_3, learning_rate, beta, num)

        
            n = n + 4

        if decay == "Yes" or decay == "yes" or decay == "y" or decay == "Y" or decay == "YES":
            learning_rate = learning_rate_decay(learning_rate,decay_rate)
        #if learning_rate < 0.001:
        #    learning_rate = 0.001
               
        acc_epoch_train = accuracy(weights_1,baises_1,weights_2, baises_2,weights_3,baises_3, a, b,activation_1, num)
        #acc_epoch_valid = accuracy(weights_1,baises_1,weights_2, baises_2,weights_3,baises_3, c, d,activation_1, num)
        #acc_epoch_test = accuracy(weights_1,baises_1,weights_2, baises_2,weights_3,baises_3, e, f,activation_1, num)

        loss_train = forward_prop_for_loss(weights_1,baises_1,weights_2, baises_2,weights_3,baises_3, a, b, alpha ,reg, prob,activation_1,dropout, num)
        #loss_valid = forward_prop_for_loss(weights_1,baises_1,weights_2, baises_2,weights_3,baises_3, c, d, alpha ,reg, prob,activation_1,dropout, num)
        #loss_test = forward_prop_for_loss(weights_1,baises_1,weights_2, baises_2,weights_3,baises_3, e, f, alpha ,reg, prob,activation_1,dropout, num)
    
        #cost_train.append(loss_train)
        #cost_valid.append(loss_valid)
        #cost_test.append(loss_test)
        #acc_training.append(acc_epoch_train)
        #acc_validation.append(acc_epoch_valid)
        #acc_test.append(acc_epoch_test)

        #print("Training Cost for Epoch:",loss_train, " Epoch : ", i)
        #print("Training Accuracy for Epoch:",acc_epoch_train, " Epoch : ", i)   
            


    #string = "_for_one_layer_" + activation_1 + "_activation_"+ Normal + "_Normalization_" + initial+ "_initalization_with_" + reg + "_regression_with_" + decay + "_learning_decay_with_" + dropout + "_dropout_with_" + optimizer + "_optimizer"

    #save_fun( weights_1, "Weights_1" + string)
    #print("Weight_1 file saved for 1 Layer Neural Network.")
    #save_fun( baises_1, "Baises_1"  + string)
    #print("Baises_1 file saved for 1 Layer Neural Network.")
    #save_fun( weights_2, "Weights_2" + string)
    #print("Weight_2 file saved for 1 Layer Neural Network.")
    #save_fun( baises_2, "Baises_2" + string)
    #print("Baises_2 file saved for 1 Layer Neural Network.")
    #save_fun(weights_3 , "Weights_3" + string)
    #print("Weight_3 file saved ")
    #save_fun( baises_3, "Baises_3" + string)
    #print("Baises_3 file saved ")

    return weights_1,baises_1,weights_2, baises_2, loss_train, acc_epoch_train

def NN_1_layers():

    batch_size = 4
    hidden_layer_1 = 64
    #hidden_layer_2 = 16
    num = 1
    #layers = input("Number of Hidden layers in Model: ")

    repeat1 =0
    while(repeat1 ==0):
        Normal = input("Enter type of normalization for Data [Simple or Normal]: ")
        print("Normalization entered : " + Normal)
        if (Normal=="Simple" or Normal=="simple" or Normal=="Normal" or Normal=="normal"):
            repeat1 = 1
    repeat1 = 0

    while(repeat1 ==0):
        initial = input("Enter function for Weights value initialization [Gauss or Xavier]: ")
        print("Weight Intialixation entered : " + initial)
        if (initial == "Gauss" or initial == "gauss" or initial == "Xavier" or initial == "xavier"):
            repeat1 = 1
    repeat1 =0
    while(repeat1 ==0):
        activation_1 = input("Enter Activation function for hidden layers [Relu or Sigmoid]: ")
        print("Activation Function for hidden layers entered : " + activation_1)
        if (activation_1 == "Relu" or activation_1 == "relu" or activation_1 == "Sigmoid" or activation_1 == "sigmoid"):
            repeat1 = 1
    repeat1 = 0
    while (repeat1 ==0):
        reg = input("Want to implement L1 Regression in the Model [y/n]: ")
        print("String entered for L1 Regression : " + reg)
        if (reg == "Y" or reg =="N" or reg =="No" or reg =="Yes" or reg =="n" or reg =="y" or reg =="no" or reg =="yes"):
            repeat1 =1

    repeat1 = 0
    while (repeat1 ==0):
        decay = input("Want to implement decaying learning rate [y/n]: ")
        print("String entered for decaying learning rate : " + decay)
        if (decay == "Y" or decay =="N" or decay =="No" or decay =="Yes" or decay =="n" or decay =="y" or decay =="no" or decay =="yes"):
            repeat1 =1

    #repeat1 = 0
    #while (repeat1 ==0):
    #    dropout = input("Want to implement Dropout[y/n]: ")
    #    print("String entered for dropout : " + dropout)
    #    if (dropout == "Y" or dropout =="N" or dropout =="No" or dropout =="Yes" or dropout =="n" or dropout =="y" or dropout =="no" or dropout =="yes"):
    #        repeat1 =1
    dropout = "n"

    repeat1 = 0
    while (repeat1 ==0):
        optimizer = input("Enter type of Optimizer want to use to train the model [SGD or Momentum]: ")
        print("Optimizer Entered : " + optimizer)
        if (optimizer =="sgd" or optimizer =="SGD" or optimizer =="Momentum" or optimizer =="momentum"  ):
            repeat1 =1

    epoch = input("Enter Number of Epochs for training the model: ")
    print("Epoch value Entered : ",epoch)
    epoch = int(epoch)

    print("\n        Model with 1 hidden layer with 64 Neuron is under training\n")

    print("Value of hyper parameters used in the model during training")
    print("Learning rate = 0.001 ")
    print("Momentum parameter [beta] = 0.9 ")
    print("Regression parameter [alpha] = 0.00001 ")
    print("Decay rate = 0.98")

    a,b,c,d,e,f = Data_pre_processing(Normal)
    cost_train = []
    cost_valid = []
    cost_test = []
    acc_training = []
    acc_validation = []
    acc_test = []
    learning_rate_list = []

    weights_1 = np.zeros((784,64),dtype = np.float32)
    baises_1 = np.zeros((64,1), dtype = np.float32)

    weights_2 = np.zeros((64,10),dtype = np.float32)
    baises_2 = np.zeros((10,1), dtype = np.float32)

    weights_3 = 0
    baises_3 = 0
    
    mov_weights_1 = np.zeros((784,64),dtype = np.float32)
    mov_baises_1 = np.zeros((64,1), dtype = np.float32)

    mov_weights_2 = np.zeros((64,10),dtype = np.float32)
    mov_baises_2 = np.zeros((10,1), dtype = np.float32)

    mov_weights_3 = 0
    mov_baises_3 = 0

    dloss_dweights_1 = np.zeros((784,64),dtype = np.float32)
    dloss_dbaises_1  = np.zeros((64,1),dtype = np.float32)

    dloss_dweights_2 = np.zeros((64,10),dtype = np.float32)
    dloss_dbaises_2  = np.zeros((10,1),dtype = np.float32)

    dloss_dweights_3 = 0
    dloss_dbaises_3 = 0

    dloss_dA1 = np.zeros((64,4),dtype = np.float32)

    Z1 = np.zeros((64,4),dtype = np.float32)
    Z2 = np.zeros((10,4),dtype = np.float32)

    Z1_back = np.zeros((64,4),dtype = np.float32)
    Z2_back = np.zeros((10,4),dtype = np.float32)

    learning_rate = 0.001
    beta = 0.9
    alpha = 0.00001
    decay_rate = 0.98
    prob = 0.8

    if (initial == "gauss" or initial == "Gauss"):
        weights_1 = Gaussian_initialization(weights_1)
        print("I am in Gauss")
    if (initial == "xavier" or initial == "Xavier"):
        weights_1 = Xavier_initialization(weights_1)
        print("I am in Xavier")

    if (initial == "gauss" or initial == "Gauss"):
        weights_2 = Gaussian_initialization(weights_2)
        print("I am in Gauss")
    if (initial == "xavier" or initial == "Xavier"):
        weights_2 = Xavier_initialization(weights_2)
        print("I am in Xavier")


    #for i in range(784):
    #    for j in range(64):
    #        weights_1[i,j] = 0.01 * np.random.randn()

    #for i in range(64):
    #    for j in range(10):
    #        weights_2[i,j] = 0.01 * np.random.randn()

    #for i in tqdm(range(10), total=10 ,desc = "First Loop", unit='Epochs', unit_scale=True):
    for i in range(epoch):
        n=0
        learning_rate_list.append(learning_rate)
        for j in tqdm(range(14500), total=14500 ,desc = "Second Loop", unit='Iterations', unit_scale=True):
        
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


            Z1 = np.matmul(weights_1.T,x_train.T) + baises_1

            #A1 = np.maximum(0,Z1)
            if activation_1 == "relu" or activation_1 == "Relu":
                A1 = relu_activation(Z1)
            if activation_1 == "sigmoid" or activation_1 == "Sigmoid":
                A1 = sigmoid_activation(Z1)

            ##### Dropout
            if dropout == "Yes" or dropout == "yes" or dropout == "y" or dropout == "Y" or dropout == "YES":
                mask_1 = np.zeros((len(A1),len(A1[1])),dtype = np.float32)
                mask_1 = np.random.rand(len(mask_1),len(mask_1[1]))
                mask_1 = mask_1 < prob

                A1 = np.multiply(A1, mask_1)
                A1 = A1/prob

            #if dropout == "Yes" or dropout == "yes" or dropout == "y" or dropout == "Y" or dropout == "YES":
            #    A1 = dropout_forward(A1, prob)
            

            Z2 = np.matmul(weights_2.T,A1) + baises_2

            #A2 = np.maximum(0,Z2)
            #A2 = relu_activation(Z2)

            ##### Dropout

            #if dropout == "Yes" or dropout == "yes" or dropout == "y" or dropout == "Y" or dropout == "YES":
            #       A2 = dropout_forward(A2, prob)
            #    mask_2 = np.zeros((len(A2),len(A2[1])),dtype = np.float32)
            #    mask_2 = np.random.rand(len(mask_2),len(mask_2[1]))
            #    mask_2 = mask_2 < prob

            #    A2 = np.multiply(A2, mask_2)
            #    A2 = A2/prob
            
            #A2= Z2
            #Z3 = np.matmul(weights_3.T,A2) + baises_3

            Z2_max = np.max(Z2, axis=0)
            Z2_max = np.reshape(Z2_max,(1, 4))
            Z2 = np.exp(Z2-Z2_max)

                
            ####################################################
            #Z1 = Z1.transpose()

            A2 = Z2/np.sum(Z2,axis=0)

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

            Z2_back = A2 - y_train.T
            #dloss_dweights_3 = (1./batch_size) * np.matmul(A2,Z3_back.T)
            #dloss_dbaises_3 = (1./batch_size) * np.sum(Z3_back, axis = 1, keepdims= True)
        
            #dloss_dA2 = np.matmul(weights_3, Z3_back)

            ### Dropout_backward

            #if dropout == "Yes" or dropout == "yes" or dropout == "y" or dropout == "Y" or dropout == "YES":
            #    dloss_dA2 =  np.multiply(dloss_dA2, mask_2)
            #    dloss_dA2 = dloss_dA2/prob


            #Z1_back = dloss_dA1

            ####### Relu Backward

            #for k in range(64):
            #    for l in range(4):
            #        if Z2[k,l] <=0:
            #            Z2_back[k,l] = 0
            #        if Z2[k,l] > 0:
            #            Z2_back[k,l] = dloss_dA2[k,l]

            #Z2_back = relu_activation_back(dloss_dA2, Z2)

            dloss_dweights_2 = (1./batch_size) * np.matmul(A1,Z2_back.T)
            dloss_dbaises_2 = (1./batch_size) * np.sum(Z2_back, axis = 1, keepdims= True)

            dloss_dA1 = np.matmul(weights_2,Z2_back)

            if dropout == "Yes" or dropout == "yes" or dropout == "y" or dropout == "Y" or dropout == "YES":
                dloss_dA1 =  np.multiply(dloss_dA1, mask_1)
                dloss_dA1 = dloss_dA1/prob
        

            #Z1_back = dloss_dA1

            ####### Relu Backward

            #for k in range(64):
            #    for l in range(4):
            #        if Z1[k,l] <=0:
            #            Z1_back[k,l] = 0
            #        if Z1[k,l] > 0:
            #            Z1_back[k,l] = dloss_dA1[k,l]

            if activation_1 == "relu" or activation_1 == "Relu":
                Z1_back = relu_activation_back(dloss_dA1, Z1)
            if activation_1 == "sigmoid" or activation_1 == "Sigmoid":
                Z1_back = sigmoid_activation_back(dloss_dA1, Z1)

            #Z1_back = relu_activation_back(dloss_dA1, Z1)

            ##################################################################################
            dloss_dweights_1 = (1./batch_size) * np.matmul(x_train.T,Z1_back.T)
            dloss_dbaises_1 = (1./batch_size) * np.sum(Z1_back, axis = 1, keepdims= True)

            if reg == "Yes" or reg == "yes" or reg == "y" or reg == "Y" or reg == "YES":
                dloss_dweights_1 = dloss_dweights_1 + alpha * weights_1
                dloss_dweights_2 = dloss_dweights_2 + alpha * weights_2
                #dloss_dweights_3 = dloss_dweights_3 + alpha * dloss_dweights_3
            

            #weights_1= weights_1- learning_rate*dloss_dweights_1
            #baises_1 = baises_1 - learning_rate*dloss_dbaises_1
        
            #weights_2= weights_2- learning_rate*dloss_dweights_2
            #baises_2 = baises_2 - learning_rate*dloss_dbaises_2

            #weights_3= weights_3- learning_rate*dloss_dweights_3
            #baises_3 = baises_3 - learning_rate*dloss_dbaises_3

            if optimizer == "SGD" or optimizer =="sgd":
                weights_1,baises_1,weights_2,baises_2 = SGD_optimizer(weights_1,baises_1,weights_2,baises_2,weights_3,baises_3,dloss_dweights_1,dloss_dbaises_1,dloss_dweights_2,dloss_dbaises_2,dloss_dweights_3, dloss_dbaises_3, learning_rate,num)


            if optimizer == "Momentum" or optimizer =="momentum":
                weights_1,baises_1,weights_2,baises_2,mov_weights_1,mov_baises_1,mov_weights_2,mov_baises_2 = Momentum_optimizer(weights_1,baises_1,weights_2,baises_2,weights_3,baises_3,mov_weights_1,mov_baises_1,mov_weights_2,mov_baises_2,mov_weights_3,mov_baises_3,dloss_dweights_1,dloss_dbaises_1,dloss_dweights_2,dloss_dbaises_2,dloss_dweights_3, dloss_dbaises_3, learning_rate, beta, num)

        
            n = n + 4

        if decay == "Yes" or decay == "yes" or decay == "y" or decay == "Y" or decay == "YES":
            learning_rate = learning_rate_decay(learning_rate,decay_rate)
        #if learning_rate < 0.001:
        #    learning_rate = 0.001
               
        acc_epoch_train = accuracy(weights_1,baises_1,weights_2, baises_2,weights_3,baises_3, a, b,activation_1, num)
        acc_epoch_valid = accuracy(weights_1,baises_1,weights_2, baises_2,weights_3,baises_3, c, d,activation_1, num)
        acc_epoch_test = accuracy(weights_1,baises_1,weights_2, baises_2,weights_3,baises_3, e, f,activation_1, num)

        loss_train = forward_prop_for_loss(weights_1,baises_1,weights_2, baises_2,weights_3,baises_3, a, b, alpha ,reg, prob,activation_1,dropout, num)
        loss_valid = forward_prop_for_loss(weights_1,baises_1,weights_2, baises_2,weights_3,baises_3, c, d, alpha ,reg, prob,activation_1,dropout, num)
        loss_test = forward_prop_for_loss(weights_1,baises_1,weights_2, baises_2,weights_3,baises_3, e, f, alpha ,reg, prob,activation_1,dropout, num)
    
        cost_train.append(loss_train)
        cost_valid.append(loss_valid)
        cost_test.append(loss_test)
        acc_training.append(acc_epoch_train)
        acc_validation.append(acc_epoch_valid)
        acc_test.append(acc_epoch_test)

        print("Training Cost for Epoch:",loss_train, " Epoch : ", i)
        print("Training Accuracy for Epoch:",acc_epoch_train, " Epoch : ", i)   
            


    string = "_for_one_layer_" + activation_1 + "_activation_"+ Normal + "_Normalization_" + initial+ "_initalization_with_" + reg + "_regression_with_" + decay + "_learning_decay_with_" + dropout + "_dropout_with_" + optimizer + "_optimizer"

    save_fun( weights_1, "Weights_1" + string)
    print("Weight_1 file saved for 1 Layer Neural Network.")
    save_fun( baises_1, "Baises_1"  + string)
    print("Baises_1 file saved for 1 Layer Neural Network.")
    save_fun( weights_2, "Weights_2" + string)
    print("Weight_2 file saved for 1 Layer Neural Network.")
    save_fun( baises_2, "Baises_2" + string)
    print("Baises_2 file saved for 1 Layer Neural Network.")
    #save_fun(weights_3 , "Weights_3" + string)
    #print("Weight_3 file saved ")
    #save_fun( baises_3, "Baises_3" + string)
    #print("Baises_3 file saved ")

    return cost_train, cost_valid , cost_test, acc_training,acc_validation,acc_test ,learning_rate_list,string,epoch

def NN_0_layers_test(weights_1,baises_1,a,b):
    batch_size = 4
    #hidden_layer_1 = 64
    #hidden_layer_2 = 16

    #layers = input("Number of Hidden layers in Model: ")
    #activation_1 = input("Activation Function for 1st Hidden layer: ")
    #activation_2 = input("Activation Function for 2nd Hidden layer: ")
    activation_1 = "no"

    #Normal = input("Want to implement simple normalization or Normalizied Normalization : ")
    #initial = input("Want to implement Gaussioan distribution or Xavier Initialization : ")
    num = 0

    reg = input("Want to implement L1 Regression: ")
    decay = input("Want to implement Decay learning rate: ")
    #dropout = input("Want to implement Dropout: ")
    dropout = "n"
    optimizer = input("Type of Optimizer want to use: ")
    #prob = 0.8

    #a,b,c,d,e,f = Data_pre_processing(Normal)

    #cost_train = []
    #cost_valid = []
    #cost_test = []
    #acc_training = []
    #acc_validation = []
    #acc_test = []
    #learning_rate_list = []

    #weights_1 = np.zeros((784,10),dtype = np.float32)
    #baises_1 = np.zeros((10,1), dtype = np.float32)

    weights_2 = 0
    baises_2 = 0

    weights_3 = 0
    baises_3 = 0

    dloss_dweights_1 = np.zeros((10,10),dtype = np.float32)
    dloss_dbaises_1  = np.zeros((10,1),dtype = np.float32)

    dloss_dweights_2 = 0
    dloss_dbaises_2  = 0

    dloss_dweights_3 = 0
    dloss_dbaises_3  = 0


    mov_weights_1 = np.zeros((10,10),dtype = np.float32)
    mov_baises_1 = np.zeros((10,1), dtype = np.float32)

    mov_weights_2 = 0
    mov_baises_2 = 0

    mov_weights_3 = 0
    mov_baises_3 = 0

    Z1 = np.zeros((10,4),dtype = np.float32)
    Z1_back = np.zeros((10,4),dtype = np.float32)

    learning_rate = 0.001
    beta = 0.9
    alpha = 0.00001
    decay_rate = 0.98
    prob = 0.8

    #for i in range(784):
    #    for j in range(10):
    #        weights[i,j] = 0.01 * np.random.randn()


    #if (initial == "gauss" or initial == "Gauss"):
    #    weights_1 = Gaussian_initialization(weights_1)
    #    print("I am in Gauss")
    #if (initial == "xavier" or initial == "Xavier"):
    #    weights_1 = Xavier_initialization(weights_1)
    #    print("I am in Xavier")

    #for i in tqdm(range(10), total=10 ,desc = "First Loop", unit='Epochs', unit_scale=True):
    for i in range(5):
        n=0
        #learning_rate_list.append(learning_rate)
        for j in tqdm(range(1), total=1 ,desc = "Second Loop", unit='Iterations', unit_scale=True):

        
            x_train = a[n:n+batch_size,:]
            y_train = b[n:n+batch_size,:]

            dloss_dweights_1 = np.zeros((10,10),dtype = np.float32)
            dloss_dbaises_1  = np.zeros((10,1),dtype = np.float32)

            Z1 = np.zeros((10,4),dtype = np.float32)

            Z1_back = np.zeros((10,4),dtype = np.float32)

            Z1 = np.matmul(weights_1.T,x_train.T) + baises_1

            Z1_max = np.max(Z1, axis=0)
            Z1_max = np.reshape(Z1_max,(1, 4))
            Z1 = np.exp(Z1-Z1_max)
                
            ####################################################
            #Z1 = Z1.transpose()

            A1 = Z1/np.sum(Z1,axis=0)

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

            Z1_back = A1 - y_train.T

            ##################################################################################
            dloss_dweights_1 = (1./batch_size) * np.matmul(x_train.T,Z1_back.T)
            dloss_dbaises_1 = (1./batch_size) * np.sum(Z1_back, axis = 1, keepdims= True)

            if reg == "Yes" or reg == "yes" or reg == "y" or reg == "Y" or reg == "YES":
                dloss_dweights_1 = dloss_dweights_1 + alpha * weights_1
                #dloss_dweights_2 = dloss_dweights_2 + alpha * dloss_dweights_2
                #dloss_dweights_3 = dloss_dweights_3 + alpha * dloss_dweights_3

            if optimizer == "SGD" or optimizer =="sgd":
                weights_1,baises_1 = SGD_optimizer(weights_1,baises_1,weights_2,baises_2,weights_3,baises_3,dloss_dweights_1,dloss_dbaises_1,dloss_dweights_2,dloss_dbaises_2,dloss_dweights_3, dloss_dbaises_3, learning_rate,num)

            if optimizer == "Momentum" or optimizer =="momentum":
                weights_1,baises_1,mov_weights_1,mov_baises_1 = Momentum_optimizer(weights_1,baises_1,weights_2,baises_2,weights_3,baises_3,mov_weights_1,mov_baises_1,mov_weights_2,mov_baises_2,mov_weights_3,mov_baises_3,dloss_dweights_1,dloss_dbaises_1,dloss_dweights_2,dloss_dbaises_2,dloss_dweights_3, dloss_dbaises_3, learning_rate, beta, num)

  

            #for k in range(784):
            #    for l in range(10):
            #        weights[k,l] = weights[k,l] - learning_rate*dloss_dweights[k,l]

            #for k in range(10):
            #    baises[k,:] = baises[k,:] - learning_rate*dloss_dbaises[k,:]


            n = n + 4

        if decay == "Yes" or decay == "yes" or decay == "y" or decay == "Y" or decay == "YES":
            learning_rate = learning_rate_decay(learning_rate,decay_rate)
        #if learning_rate < 0.001:
        #    learning_rate = 0.001

        acc_epoch_train = accuracy(weights_1,baises_1,weights_2, baises_2,weights_3,baises_3, a, b,activation_1, num)
        #acc_epoch_valid = accuracy(weights_1,baises_1,weights_2, baises_2,weights_3,baises_3, c, d,activation_1, num)
        #acc_epoch_test = accuracy(weights_1,baises_1,weights_2, baises_2,weights_3,baises_3, e, f,activation_1, num)

        loss_train = forward_prop_for_loss(weights_1,baises_1,weights_2, baises_2,weights_3,baises_3, a, b, alpha ,reg, prob,activation_1,dropout, num)
        #loss_valid = forward_prop_for_loss(weights_1,baises_1,weights_2, baises_2,weights_3,baises_3, c, d, alpha ,reg, prob,activation_1,dropout, num)
        #loss_test = forward_prop_for_loss(weights_1,baises_1,weights_2, baises_2,weights_3,baises_3, e, f , alpha ,reg, prob,activation_1,dropout, num)
    
        #cost_train.append(loss_train)
        #cost_valid.append(loss_valid)
        #cost_test.append(loss_test)
        #acc_training.append(acc_epoch_train)
        #acc_validation.append(acc_epoch_valid)
        #acc_test.append(acc_epoch_test)
    

        #print("Training Cost for Epoch:",loss_train, " Epoch : ", i)
        #print("Training Accuracy for Epoch:",acc_epoch_train, " Epoch : ", i) 

    #string = "_for_zero_layer_"  + Normal + "_Normalization_" + initial+ "_initalization_with_" + reg + "_regression_with_" + decay + "_learning_decay_with_" + optimizer + "_optimizer"

    #save_fun( weights_1, "Weights_1" + string)
    #print("Weight_1 file saved for 0 Layer Neural Network.")
    #save_fun( baises_1, "Baises_1"  + string)
    #print("Baises_1 file saved for 0 Layer Neural Network.")
    #save_fun( weights_2, "Weights_2" + string)
    #save_fun( baises_2, "Baises_2" + string)
    #save_fun(weights_3 , "Weights_3" + string)
    #save_fun( baises_3, "Baises_3" + string)
    
    return weights_1,baises_1,loss_train, acc_epoch_train

def NN_0_layers():
    batch_size = 4
    #hidden_layer_1 = 64
    #hidden_layer_2 = 16

    #layers = input("Number of Hidden layers in Model: ")
    #activation_1 = input("Activation Function for 1st Hidden layer: ")
    #activation_2 = input("Activation Function for 2nd Hidden layer: ")

    repeat1 =0
    while(repeat1 ==0):
        Normal = input("Enter type of normalization for Data [Simple or Normal]: ")
        print("Normalization entered : " + Normal)
        if (Normal=="Simple" or Normal=="simple" or Normal=="Normal" or Normal=="normal"):
            repeat1 = 1
    repeat1 = 0

    while(repeat1 ==0):
        initial = input("Enter function for Weights value initialization [Gauss or Xavier]: ")
        print("Weight Intialixation entered : " + initial)
        if (initial == "Gauss" or initial == "gauss" or initial == "Xavier" or initial == "xavier"):
            repeat1 = 1
    repeat1 =0
    #while(repeat1 ==0):
    #    activation_1 = input("Enter Activation function for hidden layers [Relu or Sigmoid]: ")
    #    print("Activation Function for hidden layers entered : " + activation_1)
    #    if (activation_1 == "Relu" or activation_1 == "relu" or activation_1 == "Sigmoid" or activation_1 == "sigmoid"):
    #        repeat1 = 1
    #repeat1 = 0
    activation_1 = "no"

    while (repeat1 ==0):
        reg = input("Want to implement L1 Regression in the Model [y/n]: ")
        print("String entered for L1 Regression : " + reg)
        if (reg == "Y" or reg =="N" or reg =="No" or reg =="Yes" or reg =="n" or reg =="y" or reg =="no" or reg =="yes"):
            repeat1 =1

    repeat1 = 0
    while (repeat1 ==0):
        decay = input("Want to implement decaying learning rate [y/n]: ")
        print("String entered for decaying learning rate : " + decay)
        if (decay == "Y" or decay =="N" or decay =="No" or decay =="Yes" or decay =="n" or decay =="y" or decay =="no" or decay =="yes"):
            repeat1 =1



    #repeat1 = 0
    #while (repeat1 ==0):
    #    dropout = input("Want to implement Dropout[y/n]: ")
    #    print("String entered for dropout : " + dropout)
    #    if (dropout == "Y" or dropout =="N" or dropout =="No" or dropout =="Yes" or dropout =="n" or dropout =="y" or dropout =="no" or dropout =="yes"):
    #        repeat1 =1
    dropout = "no"

    repeat1 = 0
    while (repeat1 ==0):
        optimizer = input("Enter type of Optimizer want to use to train the model [SGD or Momentum]: ")
        print("Optimizer Entered : " + optimizer)
        if (optimizer =="sgd" or optimizer =="SGD" or optimizer =="Momentum" or optimizer =="momentum"  ):
            repeat1 =1

    epoch = input("Enter Number of Epochs for training the model: ")
    print("Epoch value Entered : ",epoch)
    epoch = int(epoch)
    #prob = 0.8
    print("\n          Model with 0 hidden layer is under training\n")

    print("Value of hyper parameters used in the model during training")

    print("Learning rate = 0.001 ")
    print("Momentum parameter [beta] = 0.9 ")
    print("Regression parameter [alpha] = 0.00001 ")
    print("Decay rate = 0.98")

    a,b,c,d,e,f = Data_pre_processing(Normal)

    cost_train = []
    cost_valid = []
    cost_test = []
    acc_training = []
    acc_validation = []
    acc_test = []
    learning_rate_list = []

    weights_1 = np.zeros((784,10),dtype = np.float32)
    baises_1 = np.zeros((10,1), dtype = np.float32)

    weights_2 = 0
    baises_2 = 0

    weights_3 = 0
    baises_3 = 0

    dloss_dweights_1 = np.zeros((784,10),dtype = np.float32)
    dloss_dbaises_1  = np.zeros((10,1),dtype = np.float32)

    dloss_dweights_2 = 0
    dloss_dbaises_2  = 0

    dloss_dweights_3 = 0
    dloss_dbaises_3  = 0


    mov_weights_1 = np.zeros((784,10),dtype = np.float32)
    mov_baises_1 = np.zeros((10,1), dtype = np.float32)

    mov_weights_2 = 0
    mov_baises_2 = 0

    mov_weights_3 = 0
    mov_baises_3 = 0

    Z1 = np.zeros((10,4),dtype = np.float32)
    Z1_back = np.zeros((10,4),dtype = np.float32)

    learning_rate = 0.001
    beta = 0.9
    alpha = 0.00001
    decay_rate = 0.98
    prob = 0.8

    #for i in range(784):
    #    for j in range(10):
    #        weights[i,j] = 0.01 * np.random.randn()


    if (initial == "gauss" or initial == "Gauss"):
        weights_1 = Gaussian_initialization(weights_1)
        print("I am in Gauss")
    if (initial == "xavier" or initial == "Xavier"):
        weights_1 = Xavier_initialization(weights_1)
        print("I am in Xavier")

    #for i in tqdm(range(10), total=10 ,desc = "First Loop", unit='Epochs', unit_scale=True):
    for i in range(epoch):
        n=0
        learning_rate_list.append(learning_rate)
        for j in tqdm(range(14500), total=14500 ,desc = "Second Loop", unit='Iterations', unit_scale=True):

        
            x_train = a[n:n+batch_size,:]
            y_train = b[n:n+batch_size,:]

            dloss_dweights_1 = np.zeros((784,10),dtype = np.float32)
            dloss_dbaises_1  = np.zeros((10,1),dtype = np.float32)

            Z1 = np.zeros((10,4),dtype = np.float32)

            Z1_back = np.zeros((10,4),dtype = np.float32)

            Z1 = np.matmul(weights_1.T,x_train.T) + baises_1

            Z1_max = np.max(Z1, axis=0)
            Z1_max = np.reshape(Z1_max,(1, 4))
            Z1 = np.exp(Z1-Z1_max)
                
            ####################################################
            #Z1 = Z1.transpose()

            A1 = Z1/np.sum(Z1,axis=0)

            ##############################################
            ###### Loss
            #loss = np.zeros((10,4), dtype = np.float32)
            #for k in range(10):
            #    for l in range(4):
            #        loss[k,l] = - y_train[l,k]*(np.log(A3[k,l]))
            #loss_per_sample = np.sum(loss,axis=1)
            #total_loss = np.sum(loss_per_sample,axis= 0)
            #total_loss = total_loss/batch_size

            #########################################################
            #####  Back propogation

            Z1_back = A1 - y_train.T


            dloss_dweights_1 = (1./batch_size) * np.matmul(x_train.T,Z1_back.T)
            dloss_dbaises_1 = (1./batch_size) * np.sum(Z1_back, axis = 1, keepdims= True)

            if reg == "Yes" or reg == "yes" or reg == "y" or reg == "Y" or reg == "YES":
                dloss_dweights_1 = dloss_dweights_1 + alpha * weights_1
                #dloss_dweights_2 = dloss_dweights_2 + alpha * dloss_dweights_2
                #dloss_dweights_3 = dloss_dweights_3 + alpha * dloss_dweights_3

            if optimizer == "SGD" or optimizer =="sgd":
                weights_1,baises_1 = SGD_optimizer(weights_1,baises_1,weights_2,baises_2,weights_3,baises_3,dloss_dweights_1,dloss_dbaises_1,dloss_dweights_2,dloss_dbaises_2,dloss_dweights_3, dloss_dbaises_3, learning_rate,num)

            if optimizer == "Momentum" or optimizer =="momentum":
                weights_1,baises_1,mov_weights_1,mov_baises_1 = Momentum_optimizer(weights_1,baises_1,weights_2,baises_2,weights_3,baises_3,mov_weights_1,mov_baises_1,mov_weights_2,mov_baises_2,mov_weights_3,mov_baises_3,dloss_dweights_1,dloss_dbaises_1,dloss_dweights_2,dloss_dbaises_2,dloss_dweights_3, dloss_dbaises_3, learning_rate, beta, num)

  

            #for k in range(784):
            #    for l in range(10):
            #        weights[k,l] = weights[k,l] - learning_rate*dloss_dweights[k,l]

            #for k in range(10):
            #    baises[k,:] = baises[k,:] - learning_rate*dloss_dbaises[k,:]


            n = n + 4

        if decay == "Yes" or decay == "yes" or decay == "y" or decay == "Y" or decay == "YES":
            learning_rate = learning_rate_decay(learning_rate,decay_rate)
        #if learning_rate < 0.001:
        #    learning_rate = 0.001

        acc_epoch_train = accuracy(weights_1,baises_1,weights_2, baises_2,weights_3,baises_3, a, b,activation_1, num)
        acc_epoch_valid = accuracy(weights_1,baises_1,weights_2, baises_2,weights_3,baises_3, c, d,activation_1, num)
        acc_epoch_test = accuracy(weights_1,baises_1,weights_2, baises_2,weights_3,baises_3, e, f,activation_1, num)

        loss_train = forward_prop_for_loss(weights_1,baises_1,weights_2, baises_2,weights_3,baises_3, a, b, alpha ,reg, prob,activation_1,dropout, num)
        loss_valid = forward_prop_for_loss(weights_1,baises_1,weights_2, baises_2,weights_3,baises_3, c, d, alpha ,reg, prob,activation_1,dropout, num)
        loss_test = forward_prop_for_loss(weights_1,baises_1,weights_2, baises_2,weights_3,baises_3, e, f , alpha ,reg, prob,activation_1,dropout, num)
    
        cost_train.append(loss_train)
        cost_valid.append(loss_valid)
        cost_test.append(loss_test)
        acc_training.append(acc_epoch_train)
        acc_validation.append(acc_epoch_valid)
        acc_test.append(acc_epoch_test)
    

        print("Training Cost for Epoch:",loss_train, " Epoch : ", i)
        print("Training Accuracy for Epoch:",acc_epoch_train, " Epoch : ", i) 

    string = "_for_zero_layer_"  + Normal + "_Normalization_" + initial+ "_initalization_with_" + reg + "_regression_with_" + decay + "_learning_decay_with_" + optimizer + "_optimizer"

    save_fun( weights_1, "Weights_1" + string)
    print("Weight_1 file saved for 0 Layer Neural Network.")
    save_fun( baises_1, "Baises_1"  + string)
    print("Baises_1 file saved for 0 Layer Neural Network.")
    #save_fun( weights_2, "Weights_2" + string)
    #save_fun( baises_2, "Baises_2" + string)
    #save_fun(weights_3 , "Weights_3" + string)
    #save_fun( baises_3, "Baises_3" + string)
    
    return cost_train, cost_valid , cost_test, acc_training,acc_validation,acc_test ,learning_rate_list,string,epoch 

if __name__=='__main__':
    repeat =0
    while(repeat==0):
        layers = input("Enter Number of Hidden layers in Model[ 0 or 1 or 2 ]: ")
        print("Hidden layers value entered : ", layers,)
        if (layers == "2" or layers == "two" or layers == "Two" or layers == "1" or layers == "one" or layers == "One" or layers == "0" or layers == "zero" or layers == "Zero"):
            repeat = 1
    if (layers == "2" or layers == "two" or layers == "Two"):
        cost_train, cost_valid , cost_test, acc_training,acc_validation,acc_test ,learning_rate_list,string,epoch  = NN_2_layers()
    if (layers == "1" or layers == "one" or layers == "One"):
        cost_train, cost_valid , cost_test, acc_training,acc_validation,acc_test ,learning_rate_list,string,epoch  = NN_1_layers()
    if (layers == "0" or layers == "zero" or layers == "Zero"):
        cost_train, cost_valid , cost_test, acc_training,acc_validation,acc_test ,learning_rate_list,string,epoch =  NN_0_layers()

    x = np.arange(0,epoch, 1)
    y = cost_train

    save_fun_list(y , "Training_loss" + string)
    print("Training Loss History file saved for Neural Network.")
    y1 = cost_valid
    save_fun_list(y1 , "Validation_loss" + string)
    print("Validation Loss History file saved for Neural Network.")
    y2 = cost_test
    save_fun_list(y2 , "Test_loss" + string)
    print("Test Loss History file saved for Neural Network.")

    plt.xlabel('Epochs')  
    plt.ylabel('Loss') 
 
    plt.title('Loss_Graph')
    plt.plot(x, y,"b", label='Training Loss')
    plt.plot(x, y1,"g", label='Validation Loss')
    plt.plot(x, y2,"r", label='Test Loss')
    #plt.plot(x, y,"b", x, y1,"g", x, y2,"r")
    plt.legend()
    plt.show()


    z= acc_training
    save_fun_list(z , "Training_acc" + string)
    print("Training Accuracy History file saved for Neural Network.")
    z1 = acc_validation
    save_fun_list(z1 , "Validation_acc" + string)
    print("Validation Accuracy History file saved for Neural Network.")
    z2 = acc_test
    save_fun_list(z2 , "Test_acc" + string)
    print("Test Accuracy History file saved for Neural Network.")
    plt.xlabel('Epochs')  
    plt.ylabel('Accuracy')

    plt.title('Accuracy Graph')

    plt.plot(x, z,"b", label='Training Accuracy')
    plt.plot(x, z1,"g", label='Validation Accuracy')
    plt.plot(x, z2,"r", label='Test Accuracy')
    #plt.plot(x, z,"b", x, z1,"g", x, z2,"r")
    plt.legend()
    plt.show()

    l = learning_rate_list

    save_fun_list(l , "Learning_rate" + string)
    plt.xlabel('Epochs')  
    plt.ylabel('Learning rate')

    plt.title('Learning rate_Graph')
    plt.plot(x, l,"b", label='Learning rate')
    plt.legend()
    plt.show()
