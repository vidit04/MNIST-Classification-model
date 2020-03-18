import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv

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
    std = np.std(image_array)
    image_array = (image_array - mean)/std
    return image_array

def Simple_normalization(image_array):
    image_array = image_array / 255
    return image_array

def Gaussian_initialization(weights):
    row_weights = weights.shape[0]
    width_weights = weights.shape[1]
    weights = 0.01 * np.random.randn(row_weights,width_weights)
    return weights

def Xavier_initialization(weights):
    row_weights = weights.shape[0]
    width_weights = weights.shape[1]
    weights = np.random.randn(row_weights,width_weights) * np.sqrt(1./row_weights)
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
    mask = mask < prob
    array_Act = np.multiply(array_Act, mask)
    array_Act = array_Act/prob
    return array_Act

def Momentum_optimizer(weights_1,baises_1,weights_2,baises_2,weights_3,baises_3,mov_weights_1,mov_baises_1,mov_weights_2,mov_baises_2,mov_weights_3,mov_baises_3,dloss_dweights_1,dloss_dbaises_1,dloss_dweights_2,dloss_dbaises_2,dloss_dweights_3, dloss_dbaises_3, learning_rate, beta):

    mov_weights_1 = beta*mov_weights_1 + (1. - beta)* dloss_dweights_1
    mov_baises_1 = beta*mov_baises_1 + (1. - beta)* dloss_dbaises_1

    mov_weights_2 = beta*mov_weights_2 + (1. - beta)* dloss_dweights_2
    mov_baises_2 = beta*mov_baises_2+(1. - beta)* dloss_dbaises_2


    mov_weights_3 = beta*mov_weights_3 + (1. - beta)* dloss_dweights_3
    mov_baises_3 = beta*mov_baises_3 + (1. - beta)* dloss_dbaises_3
    

    weights_1 = weights_1- learning_rate*mov_weights_1
    baises_1 = baises_1 - learning_rate*mov_baises_1
        
    weights_2= weights_2- learning_rate*mov_weights_2
    baises_2 = baises_2 - learning_rate*mov_baises_2

    weights_3= weights_3- learning_rate*mov_weights_3
    baises_3 = baises_3 - learning_rate*mov_baises_3

    return weights_1,baises_1,weights_2,baises_2,weights_3,baises_3,mov_weights_1,mov_baises_1,mov_weights_2,mov_baises_2,mov_weights_3,mov_baises_3



def SGD_optimizer(weights_1,baises_1,weights_2,baises_2,weights_3,baises_3,dloss_dweights_1,dloss_dbaises_1,dloss_dweights_2,dloss_dbaises_2,dloss_dweights_3, dloss_dbaises_3, learning_rate):

    weights_1= weights_1- learning_rate*dloss_dweights_1
    baises_1 = baises_1 - learning_rate*dloss_dbaises_1
        
    weights_2= weights_2- learning_rate*dloss_dweights_2
    baises_2 = baises_2 - learning_rate*dloss_dbaises_2

    weights_3= weights_3- learning_rate*dloss_dweights_3
    baises_3 = baises_3 - learning_rate*dloss_dbaises_3

    return weights_1,baises_1,weights_2,baises_2,weights_3,baises_3


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


def reg_loss_layer_2(weights_1,weights_2,weights_3, alpha):

    reg_loss_2 = (0.5 * alpha * np.sum(weights_1*weights_1)) + (0.5 * alpha * np.sum(weights_2*weights_2)) + (0.5 * alpha * np.sum(weights_3 * weights_3))

    return reg_loss_2
    
    
def Loss_function(pred_y,true_y,weights_1,weights_2,weights_3 ,alpha, reg):
    ##############################################
    ###### Loss
    length_loss = len(pred_y[1])
    epsilon=1e-12
    pred_y = np.clip(pred_y, epsilon, 1. - epsilon)
    loss = np.zeros((10,length_loss), dtype = np.float32)
    for k in range(10):
        for l in range(length_loss):
            loss[k,l] = - true_y[l,k]*(np.log(pred_y[k,l]))
    loss_per_sample = np.sum(loss,axis=1)
    total_loss = np.sum(loss_per_sample,axis= 0)
    total_loss = total_loss/length_loss

    if (reg == "Yes" or reg == "yes" or reg == "y" or reg == "Y" or reg == "YES"):
        reg_loss = reg_loss_layer_2(weights_1,weights_2,weights_3, alpha)
        total_loss = total_loss + reg_loss

    return total_loss

def forward_prop_for_loss(weights_1,baises_1,weights_2, baises_2,weights_3,baises_3, image_arr, labels_arr, alpha ,reg, prob):

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

    ##############################################
    ###### Loss
    total_loss = Loss_function(A3,labels_arr,weights_1,weights_2,weights_3, alpha, reg)

    return total_loss

    
def accuracy(weights_1,baises_1,weights_2, baises_2,weights_3,baises_3, image_arr, labels_arr):


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
def Momentum_optimizer_layer_1(weights_1,baises_1,weights_2,baises_2,mov_weights_1,mov_baises_1,mov_weights_2,mov_baises_2,dloss_dweights_1,dloss_dbaises_1,dloss_dweights_2,dloss_dbaises_2, learning_rate, beta):

    mov_weights_1 = beta*mov_weights_1 + (1. - beta)* dloss_dweights_1
    mov_baises_1 = beta*mov_baises_1 + (1. - beta)* dloss_dbaises_1

    mov_weights_2 = beta*mov_weights_2 + (1. - beta)* dloss_dweights_2
    mov_baises_2 = beta*mov_baises_2+(1. - beta)* dloss_dbaises_2
    
    weights_1 = weights_1- learning_rate*mov_weights_1
    baises_1 = baises_1 - learning_rate*mov_baises_1
        
    weights_2= weights_2- learning_rate*mov_weights_2
    baises_2 = baises_2 - learning_rate*mov_baises_2

    return weights_1,baises_1,weights_2,baises_2,mov_weights_1,mov_baises_1,mov_weights_2,mov_baises_2



def SGD_optimizer_layer_1(weights_1,baises_1,weights_2,baises_2,dloss_dweights_1,dloss_dbaises_1,dloss_dweights_2,dloss_dbaises_2, learning_rate):

    weights_1= weights_1- learning_rate*dloss_dweights_1
    baises_1 = baises_1 - learning_rate*dloss_dbaises_1
        
    weights_2= weights_2- learning_rate*dloss_dweights_2
    baises_2 = baises_2 - learning_rate*dloss_dbaises_2

    return weights_1,baises_1,weights_2,baises_2

def reg_loss_layer_1(weights_1,weights_2, alpha):

    reg_loss_1 = (0.5 * alpha * np.sum(weights_1*weights_1)) + (0.5 * alpha * np.sum(weights_2*weights_2))

    return reg_loss_1


def Loss_function_layer_1(pred_y,true_y,weights_1,weights_2,alpha, reg):
    ##############################################
    ###### Loss
    length_loss = len(pred_y[1])
    epsilon=1e-12
    pred_y = np.clip(pred_y, epsilon, 1. - epsilon)
    #print(pred_y.shape)
    #print(true_y.shape)
    loss = np.zeros((10,length_loss), dtype = np.float32)
    for k in range(10):
        for l in range(length_loss):
            loss[k,l] = - true_y[l,k]*(np.log(pred_y[k,l]))
    loss_per_sample = np.sum(loss,axis=1)
    total_loss = np.sum(loss_per_sample,axis= 0)
    total_loss = total_loss/length_loss

    if (reg == "Yes" or reg == "yes" or reg == "y" or reg == "Y" or reg == "YES"):
        reg_loss = reg_loss_layer_1(weights_1,weights_2, alpha)
        total_loss = total_loss + reg_loss
        
    return total_loss

def forward_prop_for_loss_layer_1(weights_1,baises_1,weights_2, baises_2, image_arr, labels_arr, alpha ,reg, prob):

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
    total_loss = Loss_function_layer_1(A2,labels_arr,weights_1,weights_2, alpha, reg)

    return total_loss

def accuracy_layer_1(weights_1,baises_1,weights_2, baises_2, image_arr, labels_arr):

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
    #true_array = (pred_y==true_y).all()
    #s = np.count_nonzero(true_array)
    #acc = s/length
    #return acc,total_loss
def Momentum_optimizer_layer_0(weights_1,baises_1,mov_weights_1,mov_baises_1,dloss_dweights_1,dloss_dbaises_1, learning_rate, beta):

    mov_weights_1 = beta*mov_weights_1 + (1. - beta)* dloss_dweights_1
    mov_baises_1 = beta*mov_baises_1 + (1. - beta)* dloss_dbaises_1
    
    weights_1 = weights_1- learning_rate*mov_weights_1
    baises_1 = baises_1 - learning_rate*mov_baises_1

    return weights_1,baises_1,mov_weights_1,mov_baises_1



def SGD_optimizer_layer_0(weights_1,baises_1,dloss_dweights_1,dloss_dbaises_1, learning_rate):

    weights_1= weights_1- learning_rate*dloss_dweights_1
    baises_1 = baises_1 - learning_rate*dloss_dbaises_1

    return weights_1,baises_1


def reg_loss_layer_0(weights_1, alpha):

    reg_loss_0 = 0.5 * alpha * np.sum(weights_1*weights_1)

    return reg_loss_0
    
    
def Loss_function_layer_0(pred_y,true_y,weights_1,alpha, reg):
    ##############################################
    ###### Loss
    length_loss = len(pred_y[1])
    epsilon=1e-12
    pred_y = np.clip(pred_y, epsilon, 1. - epsilon)
    loss = np.zeros((10,length_loss), dtype = np.float32)
    for k in range(10):
        for l in range(length_loss):
            loss[k,l] = - true_y[l,k]*(np.log(pred_y[k,l]))
    loss_per_sample = np.sum(loss,axis=1)
    total_loss = np.sum(loss_per_sample,axis= 0)
    total_loss = total_loss/length_loss

    if (reg == "Yes" or reg == "yes" or reg == "y" or reg == "Y" or reg == "YES"):
        reg_loss = reg_loss_layer_0(weights_1, alpha)
        total_loss = total_loss + reg_loss

    return total_loss

def forward_prop_for_loss_layer_0(weights_1,baises_1, image_arr, labels_arr, alpha ,reg, prob):

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
    total_loss = Loss_function_layer_0(A1,labels_arr,weights_1, alpha, reg)

    return total_loss

def accuracy_layer_0(weights_1,baises_1, image_arr, labels_arr):
    
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
    #true_array = (pred_y==true_y).all()
    #s = np.count_nonzero(true_array)
    #acc = s/length 
    #return acc


