import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv

#os.chdir("C:\Users\user\Desktop\MNIST")
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
    total_loss = Loss_function(A1,labels_arr,weights_1, alpha, reg)

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

batch_size = 4
hidden_layer_1 = 64
hadden_layer_2 = 16

layers = input("Number of Hidden layers in Model: ")
activation_1 = input("Activation Function for 1st Hidden layer: ")
#activation_2 = input("Activation Function for 2nd Hidden layer: ")

Normal = input("Want to implement simple normalization or Normalizied Normalization : ")
initial = input("Want to implement Gaussioan distribution or Xavier Initialization : ")


reg = input("Want to implement L1 Regression: ")
decay = input("Want to implement Decay learning rate: ")
dropout = input("Want to implement Dropout: ")
optimizer = input("Type of Optimizer want to use: ")

a,b,c,d,e,f = Data_pre_processing()

cost_train = []
cost_valid = []
cost_test = []
acc_training = []
acc_validation = []
acc_test = []
learning_rate_list = []

weights_1 = np.zeros((784,10),dtype = np.float32)
baises_1 = np.zeros((10,1), dtype = np.float32)

dloss_dweights_1 = np.zeros((784,10),dtype = np.float32)
dloss_dbaises_1  = np.zeros((10,1),dtype = np.float32)

mov_weights_1 = np.zeros((784,10),dtype = np.float32)
mov_baises_1 = np.zeros((10,1), dtype = np.float32)

Z1 = np.zeros((10,4),dtype = np.float32)
Z1_back = np.zeros((10,4),dtype = np.float32)

learning_rate = 0.01
beta = 0.9
alpha = 0.00001
decay_rate = 0.9
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
for i in range(10):
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

        #####  Back propogation

        Z1_back = A1 - y_train.T

        ##################################################################################
        dloss_dweights_1 = (1./batch_size) * np.matmul(x_train.T,Z1_back.T)
        dloss_dbaises_1 = (1./batch_size) * np.sum(Z1_back, axis = 1, keepdims= True)

        if reg == "Yes" or reg == "yes" or reg == "y" or reg == "Y" or reg == "YES":
            dloss_dweights_1 = dloss_dweights_1 + alpha * dloss_dweights_1
            #dloss_dweights_2 = dloss_dweights_2 + alpha * dloss_dweights_2
            #dloss_dweights_3 = dloss_dweights_3 + alpha * dloss_dweights_3

        if optimizer == "SGD" or optimizer =="sgd":
            weights_1,baises_1 = SGD_optimizer_layer_0(weights_1,baises_1,dloss_dweights_1,dloss_dbaises_1, learning_rate)

        if optimizer == "Momentum" or optimizer =="momentum":
            weights_1,baises_1,mov_weights_1,mov_baises_1 = Momentum_optimizer_layer_0(weights_1,baises_1,mov_weights_1,mov_baises_1,dloss_dweights_1,dloss_dbaises_1, learning_rate, beta)
  

        #for k in range(784):
        #    for l in range(10):
        #        weights[k,l] = weights[k,l] - learning_rate*dloss_dweights[k,l]

        #for k in range(10):
        #    baises[k,:] = baises[k,:] - learning_rate*dloss_dbaises[k,:]


        n = n + 4

    if decay == "Yes" or decay == "yes" or decay == "y" or decay == "Y" or decay == "YES":
        learning_rate = learning_rate_decay(learning_rate,decay_rate)
    if learning_rate < 0.001:
        learning_rate = 0.001

    acc_epoch_train = accuracy_layer_0(weights_1,baises_1 , a, b )
    acc_epoch_valid = accuracy_layer_0(weights_1,baises_1, c, d )
    acc_epoch_test = accuracy_layer_0(weights_1,baises_1, e, f )

    loss_train = forward_prop_for_loss_layer_0(weights_1,baises_1, a, b , alpha ,reg, prob)
    loss_valid = forward_prop_for_loss_layer_0(weights_1,baises_1, c, d , alpha ,reg, prob)
    loss_test = forward_prop_for_loss_layer_0(weights_1,baises_1, e, f , alpha ,reg, prob)
    
    cost_train.append(loss_train)
    cost_valid.append(loss_valid)
    cost_test.append(loss_test)
    acc_training.append(acc_epoch_train)
    acc_validation.append(acc_epoch_valid)
    acc_test.append(acc_epoch_test)
    

    print("Final Cost :",loss_train, " Epoch : ", i)
    print("Final Accuracy :",acc_epoch_train, " Epoch : ", i) 

string = "_for_" + layers + "_layer_" + activation_1 + "_activation_" + Normal + "_Normalization_" + initial+ "_initalization_with_" + reg + "_regression_with_" + decay + "_learning_decay_with_" + dropout + "_dropout_with_" + optimizer + "_optimizer"

save_fun( weights_1, "Weights_1" + string)
save_fun( baises_1, "Baises_1"  + string)
#save_fun( weights_2, "Weights_2" + string)
#save_fun( baises_2, "Baises_2" + string)
#save_fun(weights_3 , "Weights_3" + string)
#save_fun( baises_3, "Baises_3" + string)


x = np.arange(0,10, 1)
y = cost_train

save_fun_list(y , "Training_loss" + string)
y1 = cost_valid
save_fun_list(y1 , "Validation_loss" + string)
y2 = cost_test
save_fun_list(y2 , "Test_loss" + string)

plt.xlabel('Epochs')  
plt.ylabel('Loss') 
 
plt.title('Training_Loss_Graph') 
plt.plot(x, y,"b", x, y1,"g", x, y2,"r")
plt.show()

z= acc_training
save_fun_list(z , "Training_acc" + string)
z1 = acc_validation
save_fun_list(z1 , "Validation_acc" + string)
z2 = acc_test
save_fun_list(z2 , "Test_acc" + string)
plt.xlabel('Epochs')  
plt.ylabel('Accuracy')

plt.title('Training_Acc_Graph') 
plt.plot(x, z,"b", x, z1,"g", x, z2,"r")
plt.show()

l = learning_rate_list

save_fun_list(l , "Learning_rate" + string)
plt.xlabel('Epochs')  
plt.ylabel('Learning rate')

plt.title('Learning rate_Graph') 
plt.plot(x, l,"b")
plt.show()


