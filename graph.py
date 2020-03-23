import csv
import numpy as np
import matplotlib.pyplot as plt

learning_rate_list = np.loadtxt('Case 1/layer2/Learning_rate_for_2_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')
cost_train_2 = np.loadtxt('Case 1/layer2/Training_loss_for_2_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')
cost_valid_2 = np.loadtxt('Case 1/layer2/Validation_loss_for_2_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')
cost_test_2 = np.loadtxt('Case 1/layer2/Test_loss_for_2_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')
acc_training_2 = np.loadtxt('Case 1/layer2/Training_acc_for_2_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')
acc_validation_2 = np.loadtxt('Case 1/layer2/Validation_acc_for_2_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')
acc_test_2 = np.loadtxt('Case 1/layer2/Test_acc_for_2_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')


cost_train_1 = np.loadtxt('Case 1/layer1/Training_loss_for_1_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')
cost_valid_1 = np.loadtxt('Case 1/layer1/Validation_loss_for_1_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')
cost_test_1 = np.loadtxt('Case 1/layer1/Test_loss_for_1_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')
acc_training_1 = np.loadtxt('Case 1/layer1/Training_acc_for_1_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')
acc_validation_1 = np.loadtxt('Case 1/layer1/Validation_acc_for_1_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')
acc_test_1 = np.loadtxt('Case 1/layer1/Test_acc_for_1_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')


cost_train_0 = np.loadtxt('Case 1/layer0/Training_loss_for_0_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')
cost_valid_0 = np.loadtxt('Case 1/layer0/Validation_loss_for_0_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')
cost_test_0 = np.loadtxt('Case 1/layer0/Test_loss_for_0_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')
acc_training_0 = np.loadtxt('Case 1/layer0/Training_acc_for_0_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')
acc_validation_0 = np.loadtxt('Case 1/layer0/Validation_acc_for_0_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')
acc_test_0 = np.loadtxt('Case 1/layer0/Test_acc_for_0_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')

#learning_rate = np.reshape(learning_rate,(10))
#print(learning_rate.shape)
#print(type(learning_rate))

x = np.arange(0,40, 1)

a = cost_train_0
#save_fun_list(y , "Training_loss" + string)
a1 = cost_valid_0
#save_fun_list(y1 , "Validation_loss" + string)
a2 = cost_test_0
#save_fun_list(y2 , "Test_loss" + string)


m = cost_train_1
#save_fun_list(y , "Training_loss" + string)
m1 = cost_valid_1
#save_fun_list(y1 , "Validation_loss" + string)
m2 = cost_test_1
#save_fun_list(y2 , "Test_loss" + string)


y = cost_train_2
#save_fun_list(y , "Training_loss" + string)
y1 = cost_valid_2
#save_fun_list(y1 , "Validation_loss" + string)
y2 = cost_test_2
#save_fun_list(y2 , "Test_loss" + string)

plt.xlabel('Epochs')  
plt.ylabel('Loss') 
 
plt.title('Loss_Graph for 2 hidden layers')
plt.plot(x, y,"b", label='Training Loss')
plt.plot(x, y1,"g", label='Validation Loss')
plt.plot(x, y2,"r", label='Test Loss')
#plt.plot(x, y,"b", x, y1,"g", x, y2,"r")
plt.legend()
plt.show()

b= acc_training_0
#save_fun_list(z , "Training_acc" + string)
b1 = acc_validation_0
#save_fun_list(z1 , "Validation_acc" + string)
b2 = acc_test_0
#save_fun_list(z2 , "Test_acc" + string)

n= acc_training_1
#save_fun_list(z , "Training_acc" + string)
n1 = acc_validation_1
#save_fun_list(z1 , "Validation_acc" + string)
n2 = acc_test_1
#save_fun_list(z2 , "Test_acc" + string)

z= acc_training_2
#save_fun_list(z , "Training_acc" + string)
z1 = acc_validation_2
#save_fun_list(z1 , "Validation_acc" + string)
z2 = acc_test_2
#save_fun_list(z2 , "Test_acc" + string)
plt.xlabel('Epochs')  
plt.ylabel('Accuracy')

plt.title('Accuracy Graph for 2 hidden layers')

plt.plot(x, z,"b", label='Training Accuracy')
plt.plot(x, z1,"g", label='Validation Accuracy')
plt.plot(x, z2,"r", label='Test Accuracy')
#plt.plot(x, z,"b", x, z1,"g", x, z2,"r")
plt.legend()
plt.show()

l = learning_rate_list

#save_fun_list(l , "Learning_rate" + string)
plt.xlabel('Epochs')  
plt.ylabel('Learning rate')

plt.title('Learning rate_Graph')
plt.plot(x, l,"b", label='Learning rate')
plt.legend()
plt.show()

plt.title('Training_Loss comparision')
plt.plot(x, a,"b", label='Training Loss 0 hidden layer')
plt.plot(x, m,"g", label='Training Loss 1 hidden layer')
plt.plot(x, y,"r", label='Training Loss 2 hidden layer')
#plt.plot(x, y,"b", x, y1,"g", x, y2,"r")
plt.legend()
plt.show()

plt.title('Validation_Loss comparision')
plt.plot(x, a1,"b", label='Validation Loss 0 hidden layer')
plt.plot(x, m1,"g", label='Validation Loss 1 hidden layer')
plt.plot(x, y1,"r", label='Validation Loss 2 hidden layer')
#plt.plot(x, y,"b", x, y1,"g", x, y2,"r")
plt.legend()
plt.show()

plt.title('Test_Loss comparision')
plt.plot(x, a2,"b", label='Test Loss 0 hidden layer')
plt.plot(x, m2,"g", label='Test Loss 1 hidden layer')
plt.plot(x, y2,"r", label='Test Loss 2 hidden layer')
#plt.plot(x, y,"b", x, y1,"g", x, y2,"r")
plt.legend()
plt.show()

plt.title('Training_Accuracy comparision')
plt.plot(x, b,"b", label='Training Accuracy 0 hidden layer')
plt.plot(x, n,"g", label='Training Accuracy 1 hidden layer')
plt.plot(x, z,"r", label='Training Accuracy 2 hidden layer')
#plt.plot(x, y,"b", x, y1,"g", x, y2,"r")
plt.legend()
plt.show()

plt.title('Validation_Accuracy comparision')
plt.plot(x, b1,"b", label='Validation Accuracy 0 hidden layer')
plt.plot(x, n1,"g", label='Validation Accuracy 1 hidden layer')
plt.plot(x, z1,"r", label='Validation Accuracy 2 hidden layer')
#plt.plot(x, y,"b", x, y1,"g", x, y2,"r")
plt.legend()
plt.show()

plt.title('Test_Accuracy comparision')
plt.plot(x, b2,"b", label='Test Accuracy 0 hidden layer')
plt.plot(x, n2,"g", label='Test Accuracy 1 hidden layer')
plt.plot(x, z2,"r", label='Test Accuracy 2 hidden layer')
#plt.plot(x, y,"b", x, y1,"g", x, y2,"r")
plt.legend()
plt.show()

plt.title('Loss comparision b/w define models ')
plt.plot(x, a, label='Training loss for 0 hidden layer')
plt.plot(x, a1, label='Validation loss 0 hidden layer')
plt.plot(x, a2, label='Test loss for 0 hidden layer')
plt.plot(x, m, label='Training loss for 1 hidden layer')
plt.plot(x, m1, label='Validation loss for 1 hidden layer')
plt.plot(x, m2, label='Test loss for 1 hidden layer')
plt.plot(x, y, label='Training loss for 2 hidden layer')
plt.plot(x, y1, label='Validation Loss for 2 hidden layer')
plt.plot(x, y2, label='Test Loss for 2 hidden layer')
#plt.plot(x, y,"b", x, y1,"g", x, y2,"r")
plt.legend()
plt.show()

plt.title('Accuracy comparision for defined models')
plt.plot(x, b, label='Training Acc for 0 hidden layer')
plt.plot(x, b1, label='Validation Acc 0 hidden layer')
plt.plot(x, b2, label='Test Acc for 0 hidden layer')
plt.plot(x, n, label='Training Acc for 1 hidden layer')
plt.plot(x, n1, label='Validation Acc for 1 hidden layer')
plt.plot(x, n2, label='Test Acc for 1 hidden layer')
plt.plot(x, z, label='Training Acc for 2 hidden layer')
plt.plot(x, z1, label='Validation Acc for 2 hidden layer')
plt.plot(x, z2, label='Test Acc for 2 hidden layer')
#plt.plot(x, y,"b", x, y1,"g", x, y2,"r")
plt.legend()
plt.show()

