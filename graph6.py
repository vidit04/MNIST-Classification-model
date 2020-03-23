import csv
import numpy as np
import matplotlib.pyplot as plt

learning_rate_list_org = np.loadtxt('Case 1/layer2/Learning_rate_for_2_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')
cost_train_2_org = np.loadtxt('Case 1/layer2/Training_loss_for_2_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')
cost_valid_2_org = np.loadtxt('Case 1/layer2/Validation_loss_for_2_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')
cost_test_2_org = np.loadtxt('Case 1/layer2/Test_loss_for_2_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')
acc_training_2_org = np.loadtxt('Case 1/layer2/Training_acc_for_2_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')
acc_validation_2_org = np.loadtxt('Case 1/layer2/Validation_acc_for_2_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')
acc_test_2_org = np.loadtxt('Case 1/layer2/Test_acc_for_2_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')

cost_train_2 = np.loadtxt('Case 6/layer2/Training_loss_for_2_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_y_dropout_with_sgd_optimizer.csv', delimiter=',')
cost_valid_2 = np.loadtxt('Case 6/layer2/Validation_loss_for_2_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_y_dropout_with_sgd_optimizer.csv', delimiter=',')
cost_test_2= np.loadtxt('Case 6/layer2/Test_loss_for_2_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_y_dropout_with_sgd_optimizer.csv', delimiter=',')
acc_training_2  = np.loadtxt('Case 6/layer2/Training_acc_for_2_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_y_dropout_with_sgd_optimizer.csv', delimiter=',')
acc_validation_2 = np.loadtxt('Case 6/layer2/Validation_acc_for_2_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_y_dropout_with_sgd_optimizer.csv', delimiter=',')
#acc_validation_2 = np.loadtxt('Case 3/layer2/Validation_acc_for_2_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_momentum_optimizer.csv', delimiter=',')
acc_test_2 = np.loadtxt('Case 6/layer2/Test_acc_for_2_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_y_dropout_with_sgd_optimizer.csv', delimiter=',')

cost_train_1_org = np.loadtxt('Case 1/layer1/Training_loss_for_1_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')
cost_valid_1_org = np.loadtxt('Case 1/layer1/Validation_loss_for_1_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')
cost_test_1_org = np.loadtxt('Case 1/layer1/Test_loss_for_1_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')
acc_training_1_org = np.loadtxt('Case 1/layer1/Training_acc_for_1_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')
acc_validation_1_org = np.loadtxt('Case 1/layer1/Validation_acc_for_1_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')
acc_test_1_org = np.loadtxt('Case 1/layer1/Test_acc_for_1_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')

#learning_rate_list = np.loadtxt('Case 3/layer1/.csv', delimiter=',')
cost_train_1 = np.loadtxt('Case 6/layer1/Training_loss_for_1_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_y_dropout_with_sgd_optimizer.csv', delimiter=',')
cost_valid_1 = np.loadtxt('Case 6/layer1/Validation_loss_for_1_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_y_dropout_with_sgd_optimizer.csv', delimiter=',')
cost_test_1 = np.loadtxt('Case 6/layer1/Test_loss_for_1_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_y_dropout_with_sgd_optimizer.csv', delimiter=',')
acc_training_1 = np.loadtxt('Case 6/layer1/Training_acc_for_1_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_y_dropout_with_sgd_optimizer.csv', delimiter=',')
acc_validation_1 = np.loadtxt('Case 6/layer1/Validation_acc_for_1_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_y_dropout_with_sgd_optimizer.csv', delimiter=',')
acc_test_1 = np.loadtxt('Case 6/layer1/Test_acc_for_1_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_y_dropout_with_sgd_optimizer.csv', delimiter=',')

cost_train_0_org = np.loadtxt('Case 1/layer0/Training_loss_for_0_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')
cost_valid_0_org = np.loadtxt('Case 1/layer0/Validation_loss_for_0_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')
cost_test_0_org = np.loadtxt('Case 1/layer0/Test_loss_for_0_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')
acc_training_0_org = np.loadtxt('Case 1/layer0/Training_acc_for_0_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')
acc_validation_0_org = np.loadtxt('Case 1/layer0/Validation_acc_for_0_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')
acc_test_0_org = np.loadtxt('Case 1/layer0/Test_acc_for_0_layer_relu_activation_normal_Normalization_Xavier_initalization_with_n_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')

#cost_train_0 = np.loadtxt('Case 5/layer0/Training_loss_for_0_layer_relu_activation_normal_Normalization_Xavier_initalization_with_y_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')
#cost_valid_0 = np.loadtxt('Case 5/layer0/Validation_loss_for_0_layer_relu_activation_normal_Normalization_Xavier_initalization_with_y_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')
#cost_test_0 = np.loadtxt('Case 5/layer0/Test_loss_for_0_layer_relu_activation_normal_Normalization_Xavier_initalization_with_y_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')
#acc_training_0 = np.loadtxt('Case 5/layer0/Training_acc_for_0_layer_relu_activation_normal_Normalization_Xavier_initalization_with_y_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')
#acc_validation_0 = np.loadtxt('Case 5/layer0/Validation_acc_for_0_layer_relu_activation_normal_Normalization_Xavier_initalization_with_y_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')
#acc_test_0 = np.loadtxt('Case 5/layer0/Test_acc_for_0_layer_relu_activation_normal_Normalization_Xavier_initalization_with_y_regression_with_n_learning_decay_with_n_dropout_with_sgd_optimizer.csv', delimiter=',')

#learning_rate = np.reshape(learning_rate,(10))
#print(learning_rate.shape)
#print(type(learning_rate))



x = np.arange(0,40, 1)

#a = cost_train_0
#save_fun_list(y , "Training_loss" + string)
#a1 = cost_valid_0
#save_fun_list(y1 , "Validation_loss" + string)
#a2 = cost_test_0
#save_fun_list(y2 , "Test_loss" + string)
aa = cost_train_0_org
#save_fun_list(y , "Training_loss" + string)
aa1 = cost_valid_0_org
#save_fun_list(y1 , "Validation_loss" + string)
aa2 = cost_test_0_org
#save_fun_list(y2 , "Test_loss" + string)

#a = cost_train_0
#save_fun_list(y , "Training_loss" + string)
#a1 = cost_valid_0
#save_fun_list(y1 , "Validation_loss" + string)
#a2 = cost_test_0
#save_fun_list(y2 , "Test_loss" + string)

#plt.xlabel('Epochs')  
#plt.ylabel('Loss') 
 
#plt.title('Comparision with & without L1 reg Loss_Graph for 0 hidden layers')
#plt.plot(x, aa, label='Training Loss with no L1 reg')
#plt.plot(x, aa1, label='Validation Loss with no L1 reg')
#plt.plot(x, aa2, label='Test Loss with no L1 reg')
#plt.plot(x, a, label='Training Loss with L1 reg')
#plt.plot(x, a1, label='Validation Loss with L1 reg')
#plt.plot(x, a2, label='Test Loss with L1 reg')
#plt.plot(x, y,"b", x, y1,"g", x, y2,"r")
#plt.legend()
#plt.show()



mm = cost_train_1_org
#save_fun_list(y , "Training_loss" + string)
mm1 = cost_valid_1_org
#save_fun_list(y1 , "Validation_loss" + string)
mm2 = cost_test_1_org
#save_fun_list(y2 , "Test_loss" + string)

m = cost_train_1
#save_fun_list(y , "Training_loss" + string)
m1 = cost_valid_1
#save_fun_list(y1 , "Validation_loss" + string)
m2 = cost_test_1
#save_fun_list(y2 , "Test_loss" + string)

plt.xlabel('Epochs')  
plt.ylabel('Loss') 
 
plt.title('Comparision with & without Dropout Loss_Graph for 1 hidden layers')
plt.plot(x, mm, label='Training Loss with no dropout')
plt.plot(x, mm1, label='Validation Loss with no dropout')
plt.plot(x, mm2, label='Test Loss with no dropout')
plt.plot(x, m, label='Training Loss with dropout')
plt.plot(x, m1, label='Validation Loss with dropout')
plt.plot(x, m2, label='Test Loss with dropout')
#plt.plot(x, y,"b", x, y1,"g", x, y2,"r")
plt.legend()
plt.show()



yy = cost_train_2_org
#save_fun_list(y , "Training_loss" + string)
yy1 = cost_valid_2_org
#save_fun_list(y1 , "Validation_loss" + string)
yy2 = cost_test_2_org
#save_fun_list(y2 , "Test_loss" + string)


y = cost_train_2
#save_fun_list(y , "Training_loss" + string)
y1 = cost_valid_2
#save_fun_list(y1 , "Validation_loss" + string)
y2 = cost_test_2
#save_fun_list(y2 , "Test_loss" + string)

plt.xlabel('Epochs')  
plt.ylabel('Loss') 
 
plt.title('Comparision with & without Dropout Loss_Graph for 2 hidden layers')
plt.plot(x, yy, label='Training Loss with no dropout')
plt.plot(x, yy1, label='Validation Loss with no dropout')
plt.plot(x, yy2, label='Test Loss with no dropout')
plt.plot(x, y, label='Training Loss with dropout')
plt.plot(x, y1, label='Validation Loss with dropout')
plt.plot(x, y2, label='Test Loss with dropout')
#plt.plot(x, y,"b", x, y1,"g", x, y2,"r")
plt.legend()
plt.show()


#b= acc_training_0
#save_fun_list(z , "Training_acc" + string)
#b1 = acc_validation_0
#save_fun_list(z1 , "Validation_acc" + string)
#b2 = acc_test_0
#save_fun_list(z2 , "Test_acc" + string)

bb= acc_training_0_org
#save_fun_list(z , "Training_acc" + string)
bb1 = acc_validation_0_org
#save_fun_list(z1 , "Validation_acc" + string)
bb2 = acc_test_0_org
#save_fun_list(z2 , "Test_acc" + string)

#b= acc_training_0
#save_fun_list(z , "Training_acc" + string)
#b1 = acc_validation_0
#save_fun_list(z1 , "Validation_acc" + string)
#b2 = acc_test_0
#save_fun_list(z2 , "Test_acc" + string)

#plt.xlabel('Epochs')  
#plt.ylabel('Loss') 
 
#plt.title('Comparision with & without Dropout Accuracy_Graph for 0 hidden layers')
#plt.plot(x, bb, label='Training Acc with no dropout')
#plt.plot(x, bb1, label='Validation Acc with no dropout')
#plt.plot(x, bb2, label='Test Acc with no dropout')
#plt.plot(x, b, label='Training Acc with dropout')
#plt.plot(x, b1, label='Validation Acc with dropout')
#plt.plot(x, b2, label='Test Acc with dropout')
#plt.plot(x, y,"b", x, y1,"g", x, y2,"r")
#plt.legend()
#plt.show()


nn= acc_training_1_org
#save_fun_list(z , "Training_acc" + string)
nn1 = acc_validation_1_org
#save_fun_list(z1 , "Validation_acc" + string)
nn2 = acc_test_1_org
#save_fun_list(z2 , "Test_acc" + string)

n= acc_training_1
#save_fun_list(z , "Training_acc" + string)
n1 = acc_validation_1
#save_fun_list(z1 , "Validation_acc" + string)
n2 = acc_test_1
#save_fun_list(z2 , "Test_acc" + string)

plt.xlabel('Epochs')  
plt.ylabel('Loss') 
 
plt.title('Comparision with & without Dropout Accuracy_Graph for 1 hidden layers')
plt.plot(x, nn, label='Training Acc with no dropout')
plt.plot(x, nn1, label='Validation Acc with no dropout')
plt.plot(x, nn2, label='Test Acc with no dropout')
plt.plot(x, n, label='Training Acc with dropout')
plt.plot(x, n1, label='Validation Acc with dropout')
plt.plot(x, n2, label='Test Acc with dropout')
#plt.plot(x, y,"b", x, y1,"g", x, y2,"r")
plt.legend()
plt.show()

zz= acc_training_2_org
#save_fun_list(z , "Training_acc" + string)
zz1 = acc_validation_2_org
#save_fun_list(z1 , "Validation_acc" + string)
zz2 = acc_test_2_org
#save_fun_list(z2 , "Test_acc" + string)


z= acc_training_2
#save_fun_list(z , "Training_acc" + string)
z1 = acc_validation_2
#save_fun_list(z1 , "Validation_acc" + string)
z2 = acc_test_2
#save_fun_list(z2 , "Test_acc" + string)

plt.xlabel('Epochs')  
plt.ylabel('Loss') 
 
plt.title('Comparision with & without Dropout Accuracy_Graph for 2 hidden layers')
plt.plot(x, zz, label='Training Acc with no dropout')
plt.plot(x, zz1, label='Validation Acc with no dropout')
plt.plot(x, zz2, label='Test Acc with no dropout')
plt.plot(x, z, label='Training Acc with dropout')
plt.plot(x, z1, label='Validation Acc with dropout')
plt.plot(x, z2, label='Test Acc with dropout')
#plt.plot(x, y,"b", x, y1,"g", x, y2,"r")
plt.legend()
plt.show()

plt.title('Loss comparision b/w models with Dropout')
#plt.plot(x, a, label='Training loss for 0 hidden layer')
#plt.plot(x, a1, label='Validation loss 0 hidden layer')
#plt.plot(x, a2, label='Test loss for 0 hidden layer')
plt.plot(x, m, label='Training loss for 1 hidden layer')
plt.plot(x, m1, label='Validation loss for 1 hidden layer')
plt.plot(x, m2, label='Test loss for 1 hidden layer')
plt.plot(x, y, label='Training loss for 2 hidden layer')
plt.plot(x, y1, label='Validation Loss for 2 hidden layer')
plt.plot(x, y2, label='Test Loss for 2 hidden layer')
#plt.plot(x, y,"b", x, y1,"g", x, y2,"r")
plt.legend()
plt.show()

plt.title('Accuracy comparision b/w models with Dropout')
#plt.plot(x, b, label='Training Acc for 0 hidden layer')
#plt.plot(x, b1, label='Validation Acc 0 hidden layer')
#plt.plot(x, b2, label='Test Acc for 0 hidden layer')
plt.plot(x, n, label='Training Acc for 1 hidden layer')
plt.plot(x, n1, label='Validation Acc for 1 hidden layer')
plt.plot(x, n2, label='Test Acc for 1 hidden layer')
plt.plot(x, z, label='Training Acc for 2 hidden layer')
plt.plot(x, z1, label='Validation Acc for 2 hidden layer')
plt.plot(x, z2, label='Test Acc for 2 hidden layer')
#plt.plot(x, y,"b", x, y1,"g", x, y2,"r")
plt.legend()
plt.show()

plt.title('Loss comparision b/w best orginal model & best model with Dropout')
#plt.plot(x, aa, label='Training loss for 0 hidden layer original model')
#plt.plot(x, aa1, label='Validation loss 0 hidden layer original model')
#plt.plot(x, aa2, label='Test loss for 0 hidden layer original model')

plt.plot(x, mm, label='Training loss for 1 hidden layer orginal model')
plt.plot(x, mm1, label='Validation loss for 1 hidden layer orginal model')
plt.plot(x, mm2, label='Test loss for 1 hidden layer orginal model')
#plt.plot(x, yy, label='Training loss for 2 hidden layer orginal model')
#plt.plot(x, yy1, label='Validation Loss for 2 hidden layer orginal model')
#plt.plot(x, yy2, label='Test Loss for 2 hidden layer orginal model')



#plt.plot(x, m, label='Training loss for 1 hidden layer')
#plt.plot(x, m1, label='Validation loss for 1 hidden layer')
#plt.plot(x, m2, label='Test loss for 1 hidden layer')
plt.plot(x, y, label='Training loss for 2 hidden layer with dropout')
plt.plot(x, y1, label='Validation Loss for 2 hidden layer with dropout')
plt.plot(x, y2, label='Test Loss for 2 hidden layer with dropout')
#plt.plot(x, y,"b", x, y1,"g", x, y2,"r")
plt.legend()
plt.show()

plt.title('Accuracy comparision b/w best original model & best model with Dropout')

#plt.plot(x, bb, label='Training Acc for 0 hidden layer original model')
#plt.plot(x, bb1, label='Validation Acc 0 hidden layer original model')
#plt.plot(x, bb2, label='Test Acc for 0 hidden layer original model')

plt.plot(x, nn, label='Training Acc for 1 hidden layer orginal model')
plt.plot(x, nn1, label='Validation Acc for 1 hidden layer orginal model')
plt.plot(x, nn2, label='Test Acc for 1 hidden layer orginal model')
#plt.plot(x, zz, label='Training Acc for 2 hidden layer orginal model')
#plt.plot(x, zz1, label='Validation Acc for 2 hidden layer orginal model')
#plt.plot(x, zz2, label='Test Acc for 2 hidden layer orginal model')


         
#plt.plot(x, n, label='Training Acc for 1 hidden layer')
#plt.plot(x, n1, label='Validation Acc for 1 hidden layer')
#plt.plot(x, n2, label='Test Acc for 1 hidden layer')
plt.plot(x, z, label='Training Acc for 2 hidden layer with dropout')
plt.plot(x, z1, label='Validation Acc for 2 hidden layer with dropout')
plt.plot(x, z2, label='Test Acc for 2 hidden layer with dropout')
#plt.plot(x, y,"b", x, y1,"g", x, y2,"r")
plt.legend()
plt.show()





