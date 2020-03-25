import xlrd
from train3 import*
import matplotlib.pyplot as plt

def relu_activation_test():

    print("Function relu actuvation is under testing")

    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet1')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    #print(total_rows)
    #print(total_cols)
    input_array = np.zeros((10,4),dtype = np.float32)
    output_array = np.zeros((10,4),dtype = np.float32)
    for i in range(10):
        for j in range(4):
            input_array[i,j] = data_sheet.cell_value(i+5,j+1)
            output_array[i,j] = data_sheet.cell_value(i+5,j+6)

    result_array = relu_activation(input_array)
    
    if (result_array == output_array).all() :
        print("Fuction tested okay")
    if ((result_array == output_array).all()) == False :
        print("Fuction tested not okay")
    return None

def relu_activation_back_test():

    print("Function relu actuvation backward is under testing")

    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet2')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    #print(total_rows)
    #print(total_cols)
    input_array_dloss_dA = np.zeros((10,4),dtype = np.float32)
    input_array_Z = np.zeros((10,4),dtype = np.float32)
    output_array = np.zeros((10,4),dtype = np.float32)

    for i in range(10):
        for j in range(4):
            input_array_Z[i,j] = data_sheet.cell_value(i+5,j+1)
            input_array_dloss_dA[i,j] = data_sheet.cell_value(i+19,j+1)
            output_array[i,j] = data_sheet.cell_value(i+5,j+6)

    result_array = relu_activation_back(input_array_dloss_dA,input_array_Z)
    
    if (result_array == output_array).all() :
        print("Fuction tested okay")
    if ((result_array == output_array).all()) == False :
        print("Fuction tested not okay")
    return None

def sigmoid_activation_test():

    print("Function sigmoid actuvation is under testing")

    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet3')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    #print(total_rows)
    #print(total_cols)
    input_array = np.zeros((10,4),dtype = np.float32)
    output_array = np.zeros((10,4),dtype = np.float32)
    for i in range(10):
        for j in range(4):
            input_array[i,j] = data_sheet.cell_value(i+5,j+1)
            output_array[i,j] = data_sheet.cell_value(i+5,j+6)
    output_array = np.round(output_array,5)
    result_array = sigmoid_activation(input_array)
    result_array = np.round(result_array,5)

    #print(result_array)
    if (result_array == output_array).all() :
        print("Fuction tested okay")
    if ((result_array == output_array).all()) == False :
        print("Fuction tested not okay")
    return None

def sigmoid_activation_back_test():
    print("Function sigmoid actuvation backward is under testing")

    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet4')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    #print(total_rows)
    #print(total_cols)
    input_array_dloss_dA = np.zeros((10,4),dtype = np.float32)
    input_array_Z = np.zeros((10,4),dtype = np.float32)
    output_array = np.zeros((10,4),dtype = np.float32)

    for i in range(10):
        for j in range(4):
            input_array_Z[i,j] = data_sheet.cell_value(i+5,j+1)
            input_array_dloss_dA[i,j] = data_sheet.cell_value(i+19,j+1)
            output_array[i,j] = data_sheet.cell_value(i+5,j+6)

    result_array = sigmoid_activation_back(input_array_dloss_dA,input_array_Z)
    output_array = np.round(output_array,6)
    result_array = np.round(result_array,6)
    #for i in range(10):
    #    for j in range(4):
    #        if output_array[i,j] != result_array[i,j]:
    #            print(i)
    #            print(j)
    #print(result_array)
    #print(output_array)

    if (result_array == output_array).all() :
        print("Fuction tested okay")
    if ((result_array == output_array).all()) == False :
        print("Fuction tested not okay")
    return None

def SGD_optimizer_test_for_2_layers():
    print("Function SGD optimizer 2 hidden layer is under testing")
    learning_rate = 0.001
    num=2

    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet5')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    #print(total_rows)
    #print(total_cols)
    input_weights_1 = np.zeros((10,10),dtype = np.float32)
    input_weights_2 = np.zeros((10,10),dtype = np.float32)
    input_weights_3 = np.zeros((10,10),dtype = np.float32)

    input_baises_1 = np.zeros((10,1),dtype = np.float32)
    input_baises_2 = np.zeros((10,1),dtype = np.float32)
    input_baises_3 = np.zeros((10,1),dtype = np.float32)

    input_dloss_dweights_1 = np.zeros((10,10),dtype = np.float32)
    input_dloss_dweights_2 = np.zeros((10,10),dtype = np.float32)
    input_dloss_dweights_3 = np.zeros((10,10),dtype = np.float32)

    input_dloss_dbaises_1 = np.zeros((10,1),dtype = np.float32)
    input_dloss_dbaises_2 = np.zeros((10,1),dtype = np.float32)
    input_dloss_dbaises_3 = np.zeros((10,1),dtype = np.float32)

    output_weights_1 = np.zeros((10,10),dtype = np.float32)
    output_weights_2 = np.zeros((10,10),dtype = np.float32)
    output_weights_3 = np.zeros((10,10),dtype = np.float32)

    output_baises_1 = np.zeros((10,1),dtype = np.float32)
    output_baises_2 = np.zeros((10,1),dtype = np.float32)
    output_baises_3 = np.zeros((10,1),dtype = np.float32)

    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_weights_1[i,j] = data_sheet.cell_value(i+15,j+1)
            input_dloss_dweights_1[i,j] = data_sheet.cell_value(i+15,j+13)
            output_weights_1[i,j] = data_sheet.cell_value(i+15,j+25)
            #print(i)
            #print(j)
            input_weights_2[i,j] = data_sheet.cell_value(i+33,j+1)
            input_dloss_dweights_2[i,j] = data_sheet.cell_value(i+33,j+13)
            output_weights_2[i,j] = data_sheet.cell_value(i+33,j+25)
            
            input_weights_3[i,j] = data_sheet.cell_value(i+51,j+1)
            input_dloss_dweights_3[i,j] = data_sheet.cell_value(i+51,j+13)
            output_weights_3[i,j] = data_sheet.cell_value(i+51,j+25)

    for i in range(10):
            input_baises_1[i,0] = data_sheet.cell_value(11,i+1)
            input_baises_2[i,0] = data_sheet.cell_value(29,i+1)
            input_baises_3[i,0] = data_sheet.cell_value(47,i+1)

            input_dloss_dbaises_1[i,0] = data_sheet.cell_value(11,i+13)
            input_dloss_dbaises_2[i,0] = data_sheet.cell_value(29,i+13)
            input_dloss_dbaises_3[i,0] = data_sheet.cell_value(47,i+13)
            
            output_baises_1[i,0] = data_sheet.cell_value(11,i+25)
            output_baises_2[i,0] = data_sheet.cell_value(29,i+25)
            output_baises_3[i,0] = data_sheet.cell_value(47,i+25)


    result_weights_1,result_baises_1,result_weights_2,result_baises_2,result_weights_3,result_baises_3 = SGD_optimizer(input_weights_1,input_baises_1,input_weights_2,input_baises_2,input_weights_3,input_baises_3,input_dloss_dweights_1,input_dloss_dbaises_1,input_dloss_dweights_2,input_dloss_dbaises_2,input_dloss_dweights_3, input_dloss_dbaises_3, learning_rate,num)

    result_weights_1 = np.round(result_weights_1 ,4)
    output_weights_1 = np.round(output_weights_1, 4)
    result_weights_2 = np.round(result_weights_2 ,4)
    output_weights_2 = np.round(output_weights_2, 4)
    result_weights_3 = np.round(result_weights_3 ,4)
    output_weights_3 = np.round(output_weights_3, 4)


    result_baises_1 = np.round(result_baises_1 ,4)
    output_baises_1 = np.round(output_baises_1, 4) 
    result_baises_2 = np.round(result_baises_2 ,4)
    output_baises_2 = np.round(output_baises_2, 4) 
    result_baises_3 = np.round(result_baises_3 ,4)
    output_baises_3 = np.round(output_baises_3, 4) 
    #print(result_baises_3)
    #print(output_baises_3)
    
    #if ((result_baises_3 == output_baises_3).all()):
    #    print("Fuction tested okay")
    #if ((result_baises_3 == output_baises_3).all()) == False:
    #    print("Fuction tested not okay")
        
    if (((result_weights_1 == output_weights_1).all()) and ((result_weights_2 == output_weights_2).all()) and ((result_weights_3 == output_weights_3).all()) and ((result_baises_1 == output_baises_1).all()) and ((result_baises_2 == output_baises_2).all()) and ((result_baises_3 == output_baises_3).all())):
        print("Fuction tested okay")
    if (((result_weights_1 == output_weights_1).all()) and ((result_weights_2 == output_weights_2).all()) and ((result_weights_3 == output_weights_3).all()) and ((result_baises_1 == output_baises_1).all()) and ((result_baises_2 == output_baises_2).all()) and ((result_baises_3 == output_baises_3).all())) == False:
        print("Fuction tested not okay")
    return None

def SGD_optimizer_test_for_1_layers():

    print("Function SGD optimizer one hidden layer is under testing")
    learning_rate = 0.001
    num =1

    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet5')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    #print(total_rows)
    #print(total_cols)
    input_weights_1 = np.zeros((10,10),dtype = np.float32)
    input_weights_2 = np.zeros((10,10),dtype = np.float32)
    input_weights_3 = 0

    input_baises_1 = np.zeros((10,1),dtype = np.float32)
    input_baises_2 = np.zeros((10,1),dtype = np.float32)
    input_baises_3 = 0

    input_dloss_dweights_1 = np.zeros((10,10),dtype = np.float32)
    input_dloss_dweights_2 = np.zeros((10,10),dtype = np.float32)
    input_dloss_dweights_3 = 0

    input_dloss_dbaises_1 = np.zeros((10,1),dtype = np.float32)
    input_dloss_dbaises_2 = np.zeros((10,1),dtype = np.float32)
    input_dloss_dbaises_3 = 0

    output_weights_1 = np.zeros((10,10),dtype = np.float32)
    output_weights_2 = np.zeros((10,10),dtype = np.float32)
    #output_weights_3 = np.zeros((10,10),dtype = np.float32)

    output_baises_1 = np.zeros((10,1),dtype = np.float32)
    output_baises_2 = np.zeros((10,1),dtype = np.float32)
    #output_baises_3 = np.zeros((10,1),dtype = np.float32)

    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_weights_1[i,j] = data_sheet.cell_value(i+15,j+1)
            input_dloss_dweights_1[i,j] = data_sheet.cell_value(i+15,j+13)
            output_weights_1[i,j] = data_sheet.cell_value(i+15,j+25)
            #print(i)
            #print(j)
            input_weights_2[i,j] = data_sheet.cell_value(i+33,j+1)
            input_dloss_dweights_2[i,j] = data_sheet.cell_value(i+33,j+13)
            output_weights_2[i,j] = data_sheet.cell_value(i+33,j+25)
            
            #input_weights_3[i,j] = data_sheet.cell_value(i+51,j+1)
            #input_dloss_dweights_3[i,j] = data_sheet.cell_value(i+51,j+13)
            #output_weights_3[i,j] = data_sheet.cell_value(i+51,j+25)

    for i in range(10):
            input_baises_1[i,0] = data_sheet.cell_value(11,i+1)
            input_baises_2[i,0] = data_sheet.cell_value(29,i+1)
            #input_baises_3[i,0] = data_sheet.cell_value(47,i+1)

            input_dloss_dbaises_1[i,0] = data_sheet.cell_value(11,i+13)
            input_dloss_dbaises_2[i,0] = data_sheet.cell_value(29,i+13)
            #input_dloss_dbaises_3[i,0] = data_sheet.cell_value(47,i+13)
            
            output_baises_1[i,0] = data_sheet.cell_value(11,i+25)
            output_baises_2[i,0] = data_sheet.cell_value(29,i+25)
            #output_baises_3[i,0] = data_sheet.cell_value(47,i+25)


    result_weights_1,result_baises_1,result_weights_2,result_baises_2 = SGD_optimizer(input_weights_1,input_baises_1,input_weights_2,input_baises_2,input_weights_3,input_baises_3,input_dloss_dweights_1,input_dloss_dbaises_1,input_dloss_dweights_2,input_dloss_dbaises_2,input_dloss_dweights_3, input_dloss_dbaises_3, learning_rate,num)

    result_weights_1 = np.round(result_weights_1 ,4)
    output_weights_1 = np.round(output_weights_1, 4)
    result_weights_2 = np.round(result_weights_2 ,4)
    output_weights_2 = np.round(output_weights_2, 4)
    #result_weights_3 = np.round(result_weights_3 ,4)
    #output_weights_3 = np.round(output_weights_3, 4)


    result_baises_1 = np.round(result_baises_1 ,4)
    output_baises_1 = np.round(output_baises_1, 4) 
    result_baises_2 = np.round(result_baises_2 ,4)
    output_baises_2 = np.round(output_baises_2, 4) 
    #result_baises_3 = np.round(result_baises_3 ,4)
    #output_baises_3 = np.round(output_baises_3, 4) 
    #print(result_baises_3)
    #print(output_baises_3)
    
    #if ((result_baises_3 == output_baises_3).all()):
    #    print("Fuction tested okay")
    #if ((result_baises_3 == output_baises_3).all()) == False:
    #    print("Fuction tested not okay")
        
    if (((result_weights_1 == output_weights_1).all()) and ((result_weights_2 == output_weights_2).all()) and ((result_baises_1 == output_baises_1).all()) and ((result_baises_2 == output_baises_2).all())):
        print("Fuction tested okay")
    if (((result_weights_1 == output_weights_1).all()) and ((result_weights_2 == output_weights_2).all()) and ((result_baises_1 == output_baises_1).all()) and ((result_baises_2 == output_baises_2).all())) == False:
        print("Fuction tested not okay")
    return None

def SGD_optimizer_test_for_0_layers():
    print("Function SGD optimizer zero hidden layer is under testing")
    learning_rate = 0.001
    num =0

    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet5')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    #print(total_rows)
    #print(total_cols)
    input_weights_1 = np.zeros((10,10),dtype = np.float32)
    input_weights_2 = 0
    input_weights_3 = 0

    input_baises_1 = np.zeros((10,1),dtype = np.float32)
    input_baises_2 = 0
    input_baises_3 = 0

    input_dloss_dweights_1 = np.zeros((10,10),dtype = np.float32)
    input_dloss_dweights_2 = 0
    input_dloss_dweights_3 = 0

    input_dloss_dbaises_1 = np.zeros((10,1),dtype = np.float32)
    input_dloss_dbaises_2 = 0
    input_dloss_dbaises_3 = 0

    output_weights_1 = np.zeros((10,10),dtype = np.float32)
    #output_weights_2 = np.zeros((10,10),dtype = np.float32)
    #output_weights_3 = np.zeros((10,10),dtype = np.float32)

    output_baises_1 = np.zeros((10,1),dtype = np.float32)
    #output_baises_2 = np.zeros((10,1),dtype = np.float32)
    #output_baises_3 = np.zeros((10,1),dtype = np.float32)

    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_weights_1[i,j] = data_sheet.cell_value(i+15,j+1)
            input_dloss_dweights_1[i,j] = data_sheet.cell_value(i+15,j+13)
            output_weights_1[i,j] = data_sheet.cell_value(i+15,j+25)
            #print(i)
            #print(j)
            #input_weights_2[i,j] = data_sheet.cell_value(i+33,j+1)
            #input_dloss_dweights_2[i,j] = data_sheet.cell_value(i+33,j+13)
            #output_weights_2[i,j] = data_sheet.cell_value(i+33,j+25)
            
            #input_weights_3[i,j] = data_sheet.cell_value(i+51,j+1)
            #input_dloss_dweights_3[i,j] = data_sheet.cell_value(i+51,j+13)
            #output_weights_3[i,j] = data_sheet.cell_value(i+51,j+25)

    for i in range(10):
            input_baises_1[i,0] = data_sheet.cell_value(11,i+1)
            #input_baises_2[i,0] = data_sheet.cell_value(29,i+1)
            #input_baises_3[i,0] = data_sheet.cell_value(47,i+1)

            input_dloss_dbaises_1[i,0] = data_sheet.cell_value(11,i+13)
            #input_dloss_dbaises_2[i,0] = data_sheet.cell_value(29,i+13)
            #input_dloss_dbaises_3[i,0] = data_sheet.cell_value(47,i+13)
            
            output_baises_1[i,0] = data_sheet.cell_value(11,i+25)
            #output_baises_2[i,0] = data_sheet.cell_value(29,i+25)
            #output_baises_3[i,0] = data_sheet.cell_value(47,i+25)


    result_weights_1,result_baises_1= SGD_optimizer(input_weights_1,input_baises_1,input_weights_2,input_baises_2,input_weights_3,input_baises_3,input_dloss_dweights_1,input_dloss_dbaises_1,input_dloss_dweights_2,input_dloss_dbaises_2,input_dloss_dweights_3, input_dloss_dbaises_3, learning_rate,num)

    result_weights_1 = np.round(result_weights_1 ,4)
    output_weights_1 = np.round(output_weights_1, 4)
    #result_weights_2 = np.round(result_weights_2 ,4)
    #output_weights_2 = np.round(output_weights_2, 4)
    #result_weights_3 = np.round(result_weights_3 ,4)
    #output_weights_3 = np.round(output_weights_3, 4)


    result_baises_1 = np.round(result_baises_1 ,4)
    output_baises_1 = np.round(output_baises_1, 4) 
    #result_baises_2 = np.round(result_baises_2 ,4)
    #output_baises_2 = np.round(output_baises_2, 4) 
    #result_baises_3 = np.round(result_baises_3 ,4)
    #output_baises_3 = np.round(output_baises_3, 4) 
    #print(result_baises_3)
    #print(output_baises_3)
    
    #if ((result_baises_3 == output_baises_3).all()):
    #    print("Fuction tested okay")
    #if ((result_baises_3 == output_baises_3).all()) == False:
    #    print("Fuction tested not okay")
        
    if (((result_weights_1 == output_weights_1).all()) and ((result_baises_1 == output_baises_1).all())):
        print("Fuction tested okay")
    if ((((result_weights_1 == output_weights_1).all()) and ((result_baises_1 == output_baises_1).all()))) == False:
        print("Fuction tested not okay")
    return None

def Momentum_optimizer_test_for_2_layers():

    print("Function Momentum optimizer two hidden layer is under testing")
    learning_rate = 0.001
    beta = 0.9
    num = 2
    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet6')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    #print(total_rows)
    #print(total_cols)
    input_weights_1 = np.zeros((10,10),dtype = np.float32)
    input_weights_2 = np.zeros((10,10),dtype = np.float32)
    input_weights_3 = np.zeros((10,10),dtype = np.float32)

    input_baises_1 = np.zeros((10,1),dtype = np.float32)
    input_baises_2 = np.zeros((10,1),dtype = np.float32)
    input_baises_3 = np.zeros((10,1),dtype = np.float32)

    input_dloss_dweights_1 = np.zeros((10,10),dtype = np.float32)
    input_dloss_dweights_2 = np.zeros((10,10),dtype = np.float32)
    input_dloss_dweights_3 = np.zeros((10,10),dtype = np.float32)

    input_dloss_dbaises_1 = np.zeros((10,1),dtype = np.float32)
    input_dloss_dbaises_2 = np.zeros((10,1),dtype = np.float32)
    input_dloss_dbaises_3 = np.zeros((10,1),dtype = np.float32)

    mov_weights_1 = np.zeros((10,10),dtype = np.float32)
    mov_weights_2 = np.zeros((10,10),dtype = np.float32)
    mov_weights_3 = np.zeros((10,10),dtype = np.float32)

    mov_baises_1 = np.zeros((10,1),dtype = np.float32)
    mov_baises_2 = np.zeros((10,1),dtype = np.float32)
    mov_baises_3 = np.zeros((10,1),dtype = np.float32)

    output_weights_1 = np.zeros((10,10),dtype = np.float32)
    output_weights_2 = np.zeros((10,10),dtype = np.float32)
    output_weights_3 = np.zeros((10,10),dtype = np.float32)

    output_baises_1 = np.zeros((10,1),dtype = np.float32)
    output_baises_2 = np.zeros((10,1),dtype = np.float32)
    output_baises_3 = np.zeros((10,1),dtype = np.float32)

    output_mov_weights_1 = np.zeros((10,10),dtype = np.float32)
    output_mov_weights_2 = np.zeros((10,10),dtype = np.float32)
    output_mov_weights_3 = np.zeros((10,10),dtype = np.float32)

    output_mov_baises_1 = np.zeros((10,1),dtype = np.float32)
    output_mov_baises_2 = np.zeros((10,1),dtype = np.float32)
    output_mov_baises_3 = np.zeros((10,1),dtype = np.float32)

    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_weights_1[i,j] = data_sheet.cell_value(i+15,j+1)
            mov_weights_1[i,j] = data_sheet.cell_value(i+15,j+25)
            input_dloss_dweights_1[i,j] = data_sheet.cell_value(i+15,j+13)
            output_mov_weights_1[i,j] = data_sheet.cell_value(i+15,j+37)
            output_weights_1[i,j] = data_sheet.cell_value(i+15,j+49)
            #print(i)
            #print(j)
            input_weights_2[i,j] = data_sheet.cell_value(i+33,j+1)
            mov_weights_2[i,j] = data_sheet.cell_value(i+33,j+25)
            input_dloss_dweights_2[i,j] = data_sheet.cell_value(i+33,j+13)
            output_mov_weights_2[i,j] = data_sheet.cell_value(i+33,j+37)
            output_weights_2[i,j] = data_sheet.cell_value(i+33,j+49)
            
            input_weights_3[i,j] = data_sheet.cell_value(i+51,j+1)
            mov_weights_3[i,j] = data_sheet.cell_value(i+51,j+25)
            input_dloss_dweights_3[i,j] = data_sheet.cell_value(i+51,j+13)
            output_mov_weights_3[i,j] = data_sheet.cell_value(i+51,j+37)
            output_weights_3[i,j] = data_sheet.cell_value(i+51,j+49)

    for i in range(10):
            input_baises_1[i,0] = data_sheet.cell_value(11,i+1)
            input_baises_2[i,0] = data_sheet.cell_value(29,i+1)
            input_baises_3[i,0] = data_sheet.cell_value(47,i+1)

            input_dloss_dbaises_1[i,0] = data_sheet.cell_value(11,i+13)
            input_dloss_dbaises_2[i,0] = data_sheet.cell_value(29,i+13)
            input_dloss_dbaises_3[i,0] = data_sheet.cell_value(47,i+13)

            mov_baises_1[i,0] = data_sheet.cell_value(11,i+25)
            mov_baises_2[i,0] = data_sheet.cell_value(29,i+25)
            mov_baises_3[i,0] = data_sheet.cell_value(47,i+25)

            output_mov_baises_1[i,0] = data_sheet.cell_value(11,i+37)
            output_mov_baises_2[i,0] = data_sheet.cell_value(29,i+37)
            output_mov_baises_3[i,0] = data_sheet.cell_value(47,i+37)
            
            output_baises_1[i,0] = data_sheet.cell_value(11,i+49)
            output_baises_2[i,0] = data_sheet.cell_value(29,i+49)
            output_baises_3[i,0] = data_sheet.cell_value(47,i+49)


    result_weights_1,result_baises_1,result_weights_2,result_baises_2,result_weights_3,result_baises_3,result_mov_weights_1,result_mov_baises_1,result_mov_weights_2,result_mov_baises_2,result_mov_weights_3,result_mov_baises_3 = Momentum_optimizer(input_weights_1,input_baises_1,input_weights_2,input_baises_2,input_weights_3,input_baises_3,mov_weights_1,mov_baises_1,mov_weights_2,mov_baises_2,mov_weights_3,mov_baises_3,input_dloss_dweights_1,input_dloss_dbaises_1,input_dloss_dweights_2,input_dloss_dbaises_2,input_dloss_dweights_3, input_dloss_dbaises_3, learning_rate, beta,num)

    result_weights_1 = np.round(result_weights_1 ,4)
    output_weights_1 = np.round(output_weights_1, 4)
    result_weights_2 = np.round(result_weights_2 ,4)
    output_weights_2 = np.round(output_weights_2, 4)
    result_weights_3 = np.round(result_weights_3 ,4)
    output_weights_3 = np.round(output_weights_3, 4)


    result_baises_1 = np.round(result_baises_1 ,4)
    output_baises_1 = np.round(output_baises_1, 4) 
    result_baises_2 = np.round(result_baises_2 ,4)
    output_baises_2 = np.round(output_baises_2, 4) 
    result_baises_3 = np.round(result_baises_3 ,4)
    output_baises_3 = np.round(output_baises_3, 4)

    result_mov_weights_1 = np.round(result_mov_weights_1 ,4)
    output_mov_weights_1 = np.round(output_mov_weights_1, 4)
    result_mov_weights_2 = np.round(result_mov_weights_2 ,4)
    output_mov_weights_2 = np.round(output_mov_weights_2, 4)
    result_mov_weights_3 = np.round(result_mov_weights_3 ,4)
    output_mov_weights_3 = np.round(output_mov_weights_3, 4)


    result_mov_baises_1 = np.round(result_mov_baises_1 ,4)
    output_mov_baises_1 = np.round(output_mov_baises_1, 4) 
    result_mov_baises_2 = np.round(result_mov_baises_2 ,4)
    output_mov_baises_2 = np.round(output_mov_baises_2, 4) 
    result_mov_baises_3 = np.round(result_mov_baises_3 ,4)
    output_mov_baises_3 = np.round(output_mov_baises_3, 4)
    #print(result_baises_3)
    #print(output_baises_3)
    
    #if ((result_baises_3 == output_baises_3).all()):
    #    print("Fuction tested okay")
    #if ((result_baises_3 == output_baises_3).all()) == False:
    #    print("Fuction tested not okay")
        
    if (((result_weights_1 == output_weights_1).all()) and ((result_weights_2 == output_weights_2).all()) and ((result_weights_3 == output_weights_3).all()) and ((result_baises_1 == output_baises_1).all()) and ((result_baises_2 == output_baises_2).all()) and ((result_baises_3 == output_baises_3).all()) and ((result_mov_weights_1 == output_mov_weights_1).all()) and ((result_mov_weights_2 == output_mov_weights_2).all()) and ((result_mov_weights_3 == output_mov_weights_3).all()) and ((result_mov_baises_1 == output_mov_baises_1).all()) and ((result_mov_baises_2 == output_mov_baises_2).all()) and ((result_mov_baises_3 == output_mov_baises_3).all())):
        print("Fuction tested okay")
    if (((result_weights_1 == output_weights_1).all()) and ((result_weights_2 == output_weights_2).all()) and ((result_weights_3 == output_weights_3).all()) and ((result_baises_1 == output_baises_1).all()) and ((result_baises_2 == output_baises_2).all()) and ((result_baises_3 == output_baises_3).all()) and ((result_mov_weights_1 == output_mov_weights_1).all()) and ((result_mov_weights_2 == output_mov_weights_2).all()) and ((result_mov_weights_3 == output_mov_weights_3).all()) and ((result_mov_baises_1 == output_mov_baises_1).all()) and ((result_mov_baises_2 == output_mov_baises_2).all()) and ((result_mov_baises_3 == output_mov_baises_3).all())) == False:
        print("Fuction tested not okay")
    return None

def Momentum_optimizer_test_for_1_layers():

    print("Function Momentum optimizer 1 hidden layer is under testing")
    learning_rate = 0.001
    beta = 0.9
    num = 1
    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet6')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    #print(total_rows)
    #print(total_cols)
    input_weights_1 = np.zeros((10,10),dtype = np.float32)
    input_weights_2 = np.zeros((10,10),dtype = np.float32)
    input_weights_3 = 0

    input_baises_1 = np.zeros((10,1),dtype = np.float32)
    input_baises_2 = np.zeros((10,1),dtype = np.float32)
    input_baises_3 = 0

    input_dloss_dweights_1 = np.zeros((10,10),dtype = np.float32)
    input_dloss_dweights_2 = np.zeros((10,10),dtype = np.float32)
    input_dloss_dweights_3 = 0

    input_dloss_dbaises_1 = np.zeros((10,1),dtype = np.float32)
    input_dloss_dbaises_2 = np.zeros((10,1),dtype = np.float32)
    input_dloss_dbaises_3 = 0

    mov_weights_1 = np.zeros((10,10),dtype = np.float32)
    mov_weights_2 = np.zeros((10,10),dtype = np.float32)
    mov_weights_3 = 0

    mov_baises_1 = np.zeros((10,1),dtype = np.float32)
    mov_baises_2 = np.zeros((10,1),dtype = np.float32)
    mov_baises_3 = 0

    output_weights_1 = np.zeros((10,10),dtype = np.float32)
    output_weights_2 = np.zeros((10,10),dtype = np.float32)
    #output_weights_3 = np.zeros((10,10),dtype = np.float32)

    output_baises_1 = np.zeros((10,1),dtype = np.float32)
    output_baises_2 = np.zeros((10,1),dtype = np.float32)
    #output_baises_3 = np.zeros((10,1),dtype = np.float32)

    output_mov_weights_1 = np.zeros((10,10),dtype = np.float32)
    output_mov_weights_2 = np.zeros((10,10),dtype = np.float32)
    #output_mov_weights_3 = np.zeros((10,10),dtype = np.float32)

    output_mov_baises_1 = np.zeros((10,1),dtype = np.float32)
    output_mov_baises_2 = np.zeros((10,1),dtype = np.float32)
    #output_mov_baises_3 = np.zeros((10,1),dtype = np.float32)

    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_weights_1[i,j] = data_sheet.cell_value(i+15,j+1)
            mov_weights_1[i,j] = data_sheet.cell_value(i+15,j+25)
            input_dloss_dweights_1[i,j] = data_sheet.cell_value(i+15,j+13)
            output_mov_weights_1[i,j] = data_sheet.cell_value(i+15,j+37)
            output_weights_1[i,j] = data_sheet.cell_value(i+15,j+49)
            #print(i)
            #print(j)
            input_weights_2[i,j] = data_sheet.cell_value(i+33,j+1)
            mov_weights_2[i,j] = data_sheet.cell_value(i+33,j+25)
            input_dloss_dweights_2[i,j] = data_sheet.cell_value(i+33,j+13)
            output_mov_weights_2[i,j] = data_sheet.cell_value(i+33,j+37)
            output_weights_2[i,j] = data_sheet.cell_value(i+33,j+49)
            
            #input_weights_3[i,j] = data_sheet.cell_value(i+51,j+1)
            #mov_weights_3[i,j] = data_sheet.cell_value(i+51,j+25)
            #input_dloss_dweights_3[i,j] = data_sheet.cell_value(i+51,j+13)
            #output_mov_weights_3[i,j] = data_sheet.cell_value(i+51,j+37)
            #output_weights_3[i,j] = data_sheet.cell_value(i+51,j+49)

    for i in range(10):
            input_baises_1[i,0] = data_sheet.cell_value(11,i+1)
            input_baises_2[i,0] = data_sheet.cell_value(29,i+1)
            #input_baises_3[i,0] = data_sheet.cell_value(47,i+1)

            input_dloss_dbaises_1[i,0] = data_sheet.cell_value(11,i+13)
            input_dloss_dbaises_2[i,0] = data_sheet.cell_value(29,i+13)
            #input_dloss_dbaises_3[i,0] = data_sheet.cell_value(47,i+13)

            mov_baises_1[i,0] = data_sheet.cell_value(11,i+25)
            mov_baises_2[i,0] = data_sheet.cell_value(29,i+25)
            #mov_baises_3[i,0] = data_sheet.cell_value(47,i+25)

            output_mov_baises_1[i,0] = data_sheet.cell_value(11,i+37)
            output_mov_baises_2[i,0] = data_sheet.cell_value(29,i+37)
            #output_mov_baises_3[i,0] = data_sheet.cell_value(47,i+37)
            
            output_baises_1[i,0] = data_sheet.cell_value(11,i+49)
            output_baises_2[i,0] = data_sheet.cell_value(29,i+49)
            #output_baises_3[i,0] = data_sheet.cell_value(47,i+49)


    result_weights_1,result_baises_1,result_weights_2,result_baises_2,result_mov_weights_1,result_mov_baises_1,result_mov_weights_2,result_mov_baises_2 = Momentum_optimizer(input_weights_1,input_baises_1,input_weights_2,input_baises_2,input_weights_3,input_baises_3,mov_weights_1,mov_baises_1,mov_weights_2,mov_baises_2,mov_weights_3,mov_baises_3,input_dloss_dweights_1,input_dloss_dbaises_1,input_dloss_dweights_2,input_dloss_dbaises_2,input_dloss_dweights_3, input_dloss_dbaises_3, learning_rate, beta,num)

    result_weights_1 = np.round(result_weights_1 ,4)
    output_weights_1 = np.round(output_weights_1, 4)
    result_weights_2 = np.round(result_weights_2 ,4)
    output_weights_2 = np.round(output_weights_2, 4)
    #result_weights_3 = np.round(result_weights_3 ,4)
    #output_weights_3 = np.round(output_weights_3, 4)


    result_baises_1 = np.round(result_baises_1 ,4)
    output_baises_1 = np.round(output_baises_1, 4) 
    result_baises_2 = np.round(result_baises_2 ,4)
    output_baises_2 = np.round(output_baises_2, 4) 
    #result_baises_3 = np.round(result_baises_3 ,4)
    #output_baises_3 = np.round(output_baises_3, 4)

    result_mov_weights_1 = np.round(result_mov_weights_1 ,4)
    output_mov_weights_1 = np.round(output_mov_weights_1, 4)
    result_mov_weights_2 = np.round(result_mov_weights_2 ,4)
    output_mov_weights_2 = np.round(output_mov_weights_2, 4)
    #result_mov_weights_3 = np.round(result_mov_weights_3 ,4)
    #output_mov_weights_3 = np.round(output_mov_weights_3, 4)


    result_mov_baises_1 = np.round(result_mov_baises_1 ,4)
    output_mov_baises_1 = np.round(output_mov_baises_1, 4) 
    result_mov_baises_2 = np.round(result_mov_baises_2 ,4)
    output_mov_baises_2 = np.round(output_mov_baises_2, 4) 
    #result_mov_baises_3 = np.round(result_mov_baises_3 ,4)
    #output_mov_baises_3 = np.round(output_mov_baises_3, 4)
    #print(result_baises_3)
    #print(output_baises_3)
    
    #if ((result_baises_3 == output_baises_3).all()):
    #    print("Fuction tested okay")
    #if ((result_baises_3 == output_baises_3).all()) == False:
    #    print("Fuction tested not okay")
        
    if (((result_weights_1 == output_weights_1).all()) and ((result_weights_2 == output_weights_2).all()) and  ((result_baises_1 == output_baises_1).all()) and ((result_baises_2 == output_baises_2).all()) and  ((result_mov_weights_1 == output_mov_weights_1).all()) and ((result_mov_weights_2 == output_mov_weights_2).all()) and ((result_mov_baises_1 == output_mov_baises_1).all()) and ((result_mov_baises_2 == output_mov_baises_2).all())):
        print("Fuction tested okay")
    if (((result_weights_1 == output_weights_1).all()) and ((result_weights_2 == output_weights_2).all()) and  ((result_baises_1 == output_baises_1).all()) and ((result_baises_2 == output_baises_2).all()) and  ((result_mov_weights_1 == output_mov_weights_1).all()) and ((result_mov_weights_2 == output_mov_weights_2).all()) and ((result_mov_baises_1 == output_mov_baises_1).all()) and ((result_mov_baises_2 == output_mov_baises_2).all())) == False:
        print("Fuction tested not okay")
    return None

def Momentum_optimizer_test_for_0_layers():

    print("Function Momentum optimizer zero hidden layer is under testing")
    learning_rate = 0.001
    beta = 0.9
    num = 0
    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet6')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    #print(total_rows)
    #print(total_cols)
    input_weights_1 = np.zeros((10,10),dtype = np.float32)
    input_weights_2 = 0
    input_weights_3 = 0

    input_baises_1 = np.zeros((10,1),dtype = np.float32)
    input_baises_2 = 0
    input_baises_3 = 0

    input_dloss_dweights_1 = np.zeros((10,10),dtype = np.float32)
    input_dloss_dweights_2 = 0
    input_dloss_dweights_3 = 0

    input_dloss_dbaises_1 = np.zeros((10,1),dtype = np.float32)
    input_dloss_dbaises_2 = 0
    input_dloss_dbaises_3 = 0

    mov_weights_1 = np.zeros((10,10),dtype = np.float32)
    mov_weights_2 = 0
    mov_weights_3 = 0

    mov_baises_1 = np.zeros((10,1),dtype = np.float32)
    mov_baises_2 = 0
    mov_baises_3 = 0

    output_weights_1 = np.zeros((10,10),dtype = np.float32)
    #output_weights_2 = np.zeros((10,10),dtype = np.float32)
    #output_weights_3 = np.zeros((10,10),dtype = np.float32)

    output_baises_1 = np.zeros((10,1),dtype = np.float32)
    #output_baises_2 = np.zeros((10,1),dtype = np.float32)
    #output_baises_3 = np.zeros((10,1),dtype = np.float32)

    output_mov_weights_1 = np.zeros((10,10),dtype = np.float32)
    #output_mov_weights_2 = np.zeros((10,10),dtype = np.float32)
    #output_mov_weights_3 = np.zeros((10,10),dtype = np.float32)

    output_mov_baises_1 = np.zeros((10,1),dtype = np.float32)
    #output_mov_baises_2 = np.zeros((10,1),dtype = np.float32)
    #output_mov_baises_3 = np.zeros((10,1),dtype = np.float32)

    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_weights_1[i,j] = data_sheet.cell_value(i+15,j+1)
            mov_weights_1[i,j] = data_sheet.cell_value(i+15,j+25)
            input_dloss_dweights_1[i,j] = data_sheet.cell_value(i+15,j+13)
            output_mov_weights_1[i,j] = data_sheet.cell_value(i+15,j+37)
            output_weights_1[i,j] = data_sheet.cell_value(i+15,j+49)
            #print(i)
            #print(j)
            #input_weights_2[i,j] = data_sheet.cell_value(i+33,j+1)
            #mov_weights_2[i,j] = data_sheet.cell_value(i+33,j+25)
            #input_dloss_dweights_2[i,j] = data_sheet.cell_value(i+33,j+13)
            #output_mov_weights_2[i,j] = data_sheet.cell_value(i+33,j+37)
            #output_weights_2[i,j] = data_sheet.cell_value(i+33,j+49)
            
            #input_weights_3[i,j] = data_sheet.cell_value(i+51,j+1)
            #mov_weights_3[i,j] = data_sheet.cell_value(i+51,j+25)
            #input_dloss_dweights_3[i,j] = data_sheet.cell_value(i+51,j+13)
            #output_mov_weights_3[i,j] = data_sheet.cell_value(i+51,j+37)
            #output_weights_3[i,j] = data_sheet.cell_value(i+51,j+49)

    for i in range(10):
            input_baises_1[i,0] = data_sheet.cell_value(11,i+1)
            #input_baises_2[i,0] = data_sheet.cell_value(29,i+1)
            #input_baises_3[i,0] = data_sheet.cell_value(47,i+1)

            input_dloss_dbaises_1[i,0] = data_sheet.cell_value(11,i+13)
            #input_dloss_dbaises_2[i,0] = data_sheet.cell_value(29,i+13)
            #input_dloss_dbaises_3[i,0] = data_sheet.cell_value(47,i+13)

            mov_baises_1[i,0] = data_sheet.cell_value(11,i+25)
            #mov_baises_2[i,0] = data_sheet.cell_value(29,i+25)
            #mov_baises_3[i,0] = data_sheet.cell_value(47,i+25)

            output_mov_baises_1[i,0] = data_sheet.cell_value(11,i+37)
            #output_mov_baises_2[i,0] = data_sheet.cell_value(29,i+37)
            #output_mov_baises_3[i,0] = data_sheet.cell_value(47,i+37)
            
            output_baises_1[i,0] = data_sheet.cell_value(11,i+49)
            #output_baises_2[i,0] = data_sheet.cell_value(29,i+49)
            #output_baises_3[i,0] = data_sheet.cell_value(47,i+49)


    result_weights_1,result_baises_1,result_mov_weights_1,result_mov_baises_1 = Momentum_optimizer(input_weights_1,input_baises_1,input_weights_2,input_baises_2,input_weights_3,input_baises_3,mov_weights_1,mov_baises_1,mov_weights_2,mov_baises_2,mov_weights_3,mov_baises_3,input_dloss_dweights_1,input_dloss_dbaises_1,input_dloss_dweights_2,input_dloss_dbaises_2,input_dloss_dweights_3, input_dloss_dbaises_3, learning_rate, beta,num)

    result_weights_1 = np.round(result_weights_1 ,4)
    output_weights_1 = np.round(output_weights_1, 4)
    #result_weights_2 = np.round(result_weights_2 ,4)
    #output_weights_2 = np.round(output_weights_2, 4)
    #result_weights_3 = np.round(result_weights_3 ,4)
    #output_weights_3 = np.round(output_weights_3, 4)


    result_baises_1 = np.round(result_baises_1 ,4)
    output_baises_1 = np.round(output_baises_1, 4) 
    #result_baises_2 = np.round(result_baises_2 ,4)
    #output_baises_2 = np.round(output_baises_2, 4) 
    #result_baises_3 = np.round(result_baises_3 ,4)
    #output_baises_3 = np.round(output_baises_3, 4)

    result_mov_weights_1 = np.round(result_mov_weights_1 ,4)
    output_mov_weights_1 = np.round(output_mov_weights_1, 4)
    #result_mov_weights_2 = np.round(result_mov_weights_2 ,4)
    #output_mov_weights_2 = np.round(output_mov_weights_2, 4)
    #result_mov_weights_3 = np.round(result_mov_weights_3 ,4)
    #output_mov_weights_3 = np.round(output_mov_weights_3, 4)


    result_mov_baises_1 = np.round(result_mov_baises_1 ,4)
    output_mov_baises_1 = np.round(output_mov_baises_1, 4) 
    #result_mov_baises_2 = np.round(result_mov_baises_2 ,4)
    #output_mov_baises_2 = np.round(output_mov_baises_2, 4) 
    #result_mov_baises_3 = np.round(result_mov_baises_3 ,4)
    #output_mov_baises_3 = np.round(output_mov_baises_3, 4)
    #print(result_baises_3)
    #print(output_baises_3)
    
    #if ((result_baises_3 == output_baises_3).all()):
    #    print("Fuction tested okay")
    #if ((result_baises_3 == output_baises_3).all()) == False:
    #    print("Fuction tested not okay")
        
    if (((result_weights_1 == output_weights_1).all()) and  ((result_baises_1 == output_baises_1).all()) and ((result_mov_weights_1 == output_mov_weights_1).all()) and ((result_mov_baises_1 == output_mov_baises_1).all())):
        print("Fuction tested okay")
    if (((result_weights_1 == output_weights_1).all()) and  ((result_baises_1 == output_baises_1).all()) and ((result_mov_weights_1 == output_mov_weights_1).all()) and ((result_mov_baises_1 == output_mov_baises_1).all())) == False:
        print("Fuction tested not okay")
    return None

def reg_loss_test_2_layers():

    print("Function reg_loss_for two hidden layer is under testing")
    num = 2
    alpha  = 0.00001
    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet7')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    #print(total_rows)
    #print(total_cols)
    input_weights_1 = np.zeros((10,10),dtype = np.float32)
    input_weights_2 = np.zeros((10,10),dtype = np.float32)
    input_weights_3 = np.zeros((10,10),dtype = np.float32)
    output_loss = data_sheet.cell_value(2,15)
    #print(output_loss)
    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_weights_1[i,j] = data_sheet.cell_value(i+6,j+1)
            input_weights_2[i,j] = data_sheet.cell_value(i+19,j+1)
            input_weights_3[i,j] = data_sheet.cell_value(i+32,j+1)

    result_loss = reg_loss(input_weights_1, input_weights_2, input_weights_3, alpha, num)
    output_loss = round(output_loss, 6)
    result_loss = round(result_loss, 6)
    #for i in range(10):
    #    for j in range(4):
    #        if output_array[i,j] != result_array[i,j]:
    #            print(i)
    #            print(j)
    #print(result_array)
    #print(output_array)
    #print(result_loss)
    

    if result_loss == output_loss:
        print("Fuction tested okay")
    if result_loss != output_loss:
        print("Fuction tested not okay")
    return None

def reg_loss_test_1_layers():

    print("Function reg_loss_for one hidden layer is under testing")
    num = 1
    alpha  = 0.00001
    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet7')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    #print(total_rows)
    #print(total_cols)
    input_weights_1 = np.zeros((10,10),dtype = np.float32)
    input_weights_2 = np.zeros((10,10),dtype = np.float32)
    input_weights_3 = 0
    output_loss = data_sheet.cell_value(2,16)
    #print(output_loss)
    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_weights_1[i,j] = data_sheet.cell_value(i+6,j+1)
            input_weights_2[i,j] = data_sheet.cell_value(i+19,j+1)
            #input_weights_3[i,j] = data_sheet.cell_value(i+32,j+1)

    result_loss = reg_loss(input_weights_1, input_weights_2, input_weights_3, alpha, num)
    output_loss = round(output_loss, 6)
    result_loss = round(result_loss, 6)
    #for i in range(10):
    #    for j in range(4):
    #        if output_array[i,j] != result_array[i,j]:
    #            print(i)
    #            print(j)
    #print(result_array)
    #print(output_array)
    #print(result_loss)
    

    if result_loss == output_loss:
        print("Fuction tested okay")
    if result_loss != output_loss:
        print("Fuction tested not okay")
    return None

def reg_loss_test_0_layers():

    print("Function reg_loss_for zero hidden layer is under testing")
    num = 0
    alpha  = 0.00001
    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet7')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    #print(total_rows)
    #print(total_cols)
    input_weights_1 = np.zeros((10,10),dtype = np.float32)
    input_weights_2 = 0
    input_weights_3 = 0
    output_loss = data_sheet.cell_value(2,17)
    #print(output_loss)
    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_weights_1[i,j] = data_sheet.cell_value(i+6,j+1)
            #input_weights_2[i,j] = data_sheet.cell_value(i+19,j+1)
            #input_weights_3[i,j] = data_sheet.cell_value(i+32,j+1)

    result_loss = reg_loss(input_weights_1, input_weights_2, input_weights_3, alpha, num)
    output_loss = round(output_loss, 6)
    result_loss = round(result_loss, 6)
    #for i in range(10):
    #    for j in range(4):
    #        if output_array[i,j] != result_array[i,j]:
    #            print(i)
    #            print(j)
    #print(result_array)
    #print(output_array)
    #print(result_loss)
    

    if result_loss == output_loss:
        print("Fuction tested okay")
    if result_loss != output_loss:
        print("Fuction tested not okay")
    return None  

def learning_rate_decay_test():

    print("Function learning_rate_decay is under testing")
    #learning_rate  = 0.001
    #decay = 0.95
    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet8')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    input_learning_rate = data_sheet.cell_value(2,3)
    input_decay = data_sheet.cell_value(3,3)
    output_decay_learning_rate = data_sheet.cell_value(5,3)

    result_learning_rate = learning_rate_decay(input_learning_rate,input_decay)

    if output_decay_learning_rate == result_learning_rate:
        print("Fuction tested okay")
    if output_decay_learning_rate != result_learning_rate:
        print("Fuction tested not okay")
    return None 


def Normal_normalization_test():

    print("Function Normal_normalization is under testing")

    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet9')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    #print(total_rows)
    #print(total_cols)
    input_image_array = np.zeros((10,10),dtype = np.float32)
    output_image_array = np.zeros((10,10),dtype = np.float32)
    
    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_image_array [i,j] = data_sheet.cell_value(i+6,j+2)
            output_image_array [i,j] = data_sheet.cell_value(i+6,j+14)


    result_image_array = Normal_normalization(input_image_array)
    output_image_array = np.round(output_image_array,4)
    result_image_array = np.round(result_image_array, 4)

    #for i in range(10):
    #    for j in range(4):
    #        if output_array[i,j] != result_array[i,j]:
    #            print(i)
    #            print(j)
    #print(result_array)
    #print(output_image_array)
    #print(result_image_array)
    

    if (output_image_array == result_image_array).all():
        print("Fuction tested okay")
    if ((output_image_array == result_image_array).all())==False:
        print("Fuction tested not okay")
    return None 
    
def Simple_normalization_test():

    print("Function Simple_normalization is under testing")

    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet10')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    #print(total_rows)
    #print(total_cols)
    input_image_array = np.zeros((10,10),dtype = np.float32)
    output_image_array = np.zeros((10,10),dtype = np.float32)
    
    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_image_array [i,j] = data_sheet.cell_value(i+6,j+2)
            output_image_array [i,j] = data_sheet.cell_value(i+6,j+14)


    result_image_array = Simple_normalization(input_image_array)
    output_image_array = np.round(output_image_array,4)
    result_image_array = np.round(result_image_array, 4)

    #for i in range(10):
    #    for j in range(4):
    #        if output_array[i,j] != result_array[i,j]:
    #            print(i)
    #            print(j)
    #print(result_array)
    #print(output_image_array)
    #print(result_image_array)
    

    if (output_image_array == result_image_array).all():
        print("Fuction tested okay")
    if ((output_image_array == result_image_array).all())==False:
        print("Fuction tested not okay")
    return None

def Gaussian_initialization_test():
    print("Function Gaussian_initialization is under testing")

    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet11')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    #print(total_rows)
    #print(total_cols)
    input_image_array = np.zeros((10,10),dtype = np.float32)
    output_image_array = np.zeros((10,10),dtype = np.float32)
    
    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_image_array [i,j] = data_sheet.cell_value(i+6,j+2)
            output_image_array [i,j] = data_sheet.cell_value(i+6,j+14)


    result_image_array = Gaussian_initialization(input_image_array)
    output_image_array = np.round(output_image_array,4)
    result_image_array = np.round(result_image_array, 4)

    #for i in range(10):
    #    for j in range(4):
    #        if output_array[i,j] != result_array[i,j]:
    #            print(i)
    #            print(j)
    #print(result_array)
    #print(output_image_array)
    #print(result_image_array)
    

    if (output_image_array == result_image_array).all():
        print("Fuction tested okay")
    if ((output_image_array == result_image_array).all())==False:
        print("Fuction tested not okay")
    return None


def Xavier_initialization_test():
    print("Function Xavier_initialization is under testing")

    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet12')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    #print(total_rows)
    #print(total_cols)
    input_image_array = np.zeros((10,10),dtype = np.float32)
    output_image_array = np.zeros((10,10),dtype = np.float32)
    
    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_image_array [i,j] = data_sheet.cell_value(i+6,j+2)
            output_image_array [i,j] = data_sheet.cell_value(i+6,j+14)


    result_image_array = Xavier_initialization(input_image_array)
    output_image_array = np.round(output_image_array,4)
    result_image_array = np.round(result_image_array, 4)

    #for i in range(10):
    #    for j in range(4):
    #        if output_array[i,j] != result_array[i,j]:
    #            print(i)
    #            print(j)
    #print(result_array)
    #print(output_image_array)
    #print(result_image_array)
    

    if (output_image_array == result_image_array).all():
        print("Fuction tested okay")
    if ((output_image_array == result_image_array).all())==False:
        print("Fuction tested not okay")
    return None

def dropout_forward_test():
    print("Function dropout forward under testing")

    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet13')
    prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    #print(total_rows)
    #print(total_cols)
    input_array = np.zeros((10,4),dtype = np.float32)
    output_array = np.zeros((10,4),dtype = np.float32)
    for i in range(10):
        for j in range(4):
            input_array[i,j] = data_sheet.cell_value(i+5,j+1)
            output_array[i,j] = data_sheet.cell_value(i+5,j+6)

    result_array = dropout_forward(input_array,prob)
    
    if (result_array == output_array).all() :
        print("Fuction tested okay")
    if ((result_array == output_array).all()) == False :
        print("Fuction tested not okay")
    return None

def Loss_function_with_reg_2_layer_test():

    print("Loss function for 2 hidden layer with reggresion is under testing")
    num = 2
    alpha  = 0.00001
    reg = "y"
    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet14')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    #print(total_rows)
    #print(total_cols)
    input_weights_1 = np.zeros((10,10),dtype = np.float32)
    input_weights_2 = np.zeros((10,10),dtype = np.float32)
    input_weights_3 = np.zeros((10,10),dtype = np.float32)
    output_loss = data_sheet.cell_value(1,22)
    input_pred_y_array = np.zeros((10,4),dtype = np.float32)
    input_true_y_array = np.zeros((4,10),dtype = np.float32)
    for i in range(10):
        for j in range(4):
            input_pred_y_array[i,j] = data_sheet.cell_value(i+6,j+7)
            input_true_y_array [j,i] = data_sheet.cell_value(i+6,j+12)
    #print(output_loss)
    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_weights_1[i,j] = data_sheet.cell_value(i+22,j+1)
            input_weights_2[i,j] = data_sheet.cell_value(i+35,j+1)
            input_weights_3[i,j] = data_sheet.cell_value(i+48,j+1)

    result_loss = Loss_function(input_pred_y_array, input_true_y_array,input_weights_1, input_weights_2, input_weights_3, alpha, reg,num)
    output_loss = round(output_loss, 6)
    result_loss = round(result_loss, 6)
    #for i in range(10):
    #    for j in range(4):
    #        if output_array[i,j] != result_array[i,j]:
    #            print(i)
    #            print(j)
    #print(result_array)
    #print(output_loss)
    #print(result_loss)
    

    if result_loss == output_loss:
        print("Fuction tested okay")
    if result_loss != output_loss:
        print("Fuction tested not okay")
    return None


def Loss_function_with_reg_1_layer_test():

    print("Loss function for 1 hidden layer with reggresion is under testing")

    num = 1
    alpha  = 0.00001
    reg = "y"
    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet14')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    #print(total_rows)
    #print(total_cols)
    input_weights_1 = np.zeros((10,10),dtype = np.float32)
    input_weights_2 = np.zeros((10,10),dtype = np.float32)
    input_weights_3 = 0
    output_loss = data_sheet.cell_value(1,23)
    input_pred_y_array = np.zeros((10,4),dtype = np.float32)
    input_true_y_array = np.zeros((4,10),dtype = np.float32)
    for i in range(10):
        for j in range(4):
            input_pred_y_array[i,j] = data_sheet.cell_value(i+6,j+7)
            input_true_y_array [j,i] = data_sheet.cell_value(i+6,j+12)
    #print(output_loss)
    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_weights_1[i,j] = data_sheet.cell_value(i+22,j+1)
            input_weights_2[i,j] = data_sheet.cell_value(i+35,j+1)
            #input_weights_3[i,j] = data_sheet.cell_value(i+48,j+1)

    result_loss = Loss_function(input_pred_y_array, input_true_y_array,input_weights_1, input_weights_2, input_weights_3, alpha, reg,num)
    output_loss = round(output_loss, 6)
    result_loss = round(result_loss, 6)
    #for i in range(10):
    #    for j in range(4):
    #        if output_array[i,j] != result_array[i,j]:
    #            print(i)
    #            print(j)
    #print(result_array)
    #print(output_loss)
    #print(result_loss)
    

    if result_loss == output_loss:
        print("Fuction tested okay")
    if result_loss != output_loss:
        print("Fuction tested not okay")
    return None

def Loss_function_with_reg_0_layer_test():

    print("Loss function zero layer with reggresion is under testing")

    num = 0
    alpha  = 0.00001
    reg = "y"
    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet14')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    #print(total_rows)
    #print(total_cols)
    input_weights_1 = np.zeros((10,10),dtype = np.float32)
    input_weights_2 = 0
    input_weights_3 = 0
    output_loss = data_sheet.cell_value(1,24)
    input_pred_y_array = np.zeros((10,4),dtype = np.float32)
    input_true_y_array = np.zeros((4,10),dtype = np.float32)
    for i in range(10):
        for j in range(4):
            input_pred_y_array[i,j] = data_sheet.cell_value(i+6,j+7)
            input_true_y_array [j,i] = data_sheet.cell_value(i+6,j+12)
    #print(output_loss)
    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_weights_1[i,j] = data_sheet.cell_value(i+22,j+1)
            #input_weights_2[i,j] = data_sheet.cell_value(i+35,j+1)
            #input_weights_3[i,j] = data_sheet.cell_value(i+48,j+1)

    result_loss = Loss_function(input_pred_y_array, input_true_y_array,input_weights_1, input_weights_2, input_weights_3, alpha, reg,num)
    output_loss = round(output_loss, 6)
    result_loss = round(result_loss, 6)
    #for i in range(10):
    #    for j in range(4):
    #        if output_array[i,j] != result_array[i,j]:
    #            print(i)
    #            print(j)
    #print(result_array)
    #print(output_loss)
    #print(result_loss)
    

    if result_loss == output_loss:
        print("Fuction tested okay")
    if result_loss != output_loss:
        print("Fuction tested not okay")
    return None

def Loss_function_with_no_reg_test():
    print("Loss function with no reggresion is under testing")
    num=2

    alpha  = 0.00001
    reg = "n"
    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet14')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    #print(total_rows)
    #print(total_cols)
    input_weights_1 = np.zeros((10,10),dtype = np.float32)
    input_weights_2 = np.zeros((10,10),dtype = np.float32)
    input_weights_3 = np.zeros((10,10),dtype = np.float32)
    output_loss = data_sheet.cell_value(4,19)
    input_pred_y_array = np.zeros((10,4),dtype = np.float32)
    input_true_y_array = np.zeros((4,10),dtype = np.float32)
    for i in range(10):
        for j in range(4):
            input_pred_y_array[i,j] = data_sheet.cell_value(i+6,j+7)
            input_true_y_array [j,i] = data_sheet.cell_value(i+6,j+12)
    #print(output_loss)
    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_weights_1[i,j] = data_sheet.cell_value(i+22,j+1)
            input_weights_2[i,j] = data_sheet.cell_value(i+35,j+1)
            input_weights_3[i,j] = data_sheet.cell_value(i+48,j+1)

    result_loss = Loss_function(input_pred_y_array, input_true_y_array,input_weights_1, input_weights_2, input_weights_3, alpha, reg, num)
    output_loss = round(output_loss, 6)
    result_loss = round(result_loss, 6)
    #for i in range(10):
    #    for j in range(4):
    #        if output_array[i,j] != result_array[i,j]:
    #            print(i)
    #            print(j)
    #print(result_array)
    #print(output_loss)
    #print(result_loss)
    

    if result_loss == output_loss:
        print("Fuction tested okay")
    if result_loss != output_loss:
        print("Fuction tested not okay")
    return None

def forward_prop_for_loss_with_reg_with_relu_test():

    print("Function forward_prop_for_loss for 2 hidden layer with relu activation with reggresion is under testing")

    num=2
    alpha = 0.00001
    dropout = "n"
    prob = 0.8
    reg = "y"
    activation_1 = "relu"

    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet15')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    #print(total_rows)
    #print(total_cols)
    input_weights_1 = np.zeros((10,10),dtype = np.float32)
    input_weights_2 = np.zeros((10,10),dtype = np.float32)
    input_weights_3 = np.zeros((10,10),dtype = np.float32)
    output_loss = data_sheet.cell_value(1,73)

    input_baises_1 = np.zeros((10,1),dtype = np.float32)
    input_baises_2 = np.zeros((10,1),dtype = np.float32)
    input_baises_3 = np.zeros((10,1),dtype = np.float32)

    input_image_array = np.zeros((4,10),dtype = np.float32)
    input_label_array = np.zeros((4,10),dtype = np.float32)

    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_weights_1[j,i] = data_sheet.cell_value(i+22,j+1)
            input_weights_2[j,i] = data_sheet.cell_value(i+35,j+1)
            input_weights_3[j,i] = data_sheet.cell_value(i+48,j+1)


    for i in range(10):
        input_baises_1[i,0] = data_sheet.cell_value(i+22,12)
        input_baises_2[i,0] = data_sheet.cell_value(i+35,12)
        input_baises_3[i,0] = data_sheet.cell_value(i+48,12)


    for i in range(10):
        for j in range(4):
            #print(i)
            #print(j)
            input_image_array[j,i] = data_sheet.cell_value(i+6,j+2)
            input_label_array[j,i] = data_sheet.cell_value(i+6,j+63)

    result_loss = forward_prop_for_loss(input_weights_1,input_baises_1,input_weights_2, input_baises_2,input_weights_3,input_baises_3, input_image_array, input_label_array, alpha ,reg, prob,activation_1,dropout, num)
    output_loss = round(output_loss, 4)
    result_loss = round(result_loss, 4)
    #for i in range(10):
    #    for j in range(4):
    #        if output_array[i,j] != result_array[i,j]:
    #            print(i)
    #            print(j)
    #print(result_array)
    #print(output_loss)
    #print(result_loss)
    

    if result_loss == output_loss:
        print("Fuction tested okay")
    if result_loss != output_loss:
        print("Fuction tested not okay")

    return None



def forward_prop_for_loss_with_no_reg_with_relu_test():

    print("Function forward_prop_for_loss for 2 hidden layer with relu activation with no reggresion is under testing")
    num =2

    alpha = 0.00001
    dropout = "n"
    prob = 0.8
    reg = "n"
    activation_1 = "relu"

    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet15')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    #print(total_rows)
    #print(total_cols)
    input_weights_1 = np.zeros((10,10),dtype = np.float32)
    input_weights_2 = np.zeros((10,10),dtype = np.float32)
    input_weights_3 = np.zeros((10,10),dtype = np.float32)
    output_loss = data_sheet.cell_value(4,70)

    input_baises_1 = np.zeros((10,1),dtype = np.float32)
    input_baises_2 = np.zeros((10,1),dtype = np.float32)
    input_baises_3 = np.zeros((10,1),dtype = np.float32)

    input_image_array = np.zeros((4,10),dtype = np.float32)
    input_label_array = np.zeros((4,10),dtype = np.float32)

    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_weights_1[j,i] = data_sheet.cell_value(i+22,j+1)
            input_weights_2[j,i] = data_sheet.cell_value(i+35,j+1)
            input_weights_3[j,i] = data_sheet.cell_value(i+48,j+1)


    for i in range(10):
        input_baises_1[i,0] = data_sheet.cell_value(i+22,12)
        input_baises_2[i,0] = data_sheet.cell_value(i+35,12)
        input_baises_3[i,0] = data_sheet.cell_value(i+48,12)


    for i in range(10):
        for j in range(4):
            #print(i)
            #print(j)
            input_image_array[j,i] = data_sheet.cell_value(i+6,j+2)
            input_label_array[j,i] = data_sheet.cell_value(i+6,j+63)

    result_loss = forward_prop_for_loss(input_weights_1,input_baises_1,input_weights_2, input_baises_2,input_weights_3,input_baises_3, input_image_array, input_label_array, alpha ,reg, prob,activation_1,dropout,num)
    output_loss = round(output_loss, 4)
    result_loss = round(result_loss, 4)
    #for i in range(10):
    #    for j in range(4):
    #        if output_array[i,j] != result_array[i,j]:
    #            print(i)
    #            print(j)
    #print(result_array)
    #print(output_loss)
    #print(result_loss)
    

    if result_loss == output_loss:
        print("Fuction tested okay")
    if result_loss != output_loss:
        print("Fuction tested not okay")

    return None

def forward_prop_for_loss_with_reg_with_sigmoid_test():

    print("Function forward_prop_for_loss for 2 hidden layer with sigmoid activation with reggresion is under testing")
    num=2
    alpha = 0.00001
    dropout = "n"
    prob = 0.8
    reg = "y"
    activation_1 = "sigmoid"

    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet16')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    #print(total_rows)
    #print(total_cols)
    input_weights_1 = np.zeros((10,10),dtype = np.float32)
    input_weights_2 = np.zeros((10,10),dtype = np.float32)
    input_weights_3 = np.zeros((10,10),dtype = np.float32)
    output_loss = data_sheet.cell_value(1,73)

    input_baises_1 = np.zeros((10,1),dtype = np.float32)
    input_baises_2 = np.zeros((10,1),dtype = np.float32)
    input_baises_3 = np.zeros((10,1),dtype = np.float32)

    input_image_array = np.zeros((4,10),dtype = np.float32)
    input_label_array = np.zeros((4,10),dtype = np.float32)

    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_weights_1[j,i] = data_sheet.cell_value(i+22,j+1)
            input_weights_2[j,i] = data_sheet.cell_value(i+35,j+1)
            input_weights_3[j,i] = data_sheet.cell_value(i+48,j+1)


    for i in range(10):
        input_baises_1[i,0] = data_sheet.cell_value(i+22,12)
        input_baises_2[i,0] = data_sheet.cell_value(i+35,12)
        input_baises_3[i,0] = data_sheet.cell_value(i+48,12)


    for i in range(10):
        for j in range(4):
            #print(i)
            #print(j)
            input_image_array[j,i] = data_sheet.cell_value(i+6,j+2)
            input_label_array[j,i] = data_sheet.cell_value(i+6,j+63)

    result_loss = forward_prop_for_loss(input_weights_1,input_baises_1,input_weights_2, input_baises_2,input_weights_3,input_baises_3, input_image_array, input_label_array, alpha ,reg, prob,activation_1,dropout,num)
    output_loss = round(output_loss, 4)
    result_loss = round(result_loss, 4)
    #for i in range(10):
    #    for j in range(4):
    #        if output_array[i,j] != result_array[i,j]:
    #            print(i)
    #            print(j)
    #print(result_array)
    #print(output_loss)
    #print(result_loss)
    

    if result_loss == output_loss:
        print("Fuction tested okay")
    if result_loss != output_loss:
        print("Fuction tested not okay")

    return None

def forward_prop_for_loss_with_no_reg_with_sigmoid_test():
    
    print("Function forward_prop_for_loss for 2 hidden layer with sigmoid activation with no reggresion is under testing")
    
    num =2

    alpha = 0.00001
    dropout = "n"
    prob = 0.8
    reg = "n"
    activation_1 = "sigmoid"

    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet16')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    #print(total_rows)
    #print(total_cols)
    input_weights_1 = np.zeros((10,10),dtype = np.float32)
    input_weights_2 = np.zeros((10,10),dtype = np.float32)
    input_weights_3 = np.zeros((10,10),dtype = np.float32)
    output_loss = data_sheet.cell_value(4,70)

    input_baises_1 = np.zeros((10,1),dtype = np.float32)
    input_baises_2 = np.zeros((10,1),dtype = np.float32)
    input_baises_3 = np.zeros((10,1),dtype = np.float32)

    input_image_array = np.zeros((4,10),dtype = np.float32)
    input_label_array = np.zeros((4,10),dtype = np.float32)

    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_weights_1[j,i] = data_sheet.cell_value(i+22,j+1)
            input_weights_2[j,i] = data_sheet.cell_value(i+35,j+1)
            input_weights_3[j,i] = data_sheet.cell_value(i+48,j+1)


    for i in range(10):
        input_baises_1[i,0] = data_sheet.cell_value(i+22,12)
        input_baises_2[i,0] = data_sheet.cell_value(i+35,12)
        input_baises_3[i,0] = data_sheet.cell_value(i+48,12)


    for i in range(10):
        for j in range(4):
            #print(i)
            #print(j)
            input_image_array[j,i] = data_sheet.cell_value(i+6,j+2)
            input_label_array[j,i] = data_sheet.cell_value(i+6,j+63)

    result_loss = forward_prop_for_loss(input_weights_1,input_baises_1,input_weights_2, input_baises_2,input_weights_3,input_baises_3, input_image_array, input_label_array, alpha ,reg, prob,activation_1,dropout,num)
    output_loss = round(output_loss, 4)
    result_loss = round(result_loss, 4)
    #for i in range(10):
    #    for j in range(4):
    #        if output_array[i,j] != result_array[i,j]:
    #            print(i)
    #            print(j)
    #print(result_array)
    #print(output_loss)
    #print(result_loss)
    

    if result_loss == output_loss:
        print("Fuction tested okay")
    if result_loss != output_loss:
        print("Fuction tested not okay")

    return None

def forward_prop_for_loss_with_reg_with_relu_layer_1_test():

    print("Function forward_prop_for_loss for 1 hidden layer with relu activation with reggresion is under testing")

    num=1
    alpha = 0.00001
    dropout = "n"
    prob = 0.8
    reg = "y"
    activation_1 = "relu"

    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet103')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    #print(total_rows)
    #print(total_cols)
    input_weights_1 = np.zeros((10,10),dtype = np.float32)
    input_weights_2 = np.zeros((10,10),dtype = np.float32)
    input_weights_3 = 0
    output_loss = data_sheet.cell_value(1,73)

    input_baises_1 = np.zeros((10,1),dtype = np.float32)
    input_baises_2 = np.zeros((10,1),dtype = np.float32)
    input_baises_3 = 0

    input_image_array = np.zeros((4,10),dtype = np.float32)
    input_label_array = np.zeros((4,10),dtype = np.float32)

    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_weights_1[j,i] = data_sheet.cell_value(i+22,j+1)
            input_weights_2[j,i] = data_sheet.cell_value(i+35,j+1)
            #input_weights_3[j,i] = data_sheet.cell_value(i+48,j+1)


    for i in range(10):
        input_baises_1[i,0] = data_sheet.cell_value(i+22,12)
        input_baises_2[i,0] = data_sheet.cell_value(i+35,12)
        #input_baises_3[i,0] = data_sheet.cell_value(i+48,12)


    for i in range(10):
        for j in range(4):
            #print(i)
            #print(j)
            input_image_array[j,i] = data_sheet.cell_value(i+6,j+2)
            input_label_array[j,i] = data_sheet.cell_value(i+6,j+63)

    result_loss = forward_prop_for_loss(input_weights_1,input_baises_1,input_weights_2, input_baises_2,input_weights_3,input_baises_3, input_image_array, input_label_array, alpha ,reg, prob,activation_1,dropout, num)
    output_loss = round(output_loss, 4)
    result_loss = round(result_loss, 4)
    #for i in range(10):
    #    for j in range(4):
    #        if output_array[i,j] != result_array[i,j]:
    #            print(i)
    #            print(j)
    #print(result_array)
    #print(output_loss)
    #print(result_loss)
    

    if result_loss == output_loss:
        print("Fuction tested okay")
    if result_loss != output_loss:
        print("Fuction tested not okay")

    return None


def forward_prop_for_loss_with_no_reg_with_relu_layer_1_test():
    print("Function forward_prop_for_loss for 1 hidden layer with relu activation with no reggresion is under testing")
    num =1

    alpha = 0.00001
    dropout = "n"
    prob = 0.8
    reg = "n"
    activation_1 = "relu"

    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet103')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    #print(total_rows)
    #print(total_cols)
    input_weights_1 = np.zeros((10,10),dtype = np.float32)
    input_weights_2 = np.zeros((10,10),dtype = np.float32)
    input_weights_3 = 0
    output_loss = data_sheet.cell_value(4,70)

    input_baises_1 = np.zeros((10,1),dtype = np.float32)
    input_baises_2 = np.zeros((10,1),dtype = np.float32)
    input_baises_3 = 0

    input_image_array = np.zeros((4,10),dtype = np.float32)
    input_label_array = np.zeros((4,10),dtype = np.float32)

    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_weights_1[j,i] = data_sheet.cell_value(i+22,j+1)
            input_weights_2[j,i] = data_sheet.cell_value(i+35,j+1)
            #input_weights_3[j,i] = data_sheet.cell_value(i+48,j+1)


    for i in range(10):
        input_baises_1[i,0] = data_sheet.cell_value(i+22,12)
        input_baises_2[i,0] = data_sheet.cell_value(i+35,12)
        #input_baises_3[i,0] = data_sheet.cell_value(i+48,12)


    for i in range(10):
        for j in range(4):
            #print(i)
            #print(j)
            input_image_array[j,i] = data_sheet.cell_value(i+6,j+2)
            input_label_array[j,i] = data_sheet.cell_value(i+6,j+63)

    result_loss = forward_prop_for_loss(input_weights_1,input_baises_1,input_weights_2, input_baises_2,input_weights_3,input_baises_3, input_image_array, input_label_array, alpha ,reg, prob,activation_1,dropout,num)
    output_loss = round(output_loss, 4)
    result_loss = round(result_loss, 4)
    #for i in range(10):
    #    for j in range(4):
    #        if output_array[i,j] != result_array[i,j]:
    #            print(i)
    #            print(j)
    #print(result_array)
    #print(output_loss)
    #print(result_loss)
    

    if result_loss == output_loss:
        print("Fuction tested okay")
    if result_loss != output_loss:
        print("Fuction tested not okay")

    return None

def forward_prop_for_loss_with_reg_with_sigmoid_layer_1_test():
    print("Function forward_prop_for_loss for 1 hidden layer with sigmoid activation with reggresion is under testing")
    num=1
    alpha = 0.00001
    dropout = "n"
    prob = 0.8
    reg = "y"
    activation_1 = "sigmoid"

    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet104')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    #print(total_rows)
    #print(total_cols)
    input_weights_1 = np.zeros((10,10),dtype = np.float32)
    input_weights_2 = np.zeros((10,10),dtype = np.float32)
    input_weights_3 = 0
    output_loss = data_sheet.cell_value(1,73)

    input_baises_1 = np.zeros((10,1),dtype = np.float32)
    input_baises_2 = np.zeros((10,1),dtype = np.float32)
    input_baises_3 = 0

    input_image_array = np.zeros((4,10),dtype = np.float32)
    input_label_array = np.zeros((4,10),dtype = np.float32)

    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_weights_1[j,i] = data_sheet.cell_value(i+22,j+1)
            input_weights_2[j,i] = data_sheet.cell_value(i+35,j+1)
            #input_weights_3[j,i] = data_sheet.cell_value(i+48,j+1)


    for i in range(10):
        input_baises_1[i,0] = data_sheet.cell_value(i+22,12)
        input_baises_2[i,0] = data_sheet.cell_value(i+35,12)
        #input_baises_3[i,0] = data_sheet.cell_value(i+48,12)


    for i in range(10):
        for j in range(4):
            #print(i)
            #print(j)
            input_image_array[j,i] = data_sheet.cell_value(i+6,j+2)
            input_label_array[j,i] = data_sheet.cell_value(i+6,j+63)

    result_loss = forward_prop_for_loss(input_weights_1,input_baises_1,input_weights_2, input_baises_2,input_weights_3,input_baises_3, input_image_array, input_label_array, alpha ,reg, prob,activation_1,dropout,num)
    output_loss = round(output_loss, 4)
    result_loss = round(result_loss, 4)
    #for i in range(10):
    #    for j in range(4):
    #        if output_array[i,j] != result_array[i,j]:
    #            print(i)
    #            print(j)
    #print(result_array)
    #print(output_loss)
    #print(result_loss)
    

    if result_loss == output_loss:
        print("Fuction tested okay")
    if result_loss != output_loss:
        print("Fuction tested not okay")

    return None

def forward_prop_for_loss_with_no_reg_with_sigmoid_layer_1_test():
    print("Function forward_prop_for_loss for 1 hidden layer with sigmoid activation with no reggresion is under testing")
    num =1

    alpha = 0.00001
    dropout = "n"
    prob = 0.8
    reg = "n"
    activation_1 = "sigmoid"

    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet104')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    #print(total_rows)
    #print(total_cols)
    input_weights_1 = np.zeros((10,10),dtype = np.float32)
    input_weights_2 = np.zeros((10,10),dtype = np.float32)
    input_weights_3 = 0
    output_loss = data_sheet.cell_value(4,70)

    input_baises_1 = np.zeros((10,1),dtype = np.float32)
    input_baises_2 = np.zeros((10,1),dtype = np.float32)
    input_baises_3 = 0

    input_image_array = np.zeros((4,10),dtype = np.float32)
    input_label_array = np.zeros((4,10),dtype = np.float32)

    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_weights_1[j,i] = data_sheet.cell_value(i+22,j+1)
            input_weights_2[j,i] = data_sheet.cell_value(i+35,j+1)
            #input_weights_3[j,i] = data_sheet.cell_value(i+48,j+1)


    for i in range(10):
        input_baises_1[i,0] = data_sheet.cell_value(i+22,12)
        input_baises_2[i,0] = data_sheet.cell_value(i+35,12)
        #input_baises_3[i,0] = data_sheet.cell_value(i+48,12)


    for i in range(10):
        for j in range(4):
            #print(i)
            #print(j)
            input_image_array[j,i] = data_sheet.cell_value(i+6,j+2)
            input_label_array[j,i] = data_sheet.cell_value(i+6,j+63)

    result_loss = forward_prop_for_loss(input_weights_1,input_baises_1,input_weights_2, input_baises_2,input_weights_3,input_baises_3, input_image_array, input_label_array, alpha ,reg, prob,activation_1,dropout,num)
    output_loss = round(output_loss, 4)
    result_loss = round(result_loss, 4)
    #for i in range(10):
    #    for j in range(4):
    #        if output_array[i,j] != result_array[i,j]:
    #            print(i)
    #            print(j)
    #print(result_array)
    #print(output_loss)
    #print(result_loss)
    

    if result_loss == output_loss:
        print("Fuction tested okay")
    if result_loss != output_loss:
        print("Fuction tested not okay")
        
    return None

def forward_prop_for_loss_with_reg_layer_0_test():
    print("Function forward_prop_for_loss for zero hidden layer with reggresion is under testing")

    num=0
    alpha = 0.00001
    dropout = "n"
    prob = 0.8
    reg = "y"
    activation_1 = "n"

    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet105')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    #print(total_rows)
    #print(total_cols)
    input_weights_1 = np.zeros((10,10),dtype = np.float32)
    input_weights_2 = 0
    input_weights_3 = 0
    output_loss = data_sheet.cell_value(1,73)

    input_baises_1 = np.zeros((10,1),dtype = np.float32)
    input_baises_2 = 0
    input_baises_3 = 0

    input_image_array = np.zeros((4,10),dtype = np.float32)
    input_label_array = np.zeros((4,10),dtype = np.float32)

    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_weights_1[j,i] = data_sheet.cell_value(i+22,j+1)
            #input_weights_2[j,i] = data_sheet.cell_value(i+35,j+1)
            #input_weights_3[j,i] = data_sheet.cell_value(i+48,j+1)


    for i in range(10):
        input_baises_1[i,0] = data_sheet.cell_value(i+22,12)
        #input_baises_2[i,0] = data_sheet.cell_value(i+35,12)
        #input_baises_3[i,0] = data_sheet.cell_value(i+48,12)


    for i in range(10):
        for j in range(4):
            #print(i)
            #print(j)
            input_image_array[j,i] = data_sheet.cell_value(i+6,j+2)
            input_label_array[j,i] = data_sheet.cell_value(i+6,j+63)

    result_loss = forward_prop_for_loss(input_weights_1,input_baises_1,input_weights_2, input_baises_2,input_weights_3,input_baises_3, input_image_array, input_label_array, alpha ,reg, prob,activation_1,dropout, num)
    output_loss = round(output_loss, 4)
    result_loss = round(result_loss, 4)
    #for i in range(10):
    #    for j in range(4):
    #        if output_array[i,j] != result_array[i,j]:
    #            print(i)
    #            print(j)
    #print(result_array)
    #print(output_loss)
    #print(result_loss)
    

    if result_loss == output_loss:
        print("Fuction tested okay")
    if result_loss != output_loss:
        print("Fuction tested not okay")

    return None



def forward_prop_for_loss_with_no_reg_layer_0_test():
    print("Function forward_prop_for_loss for zero hidden layer with no reggresion is under testing")
    num =0

    alpha = 0.00001
    dropout = "n"
    prob = 0.8
    reg = "n"
    activation_1 = "n"

    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet105')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    #print(total_rows)
    #print(total_cols)
    input_weights_1 = np.zeros((10,10),dtype = np.float32)
    input_weights_2 = 0
    input_weights_3 = 0
    output_loss = data_sheet.cell_value(4,70)

    input_baises_1 = np.zeros((10,1),dtype = np.float32)
    input_baises_2 = 0
    input_baises_3 = 0

    input_image_array = np.zeros((4,10),dtype = np.float32)
    input_label_array = np.zeros((4,10),dtype = np.float32)

    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_weights_1[j,i] = data_sheet.cell_value(i+22,j+1)
            #input_weights_2[j,i] = data_sheet.cell_value(i+35,j+1)
            #input_weights_3[j,i] = data_sheet.cell_value(i+48,j+1)


    for i in range(10):
        input_baises_1[i,0] = data_sheet.cell_value(i+22,12)
        #input_baises_2[i,0] = data_sheet.cell_value(i+35,12)
        #input_baises_3[i,0] = data_sheet.cell_value(i+48,12)


    for i in range(10):
        for j in range(4):
            #print(i)
            #print(j)
            input_image_array[j,i] = data_sheet.cell_value(i+6,j+2)
            input_label_array[j,i] = data_sheet.cell_value(i+6,j+63)

    result_loss = forward_prop_for_loss(input_weights_1,input_baises_1,input_weights_2, input_baises_2,input_weights_3,input_baises_3, input_image_array, input_label_array, alpha ,reg, prob,activation_1,dropout,num)
    output_loss = round(output_loss, 4)
    result_loss = round(result_loss, 4)
    #for i in range(10):
    #    for j in range(4):
    #        if output_array[i,j] != result_array[i,j]:
    #            print(i)
    #            print(j)
    #print(result_array)
    #print(output_loss)
    #print(result_loss)
    

    if result_loss == output_loss:
        print("Fuction tested okay")
    if result_loss != output_loss:
        print("Fuction tested not okay")

    return None


def Accuracy_with_relu_test():

    print("Function Accuracy for 2 hidden layer and relu activation is under testing")
    num =2

    #alpha = 0.00001
    #dropout = "n"
    #prob = 0.8
    #reg = "y"
    activation_1 = "relu"

    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet17')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    #print(total_rows)
    #print(total_cols)
    input_weights_1 = np.zeros((10,10),dtype = np.float32)
    input_weights_2 = np.zeros((10,10),dtype = np.float32)
    input_weights_3 = np.zeros((10,10),dtype = np.float32)
    output_accuracy = data_sheet.cell_value(2,65)

    input_baises_1 = np.zeros((10,1),dtype = np.float32)
    input_baises_2 = np.zeros((10,1),dtype = np.float32)
    input_baises_3 = np.zeros((10,1),dtype = np.float32)

    input_image_array = np.zeros((4,10),dtype = np.float32)
    input_label_array = np.zeros((4,10),dtype = np.float32)

    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_weights_1[j,i] = data_sheet.cell_value(i+22,j+1)
            input_weights_2[j,i] = data_sheet.cell_value(i+35,j+1)
            input_weights_3[j,i] = data_sheet.cell_value(i+48,j+1)


    for i in range(10):
        input_baises_1[i,0] = data_sheet.cell_value(i+22,12)
        input_baises_2[i,0] = data_sheet.cell_value(i+35,12)
        input_baises_3[i,0] = data_sheet.cell_value(i+48,12)


    for i in range(10):
        for j in range(4):
            #print(i)
            #print(j)
            input_image_array[j,i] = data_sheet.cell_value(i+6,j+2)
            input_label_array[j,i] = data_sheet.cell_value(i+6,j+63)

    result_accuracy = accuracy(input_weights_1,input_baises_1,input_weights_2, input_baises_2,input_weights_3,input_baises_3, input_image_array, input_label_array,activation_1,num)
    #output_loss = round(output_loss, 4)
    #result_loss = round(result_loss, 4)
    #for i in range(10):
    #    for j in range(4):
    #        if output_array[i,j] != result_array[i,j]:
    #            print(i)
    #            print(j)
    #print(result_array)
    #print(output_loss)
    #print(result_loss)
    

    if result_accuracy == output_accuracy:
        print("Fuction tested okay")
    if result_accuracy != output_accuracy:
        print("Fuction tested not okay")
        
    return None

def Accuracy_with_sigmoid_test():

    print("Function Accuracy for 2 hidden layer and sigmoid activation is under testing")
    num =2

    #alpha = 0.00001
    #dropout = "n"
    #prob = 0.8
    #reg = "y"
    activation_1 = "sigmoid"

    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet18')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    #print(total_rows)
    #print(total_cols)
    input_weights_1 = np.zeros((10,10),dtype = np.float32)
    input_weights_2 = np.zeros((10,10),dtype = np.float32)
    input_weights_3 = np.zeros((10,10),dtype = np.float32)
    output_accuracy = data_sheet.cell_value(2,65)

    input_baises_1 = np.zeros((10,1),dtype = np.float32)
    input_baises_2 = np.zeros((10,1),dtype = np.float32)
    input_baises_3 = np.zeros((10,1),dtype = np.float32)

    input_image_array = np.zeros((4,10),dtype = np.float32)
    input_label_array = np.zeros((4,10),dtype = np.float32)

    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_weights_1[j,i] = data_sheet.cell_value(i+22,j+1)
            input_weights_2[j,i] = data_sheet.cell_value(i+35,j+1)
            input_weights_3[j,i] = data_sheet.cell_value(i+48,j+1)


    for i in range(10):
        input_baises_1[i,0] = data_sheet.cell_value(i+22,12)
        input_baises_2[i,0] = data_sheet.cell_value(i+35,12)
        input_baises_3[i,0] = data_sheet.cell_value(i+48,12)


    for i in range(10):
        for j in range(4):
            #print(i)
            #print(j)
            input_image_array[j,i] = data_sheet.cell_value(i+6,j+2)
            input_label_array[j,i] = data_sheet.cell_value(i+6,j+63)

    result_accuracy = accuracy(input_weights_1,input_baises_1,input_weights_2, input_baises_2,input_weights_3,input_baises_3, input_image_array, input_label_array,activation_1,num)
    #output_loss = round(output_loss, 4)
    #result_loss = round(result_loss, 4)
    #for i in range(10):
    #    for j in range(4):
    #        if output_array[i,j] != result_array[i,j]:
    #            print(i)
    #            print(j)
    #print(result_array)
    #print(output_loss)
    #print(result_loss)
    

    if result_accuracy == output_accuracy:
        print("Fuction tested okay")
    if result_accuracy != output_accuracy:
        print("Fuction tested not okay")
        
    return None

def Accuracy_with_relu_layer_1_test():

    print("Function Accuracy for 1 hidden layer and relu activation is under testing")

    num =1
    #alpha = 0.00001
    #dropout = "n"
    #prob = 0.8
    #reg = "y"
    activation_1 = "relu"

    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet106')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    #print(total_rows)
    #print(total_cols)
    input_weights_1 = np.zeros((10,10),dtype = np.float32)
    input_weights_2 = np.zeros((10,10),dtype = np.float32)
    input_weights_3 = 0
    output_accuracy = data_sheet.cell_value(2,65)

    input_baises_1 = np.zeros((10,1),dtype = np.float32)
    input_baises_2 = np.zeros((10,1),dtype = np.float32)
    input_baises_3 = 0

    input_image_array = np.zeros((4,10),dtype = np.float32)
    input_label_array = np.zeros((4,10),dtype = np.float32)

    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_weights_1[j,i] = data_sheet.cell_value(i+22,j+1)
            input_weights_2[j,i] = data_sheet.cell_value(i+35,j+1)
            #input_weights_3[j,i] = data_sheet.cell_value(i+48,j+1)


    for i in range(10):
        input_baises_1[i,0] = data_sheet.cell_value(i+22,12)
        input_baises_2[i,0] = data_sheet.cell_value(i+35,12)
        #input_baises_3[i,0] = data_sheet.cell_value(i+48,12)


    for i in range(10):
        for j in range(4):
            #print(i)
            #print(j)
            input_image_array[j,i] = data_sheet.cell_value(i+6,j+2)
            input_label_array[j,i] = data_sheet.cell_value(i+6,j+63)

    result_accuracy = accuracy(input_weights_1,input_baises_1,input_weights_2, input_baises_2,input_weights_3,input_baises_3, input_image_array, input_label_array,activation_1,num)
    #output_loss = round(output_loss, 4)
    #result_loss = round(result_loss, 4)
    #for i in range(10):
    #    for j in range(4):
    #        if output_array[i,j] != result_array[i,j]:
    #            print(i)
    #            print(j)
    #print(result_array)
    #print(output_loss)
    #print(result_loss)
    

    if result_accuracy == output_accuracy:
        print("Fuction tested okay")
    if result_accuracy != output_accuracy:
        print("Fuction tested not okay")
        
    return None

def Accuracy_with_sigmoid_layer_1_test():

    print("Function Accuracy for 1 hidden layer and sigmoid activation is under testing")
    num =1

    #alpha = 0.00001
    #dropout = "n"
    #prob = 0.8
    #reg = "y"
    activation_1 = "sigmoid"

    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet107')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    #print(total_rows)
    #print(total_cols)
    input_weights_1 = np.zeros((10,10),dtype = np.float32)
    input_weights_2 = np.zeros((10,10),dtype = np.float32)
    input_weights_3 = 0
    output_accuracy = data_sheet.cell_value(2,65)

    input_baises_1 = np.zeros((10,1),dtype = np.float32)
    input_baises_2 = np.zeros((10,1),dtype = np.float32)
    input_baises_3 = 0

    input_image_array = np.zeros((4,10),dtype = np.float32)
    input_label_array = np.zeros((4,10),dtype = np.float32)

    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_weights_1[j,i] = data_sheet.cell_value(i+22,j+1)
            input_weights_2[j,i] = data_sheet.cell_value(i+35,j+1)
            #input_weights_3[j,i] = data_sheet.cell_value(i+48,j+1)


    for i in range(10):
        input_baises_1[i,0] = data_sheet.cell_value(i+22,12)
        input_baises_2[i,0] = data_sheet.cell_value(i+35,12)
        #input_baises_3[i,0] = data_sheet.cell_value(i+48,12)


    for i in range(10):
        for j in range(4):
            #print(i)
            #print(j)
            input_image_array[j,i] = data_sheet.cell_value(i+6,j+2)
            input_label_array[j,i] = data_sheet.cell_value(i+6,j+63)

    result_accuracy = accuracy(input_weights_1,input_baises_1,input_weights_2, input_baises_2,input_weights_3,input_baises_3, input_image_array, input_label_array,activation_1,num)
    #output_loss = round(output_loss, 4)
    #result_loss = round(result_loss, 4)
    #for i in range(10):
    #    for j in range(4):
    #        if output_array[i,j] != result_array[i,j]:
    #            print(i)
    #            print(j)
    #print(result_array)
    #print(output_loss)
    #print(result_loss)
    

    if result_accuracy == output_accuracy:
        print("Fuction tested okay")
    if result_accuracy != output_accuracy:
        print("Fuction tested not okay")
        
    return None
def Accuracy_with_layer_0_test():

    print("Function Accuracy for zero hidden layer model is under testing")
    num =0

    #alpha = 0.00001
    #dropout = "n"
    #prob = 0.8
    #reg = "y"
    activation_1 = "n"

    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet108')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    #print(total_rows)
    #print(total_cols)
    input_weights_1 = np.zeros((10,10),dtype = np.float32)
    input_weights_2 = 0
    input_weights_3 = 0
    output_accuracy = data_sheet.cell_value(2,65)

    input_baises_1 = np.zeros((10,1),dtype = np.float32)
    input_baises_2 = 0
    input_baises_3 = 0

    input_image_array = np.zeros((4,10),dtype = np.float32)
    input_label_array = np.zeros((4,10),dtype = np.float32)

    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_weights_1[j,i] = data_sheet.cell_value(i+22,j+1)
            #input_weights_2[j,i] = data_sheet.cell_value(i+35,j+1)
            #input_weights_3[j,i] = data_sheet.cell_value(i+48,j+1)


    for i in range(10):
        input_baises_1[i,0] = data_sheet.cell_value(i+22,12)
        #input_baises_2[i,0] = data_sheet.cell_value(i+35,12)
        #input_baises_3[i,0] = data_sheet.cell_value(i+48,12)


    for i in range(10):
        for j in range(4):
            #print(i)
            #print(j)
            input_image_array[j,i] = data_sheet.cell_value(i+6,j+2)
            input_label_array[j,i] = data_sheet.cell_value(i+6,j+63)

    result_accuracy = accuracy(input_weights_1,input_baises_1,input_weights_2, input_baises_2,input_weights_3,input_baises_3, input_image_array, input_label_array,activation_1,num)
    #output_loss = round(output_loss, 4)
    #result_loss = round(result_loss, 4)
    #for i in range(10):
    #    for j in range(4):
    #        if output_array[i,j] != result_array[i,j]:
    #            print(i)
    #            print(j)
    #print(result_array)
    #print(output_loss)
    #print(result_loss)
    

    if result_accuracy == output_accuracy:
        print("Fuction tested okay")
    if result_accuracy != output_accuracy:
        print("Fuction tested not okay")
        
    return None

def Data_pre_processing_test():

    print("Function Data pre= processing is under testing")
    #alpha = 0.00001
    #dropout = "n"
    #prob = 0.8
    #reg = "y"
    Normal = "simple"
    image_train, one_hot_label_train, image_valid, one_hot_label_valid, image_test, one_hot_label_test = Data_pre_processing(Normal)
    print("Size of image_train array ")
    print(image_train.shape)
    print("Size of one hot train_label array ")
    print(one_hot_label_train.shape)
    print("Size of image_valid array ")
    print(image_valid.shape)
    print("Size of one hot valid_label array ")
    print(one_hot_label_valid.shape)
    print("Size of image_test array ")
    print(image_test.shape)
    print("Size of one hot test_label array ")
    print(one_hot_label_test.shape)
    
    image_array = np.concatenate((image_train, image_valid, image_test), axis=0)
    #image_array = image_array*255
    #image_array =  image_array.astype(np.uint8)
    one_hot_label = np.concatenate((one_hot_label_train, one_hot_label_valid, one_hot_label_test), axis=0)
    x = np.random.randint(1,10, size=70000, dtype=np.int32)
    for i in range(10):
        image = image_array[x[i],:]
        image = np.reshape(image,(28,28))
        label = one_hot_label[x[i],:]
        digit = np.argmax(label, axis =0)
        fig = plt.figure(str(digit))
        imgplot = plt.imshow(image)
        plt.show()
        
    return None

def layer_2_relu_test():

    print("Function with 2 hidden layer with relu activation with SGD optimizer with no reggression with no decaying learning rate is under testing")

    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet19')
    data_sheet1 = data.sheet_by_name('Sheet23')
    data_sheet2 = data.sheet_by_name('Sheet24')
    #prob = 0.8
    activation_1 = "relu"
    reg = "n"
    decay = "n"
    optimizer = "sgd"
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    #print(total_rows)
    #print(total_cols)
    input_weights_1 = np.zeros((10,10),dtype = np.float32)
    input_weights_2 = np.zeros((10,10),dtype = np.float32)
    input_weights_3 = np.zeros((10,10),dtype = np.float32)
    #output_loss = data_sheet.cell_value(1,73)

    input_baises_1 = np.zeros((10,1),dtype = np.float32)
    input_baises_2 = np.zeros((10,1),dtype = np.float32)
    input_baises_3 = np.zeros((10,1),dtype = np.float32)

    input_image_array = np.zeros((4,10),dtype = np.float32)
    input_label_array = np.zeros((4,10),dtype = np.float32)

    output_weights_1 = np.zeros((10,10),dtype = np.float32)
    output_weights_2 = np.zeros((10,10),dtype = np.float32)
    output_weights_3 = np.zeros((10,10),dtype = np.float32)

    output_baises_1 = np.zeros((10,1),dtype = np.float32)
    output_baises_2 = np.zeros((10,1),dtype = np.float32)
    output_baises_3 = np.zeros((10,1),dtype = np.float32)
    output_loss = data_sheet2.cell_value(1,73)
    output_accuracy = 0.25

    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_weights_1[i,j] = data_sheet.cell_value(i+22,j+18)
            input_weights_2[i,j] = data_sheet.cell_value(i+35,j+18)
            input_weights_3[i,j] = data_sheet.cell_value(i+48,j+18)
            output_weights_1[i,j] = data_sheet1.cell_value(i+15,j+167)
            output_weights_2[i,j] = data_sheet1.cell_value(i+33,j+167)
            output_weights_3[i,j] = data_sheet1.cell_value(i+51,j+167)


    for i in range(10):
        input_baises_1[i,0] = data_sheet.cell_value(i+22,12)
        input_baises_2[i,0] = data_sheet.cell_value(i+35,12)
        input_baises_3[i,0] = data_sheet.cell_value(i+48,12)

        output_baises_1[i,0] = data_sheet1.cell_value(11,i+167)
        output_baises_2[i,0] = data_sheet1.cell_value(29,i+167)
        output_baises_3[i,0] = data_sheet1.cell_value(47,i+167)


    for i in range(10):
        for j in range(4):
            #print(i)
            #print(j)
            input_image_array[j,i] = data_sheet.cell_value(i+6,j+2)
            input_label_array[j,i] = data_sheet.cell_value(i+6,j+63)

    result_weights_1,result_baises_1,result_weights_2,result_baises_2,result_weights_3,result_baises_3,result_loss, result_accuracy = NN_2_layers_test(input_weights_1,input_baises_1,input_weights_2, input_baises_2,input_weights_3,input_baises_3, input_image_array, input_label_array, activation_1,reg,decay,optimizer )

    result_weights_1 = np.round(result_weights_1 ,4)
    output_weights_1 = np.round(output_weights_1, 4)
    result_weights_2 = np.round(result_weights_2 ,4)
    output_weights_2 = np.round(output_weights_2, 4)
    result_weights_3 = np.round(result_weights_3 ,4)
    output_weights_3 = np.round(output_weights_3, 4)


    result_baises_1 = np.round(result_baises_1 ,4)
    output_baises_1 = np.round(output_baises_1, 4) 
    result_baises_2 = np.round(result_baises_2 ,4)
    output_baises_2 = np.round(output_baises_2, 4) 
    result_baises_3 = np.round(result_baises_3 ,4)
    output_baises_3 = np.round(output_baises_3, 4)

    output_loss = round(output_loss, 4)
    result_loss = round(result_loss, 4)
    #print(result_baises_3)
    #print(output_baises_3)
    
    #if ((result_baises_3 == output_baises_3).all()):
    #    print("Fuction tested okay")
    #if ((result_baises_3 == output_baises_3).all()) == False:
    #    print("Fuction tested not okay")
        
    if (((result_weights_1 == output_weights_1).all()) and ((result_weights_2 == output_weights_2).all()) and ((result_weights_3 == output_weights_3).all()) and ((result_baises_1 == output_baises_1).all()) and ((result_baises_2 == output_baises_2).all()) and ((result_baises_3 == output_baises_3).all()) and result_loss == output_loss and result_accuracy == output_accuracy):
        print("Fuction tested okay")
    if (((result_weights_1 == output_weights_1).all()) and ((result_weights_2 == output_weights_2).all()) and ((result_weights_3 == output_weights_3).all()) and ((result_baises_1 == output_baises_1).all()) and ((result_baises_2 == output_baises_2).all()) and ((result_baises_3 == output_baises_3).all()) and result_loss == output_loss and result_accuracy == output_accuracy) == False:
        print("Fuction tested not okay")
    return None

def layer_1_relu_test():

    print("Function with 1 hidden layer with relu activation with SGD optimizer with no reggression with no decaying learning rate is under testing")

    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet49')
    data_sheet1 = data.sheet_by_name('Sheet53')
    data_sheet2 = data.sheet_by_name('Sheet54')
    activation_1 = "relu"
    reg = "n"
    decay = "n"
    optimizer = "sgd"
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    #print(total_rows)
    #print(total_cols)
    input_weights_1 = np.zeros((10,10),dtype = np.float32)
    input_weights_2 = np.zeros((10,10),dtype = np.float32)
    #input_weights_3 = np.zeros((10,10),dtype = np.float32)
    #output_loss = data_sheet.cell_value(1,73)

    input_baises_1 = np.zeros((10,1),dtype = np.float32)
    input_baises_2 = np.zeros((10,1),dtype = np.float32)
    #input_baises_3 = np.zeros((10,1),dtype = np.float32)

    input_image_array = np.zeros((4,10),dtype = np.float32)
    input_label_array = np.zeros((4,10),dtype = np.float32)

    output_weights_1 = np.zeros((10,10),dtype = np.float32)
    output_weights_2 = np.zeros((10,10),dtype = np.float32)
    #output_weights_3 = np.zeros((10,10),dtype = np.float32)

    output_baises_1 = np.zeros((10,1),dtype = np.float32)
    output_baises_2 = np.zeros((10,1),dtype = np.float32)
    #output_baises_3 = np.zeros((10,1),dtype = np.float32)
    output_loss = data_sheet2.cell_value(1,73)
    output_accuracy = 0.0

    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_weights_1[i,j] = data_sheet.cell_value(i+22,j+18)
            input_weights_2[i,j] = data_sheet.cell_value(i+35,j+18)
            #input_weights_3[i,j] = data_sheet.cell_value(i+48,j+18)
            output_weights_1[i,j] = data_sheet1.cell_value(i+15,j+167)
            output_weights_2[i,j] = data_sheet1.cell_value(i+33,j+167)
            #output_weights_3[i,j] = data_sheet1.cell_value(i+51,j+167)


    for i in range(10):
        input_baises_1[i,0] = data_sheet.cell_value(i+22,12)
        input_baises_2[i,0] = data_sheet.cell_value(i+35,12)
        #input_baises_3[i,0] = data_sheet.cell_value(i+48,12)

        output_baises_1[i,0] = data_sheet1.cell_value(11,i+167)
        output_baises_2[i,0] = data_sheet1.cell_value(29,i+167)
        #output_baises_3[i,0] = data_sheet1.cell_value(47,i+167)


    for i in range(10):
        for j in range(4):
            #print(i)
            #print(j)
            input_image_array[j,i] = data_sheet.cell_value(i+6,j+2)
            input_label_array[j,i] = data_sheet.cell_value(i+6,j+63)

    result_weights_1,result_baises_1,result_weights_2,result_baises_2,result_loss, result_accuracy = NN_1_layers_test(input_weights_1,input_baises_1,input_weights_2, input_baises_2, input_image_array, input_label_array, activation_1,reg,decay,optimizer)

    result_weights_1 = np.round(result_weights_1 ,4)
    output_weights_1 = np.round(output_weights_1, 4)
    result_weights_2 = np.round(result_weights_2 ,4)
    output_weights_2 = np.round(output_weights_2, 4)
    #result_weights_3 = np.round(result_weights_3 ,4)
    #output_weights_3 = np.round(output_weights_3, 4)


    result_baises_1 = np.round(result_baises_1 ,4)
    output_baises_1 = np.round(output_baises_1, 4) 
    result_baises_2 = np.round(result_baises_2 ,4)
    output_baises_2 = np.round(output_baises_2, 4) 
    #result_baises_3 = np.round(result_baises_3 ,4)
    #output_baises_3 = np.round(output_baises_3, 4)

    output_loss = round(output_loss, 4)
    result_loss = round(result_loss, 4)
    #print(result_accuracy)
    #print(result_loss)
    
    #if ((result_baises_3 == output_baises_3).all()):
    #    print("Fuction tested okay")
    #if ((result_baises_3 == output_baises_3).all()) == False:
    #    print("Fuction tested not okay")
        
    if (((result_weights_1 == output_weights_1).all()) and ((result_weights_2 == output_weights_2).all()) and ((result_baises_1 == output_baises_1).all()) and ((result_baises_2 == output_baises_2).all()) and result_loss == output_loss and result_accuracy == output_accuracy):
        print("Fuction tested okay")
    if (((result_weights_1 == output_weights_1).all()) and ((result_weights_2 == output_weights_2).all()) and  ((result_baises_1 == output_baises_1).all()) and ((result_baises_2 == output_baises_2).all()) and  result_loss == output_loss and result_accuracy == output_accuracy) == False:
        print("Fuction tested not okay")
    return None

def layer_0_test():

    print("Function with zero hidden layer with SGD optimizer with no reggression with no decaying learning rate is under testing")

    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet79')
    data_sheet1 = data.sheet_by_name('Sheet83')
    data_sheet2 = data.sheet_by_name('Sheet84')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    activation_1 = "n"
    reg = "n"
    decay = "n"
    optimizer = "sgd"
    #print(total_rows)
    #print(total_cols)
    input_weights_1 = np.zeros((10,10),dtype = np.float32)
    #input_weights_2 = np.zeros((10,10),dtype = np.float32)
    #input_weights_3 = np.zeros((10,10),dtype = np.float32)
    #output_loss = data_sheet.cell_value(1,73)

    input_baises_1 = np.zeros((10,1),dtype = np.float32)
    #input_baises_2 = np.zeros((10,1),dtype = np.float32)
    #input_baises_3 = np.zeros((10,1),dtype = np.float32)

    input_image_array = np.zeros((4,10),dtype = np.float32)
    input_label_array = np.zeros((4,10),dtype = np.float32)

    output_weights_1 = np.zeros((10,10),dtype = np.float32)
    #output_weights_2 = np.zeros((10,10),dtype = np.float32)
    #output_weights_3 = np.zeros((10,10),dtype = np.float32)

    output_baises_1 = np.zeros((10,1),dtype = np.float32)
    #output_baises_2 = np.zeros((10,1),dtype = np.float32)
    #output_baises_3 = np.zeros((10,1),dtype = np.float32)
    output_loss = data_sheet2.cell_value(1,73)
    output_accuracy = 0.0

    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_weights_1[i,j] = data_sheet.cell_value(i+22,j+18)
            #input_weights_2[i,j] = data_sheet.cell_value(i+35,j+18)
            #input_weights_3[i,j] = data_sheet.cell_value(i+48,j+18)
            output_weights_1[i,j] = data_sheet1.cell_value(i+15,j+167)
            #output_weights_2[i,j] = data_sheet1.cell_value(i+33,j+167)
            #output_weights_3[i,j] = data_sheet1.cell_value(i+51,j+167)


    for i in range(10):
        input_baises_1[i,0] = data_sheet.cell_value(i+22,12)
        #input_baises_2[i,0] = data_sheet.cell_value(i+35,12)
        #input_baises_3[i,0] = data_sheet.cell_value(i+48,12)

        output_baises_1[i,0] = data_sheet1.cell_value(11,i+167)
        #output_baises_2[i,0] = data_sheet1.cell_value(29,i+167)
        #output_baises_3[i,0] = data_sheet1.cell_value(47,i+167)


    for i in range(10):
        for j in range(4):
            #print(i)
            #print(j)
            input_image_array[j,i] = data_sheet.cell_value(i+6,j+2)
            input_label_array[j,i] = data_sheet.cell_value(i+6,j+63)

    result_weights_1,result_baises_1,result_loss, result_accuracy = NN_0_layers_test(input_weights_1,input_baises_1,input_image_array, input_label_array,activation_1,reg,decay,optimizer)

    result_weights_1 = np.round(result_weights_1 ,4)
    output_weights_1 = np.round(output_weights_1, 4)
    #result_weights_2 = np.round(result_weights_2 ,4)
    #output_weights_2 = np.round(output_weights_2, 4)
    #result_weights_3 = np.round(result_weights_3 ,4)
    #output_weights_3 = np.round(output_weights_3, 4)


    result_baises_1 = np.round(result_baises_1 ,4)
    output_baises_1 = np.round(output_baises_1, 4) 
    #result_baises_2 = np.round(result_baises_2 ,4)
    #output_baises_2 = np.round(output_baises_2, 4) 
    #result_baises_3 = np.round(result_baises_3 ,4)
    #output_baises_3 = np.round(output_baises_3, 4)

    output_loss = round(output_loss, 4)
    result_loss = round(result_loss, 4)
    #print(result_accuracy)
    #print(result_loss)
    
    #if ((result_baises_3 == output_baises_3).all()):
    #    print("Fuction tested okay")
    #if ((result_baises_3 == output_baises_3).all()) == False:
    #    print("Fuction tested not okay")
        
    if (((result_weights_1 == output_weights_1).all()) and ((result_baises_1 == output_baises_1).all()) and result_loss == output_loss and result_accuracy == output_accuracy ):
        print("Fuction tested okay")
    if (((result_weights_1 == output_weights_1).all()) and  ((result_baises_1 == output_baises_1).all()) and result_loss == output_loss and result_accuracy == output_accuracy) == False:
        print("Fuction tested not okay")
    return None


def layer_2_sigmoid_test():

    print("Function with 2 hidden layer with Sigmoid activation with SGD optimizer with no reggression with no decaying learning rate is under testing")

    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet25')
    data_sheet1 = data.sheet_by_name('Sheet29')
    data_sheet2 = data.sheet_by_name('Sheet30')
    activation_1 = "sigmoid"
    reg = "n"
    decay = "n"
    optimizer = "sgd"
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    #print(total_rows)
    #print(total_cols)
    input_weights_1 = np.zeros((10,10),dtype = np.float32)
    input_weights_2 = np.zeros((10,10),dtype = np.float32)
    input_weights_3 = np.zeros((10,10),dtype = np.float32)
    #output_loss = data_sheet.cell_value(1,73)

    input_baises_1 = np.zeros((10,1),dtype = np.float32)
    input_baises_2 = np.zeros((10,1),dtype = np.float32)
    input_baises_3 = np.zeros((10,1),dtype = np.float32)

    input_image_array = np.zeros((4,10),dtype = np.float32)
    input_label_array = np.zeros((4,10),dtype = np.float32)

    output_weights_1 = np.zeros((10,10),dtype = np.float32)
    output_weights_2 = np.zeros((10,10),dtype = np.float32)
    output_weights_3 = np.zeros((10,10),dtype = np.float32)

    output_baises_1 = np.zeros((10,1),dtype = np.float32)
    output_baises_2 = np.zeros((10,1),dtype = np.float32)
    output_baises_3 = np.zeros((10,1),dtype = np.float32)
    output_loss = data_sheet2.cell_value(1,73)
    output_accuracy = 0.25

    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_weights_1[i,j] = data_sheet.cell_value(i+22,j+18)
            input_weights_2[i,j] = data_sheet.cell_value(i+35,j+18)
            input_weights_3[i,j] = data_sheet.cell_value(i+48,j+18)
            output_weights_1[i,j] = data_sheet1.cell_value(i+15,j+167)
            output_weights_2[i,j] = data_sheet1.cell_value(i+33,j+167)
            output_weights_3[i,j] = data_sheet1.cell_value(i+51,j+167)


    for i in range(10):
        input_baises_1[i,0] = data_sheet.cell_value(i+22,12)
        input_baises_2[i,0] = data_sheet.cell_value(i+35,12)
        input_baises_3[i,0] = data_sheet.cell_value(i+48,12)

        output_baises_1[i,0] = data_sheet1.cell_value(11,i+167)
        output_baises_2[i,0] = data_sheet1.cell_value(29,i+167)
        output_baises_3[i,0] = data_sheet1.cell_value(47,i+167)


    for i in range(10):
        for j in range(4):
            #print(i)
            #print(j)
            input_image_array[j,i] = data_sheet.cell_value(i+6,j+2)
            input_label_array[j,i] = data_sheet.cell_value(i+6,j+63)

    result_weights_1,result_baises_1,result_weights_2,result_baises_2,result_weights_3,result_baises_3,result_loss, result_accuracy = NN_2_layers_test(input_weights_1,input_baises_1,input_weights_2, input_baises_2,input_weights_3,input_baises_3, input_image_array, input_label_array, activation_1,reg,decay,optimizer)

    result_weights_1 = np.round(result_weights_1 ,4)
    output_weights_1 = np.round(output_weights_1, 4)
    result_weights_2 = np.round(result_weights_2 ,4)
    output_weights_2 = np.round(output_weights_2, 4)
    result_weights_3 = np.round(result_weights_3 ,4)
    output_weights_3 = np.round(output_weights_3, 4)


    result_baises_1 = np.round(result_baises_1 ,4)
    output_baises_1 = np.round(output_baises_1, 4) 
    result_baises_2 = np.round(result_baises_2 ,4)
    output_baises_2 = np.round(output_baises_2, 4) 
    result_baises_3 = np.round(result_baises_3 ,4)
    output_baises_3 = np.round(output_baises_3, 4)

    output_loss = round(output_loss, 4)
    result_loss = round(result_loss, 4)
    #print(result_baises_3)
    #print(output_baises_3)
    
    #if ((result_baises_3 == output_baises_3).all()):
    #    print("Fuction tested okay")
    #if ((result_baises_3 == output_baises_3).all()) == False:
    #    print("Fuction tested not okay")
        
    if (((result_weights_1 == output_weights_1).all()) and ((result_weights_2 == output_weights_2).all()) and ((result_weights_3 == output_weights_3).all()) and ((result_baises_1 == output_baises_1).all()) and ((result_baises_2 == output_baises_2).all()) and ((result_baises_3 == output_baises_3).all()) and result_loss == output_loss and result_accuracy == output_accuracy):
        print("Fuction tested okay")
    if (((result_weights_1 == output_weights_1).all()) and ((result_weights_2 == output_weights_2).all()) and ((result_weights_3 == output_weights_3).all()) and ((result_baises_1 == output_baises_1).all()) and ((result_baises_2 == output_baises_2).all()) and ((result_baises_3 == output_baises_3).all()) and result_loss == output_loss and result_accuracy == output_accuracy) == False:
        print("Fuction tested not okay")
    return None

def layer_1_sigmoid_test():

    print("Function with 1 hidden layer with Sigmoid activation with SGD optimizer with no reggression with no decaying learning rate is under testing")

    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet55')
    data_sheet1 = data.sheet_by_name('Sheet59')
    data_sheet2 = data.sheet_by_name('Sheet60')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    activation_1 = "sigmoid"
    reg = "n"
    decay = "n"
    optimizer = "sgd"
    #print(total_rows)
    #print(total_cols)
    input_weights_1 = np.zeros((10,10),dtype = np.float32)
    input_weights_2 = np.zeros((10,10),dtype = np.float32)
    #input_weights_3 = np.zeros((10,10),dtype = np.float32)
    #output_loss = data_sheet.cell_value(1,73)

    input_baises_1 = np.zeros((10,1),dtype = np.float32)
    input_baises_2 = np.zeros((10,1),dtype = np.float32)
    #input_baises_3 = np.zeros((10,1),dtype = np.float32)

    input_image_array = np.zeros((4,10),dtype = np.float32)
    input_label_array = np.zeros((4,10),dtype = np.float32)

    output_weights_1 = np.zeros((10,10),dtype = np.float32)
    output_weights_2 = np.zeros((10,10),dtype = np.float32)
    #output_weights_3 = np.zeros((10,10),dtype = np.float32)

    output_baises_1 = np.zeros((10,1),dtype = np.float32)
    output_baises_2 = np.zeros((10,1),dtype = np.float32)
    #output_baises_3 = np.zeros((10,1),dtype = np.float32)
    output_loss = data_sheet2.cell_value(1,73)
    output_accuracy = 0.0

    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_weights_1[i,j] = data_sheet.cell_value(i+22,j+18)
            input_weights_2[i,j] = data_sheet.cell_value(i+35,j+18)
            #input_weights_3[i,j] = data_sheet.cell_value(i+48,j+18)
            output_weights_1[i,j] = data_sheet1.cell_value(i+15,j+167)
            output_weights_2[i,j] = data_sheet1.cell_value(i+33,j+167)
            #output_weights_3[i,j] = data_sheet1.cell_value(i+51,j+167)


    for i in range(10):
        input_baises_1[i,0] = data_sheet.cell_value(i+22,12)
        input_baises_2[i,0] = data_sheet.cell_value(i+35,12)
        #input_baises_3[i,0] = data_sheet.cell_value(i+48,12)

        output_baises_1[i,0] = data_sheet1.cell_value(11,i+167)
        output_baises_2[i,0] = data_sheet1.cell_value(29,i+167)
        #output_baises_3[i,0] = data_sheet1.cell_value(47,i+167)


    for i in range(10):
        for j in range(4):
            #print(i)
            #print(j)
            input_image_array[j,i] = data_sheet.cell_value(i+6,j+2)
            input_label_array[j,i] = data_sheet.cell_value(i+6,j+63)

    result_weights_1,result_baises_1,result_weights_2,result_baises_2,result_loss, result_accuracy = NN_1_layers_test(input_weights_1,input_baises_1,input_weights_2, input_baises_2, input_image_array, input_label_array, activation_1,reg,decay,optimizer)

    result_weights_1 = np.round(result_weights_1 ,4)
    output_weights_1 = np.round(output_weights_1, 4)
    result_weights_2 = np.round(result_weights_2 ,4)
    output_weights_2 = np.round(output_weights_2, 4)
    #result_weights_3 = np.round(result_weights_3 ,4)
    #output_weights_3 = np.round(output_weights_3, 4)


    result_baises_1 = np.round(result_baises_1 ,4)
    output_baises_1 = np.round(output_baises_1, 4) 
    result_baises_2 = np.round(result_baises_2 ,4)
    output_baises_2 = np.round(output_baises_2, 4) 
    #result_baises_3 = np.round(result_baises_3 ,4)
    #output_baises_3 = np.round(output_baises_3, 4)

    output_loss = round(output_loss, 4)
    result_loss = round(result_loss, 4)
    #print(result_accuracy)
    #print(result_loss)
    
    #if ((result_baises_3 == output_baises_3).all()):
    #    print("Fuction tested okay")
    #if ((result_baises_3 == output_baises_3).all()) == False:
    #    print("Fuction tested not okay")
        
    if (((result_weights_1 == output_weights_1).all()) and ((result_weights_2 == output_weights_2).all()) and ((result_baises_1 == output_baises_1).all()) and ((result_baises_2 == output_baises_2).all()) and result_loss == output_loss and result_accuracy == output_accuracy):
        print("Fuction tested okay")
    if (((result_weights_1 == output_weights_1).all()) and ((result_weights_2 == output_weights_2).all()) and  ((result_baises_1 == output_baises_1).all()) and ((result_baises_2 == output_baises_2).all()) and  result_loss == output_loss and result_accuracy == output_accuracy) == False:
        print("Fuction tested not okay")
    return None

def layer_2_momentum_opt_test():

    print("Function with 2 hidden layer with relu activation with Momentum optimizer with no reggression with no decaying learning rate is under testing")

    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet31')
    data_sheet1 = data.sheet_by_name('Sheet35')
    data_sheet2 = data.sheet_by_name('Sheet36')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    activation_1 = "relu"
    reg = "n"
    decay = "n"
    optimizer = "momentum"
    #print(total_rows)
    #print(total_cols)
    input_weights_1 = np.zeros((10,10),dtype = np.float32)
    input_weights_2 = np.zeros((10,10),dtype = np.float32)
    input_weights_3 = np.zeros((10,10),dtype = np.float32)
    #output_loss = data_sheet.cell_value(1,73)

    input_baises_1 = np.zeros((10,1),dtype = np.float32)
    input_baises_2 = np.zeros((10,1),dtype = np.float32)
    input_baises_3 = np.zeros((10,1),dtype = np.float32)

    input_image_array = np.zeros((4,10),dtype = np.float32)
    input_label_array = np.zeros((4,10),dtype = np.float32)

    output_weights_1 = np.zeros((10,10),dtype = np.float32)
    output_weights_2 = np.zeros((10,10),dtype = np.float32)
    output_weights_3 = np.zeros((10,10),dtype = np.float32)

    output_baises_1 = np.zeros((10,1),dtype = np.float32)
    output_baises_2 = np.zeros((10,1),dtype = np.float32)
    output_baises_3 = np.zeros((10,1),dtype = np.float32)
    output_loss = data_sheet2.cell_value(1,73)
    output_accuracy = 0

    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_weights_1[i,j] = data_sheet.cell_value(i+22,j+18)
            input_weights_2[i,j] = data_sheet.cell_value(i+35,j+18)
            input_weights_3[i,j] = data_sheet.cell_value(i+48,j+18)
            output_weights_1[i,j] = data_sheet1.cell_value(i+15,j+191)
            output_weights_2[i,j] = data_sheet1.cell_value(i+33,j+191)
            output_weights_3[i,j] = data_sheet1.cell_value(i+51,j+191)


    for i in range(10):
        input_baises_1[i,0] = data_sheet.cell_value(i+22,12)
        input_baises_2[i,0] = data_sheet.cell_value(i+35,12)
        input_baises_3[i,0] = data_sheet.cell_value(i+48,12)

        output_baises_1[i,0] = data_sheet1.cell_value(11,i+191)
        output_baises_2[i,0] = data_sheet1.cell_value(29,i+191)
        output_baises_3[i,0] = data_sheet1.cell_value(47,i+191)


    for i in range(10):
        for j in range(4):
            #print(i)
            #print(j)
            input_image_array[j,i] = data_sheet.cell_value(i+6,j+2)
            input_label_array[j,i] = data_sheet.cell_value(i+6,j+63)

    #print(input_baises_1)
    #print(input_baises_2)
    #print(input_baises_3)
    
    result_weights_1,result_baises_1,result_weights_2,result_baises_2,result_weights_3,result_baises_3,result_loss, result_accuracy = NN_2_layers_test(input_weights_1,input_baises_1,input_weights_2, input_baises_2,input_weights_3,input_baises_3, input_image_array, input_label_array, activation_1,reg,decay,optimizer)

    result_weights_1 = np.round(result_weights_1 ,3)
    #print(result_weights_1)
    output_weights_1 = np.round(output_weights_1, 3)
    #print(output_weights_1)
    result_weights_2 = np.round(result_weights_2 ,3)
    #print(result_weights_2)
    output_weights_2 = np.round(output_weights_2, 3)
    result_weights_3 = np.round(result_weights_3 ,3)
    output_weights_3 = np.round(output_weights_3, 3)


    result_baises_1 = np.round(result_baises_1 ,3)
    output_baises_1 = np.round(output_baises_1, 3) 
    result_baises_2 = np.round(result_baises_2 ,3)
    output_baises_2 = np.round(output_baises_2, 3) 
    result_baises_3 = np.round(result_baises_3 ,3)
    output_baises_3 = np.round(output_baises_3, 3)

    output_loss = round(output_loss, 4)
    result_loss = round(result_loss, 4)
    #print(result_loss)
    #print(output_loss)
    #print(result_accuracy)
    #print(output_accuracy)
    #for i in range(10):
    #    for j in range(10):
    #        if (result_weights_1[i,j] == output_weights_1[i,j]) == False:
    #            print(i)
    #            print(j)
    #if ((result_weights_2 == output_weights_2).all()):
    #    print("Fuction tested okay")
    #if ((result_weights_2 == output_weights_2).all()) == False:
    #    print("Fuction tested not okay")
    #if ((result_weights_3 == output_weights_3).all()):
    #    print("Fuction tested okay")
    #if ((result_weights_3 == output_weights_3).all()) == False:
    #    print("Fuction tested not okay")
    #if ((result_weights_1 == output_weights_1).all()):
    #    print("Fuction tested okay")
    #if ((result_weights_1 == output_weights_1).all()) == False:
    #    print("Fuction tested not okay")
    #if ((result_baises_1 == output_baises_1).all()):
    #    print("Fuction tested okay")
    #if ((result_baises_1 == output_baises_1).all()) == False:
    #    print("Fuction tested not okay")
    #if ((result_baises_2 == output_baises_2).all()):
    #    print("Fuction tested okay")
    #if ((result_baises_2 == output_baises_2).all()) == False:
    #    print("Fuction tested not okay")
    #if ((result_baises_3 == output_baises_3).all()):
    #    print("Fuction tested okay")
    #if ((result_baises_3 == output_baises_3).all()) == False:
    #    print("Fuction tested not okay")
        
    if (((result_weights_1 == output_weights_1).all()) and ((result_weights_2 == output_weights_2).all()) and ((result_weights_3 == output_weights_3).all()) and ((result_baises_1 == output_baises_1).all()) and ((result_baises_2 == output_baises_2).all()) and ((result_baises_3 == output_baises_3).all()) and result_loss == output_loss and result_accuracy == output_accuracy):
        print("Fuction tested okay")
    if (((result_weights_1 == output_weights_1).all()) and ((result_weights_2 == output_weights_2).all()) and ((result_weights_3 == output_weights_3).all()) and ((result_baises_1 == output_baises_1).all()) and ((result_baises_2 == output_baises_2).all()) and ((result_baises_3 == output_baises_3).all()) and result_loss == output_loss and result_accuracy == output_accuracy) == False:
        print("Fuction tested not okay")
    return None

def layer_1_momentum_opt_test():

    print("Function with 1 hidden layer with relu activation with Momentum optimizer with no reggression with no decaying learning rate is under testing")

    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet61')
    data_sheet1 = data.sheet_by_name('Sheet65')
    data_sheet2 = data.sheet_by_name('Sheet66')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols

    activation_1 = "relu"
    reg = "n"
    decay = "n"
    optimizer = "momentum"
    #print(total_rows)
    #print(total_cols)
    input_weights_1 = np.zeros((10,10),dtype = np.float32)
    input_weights_2 = np.zeros((10,10),dtype = np.float32)
    #input_weights_3 = np.zeros((10,10),dtype = np.float32)
    #output_loss = data_sheet.cell_value(1,73)

    input_baises_1 = np.zeros((10,1),dtype = np.float32)
    input_baises_2 = np.zeros((10,1),dtype = np.float32)
    #input_baises_3 = np.zeros((10,1),dtype = np.float32)

    input_image_array = np.zeros((4,10),dtype = np.float32)
    input_label_array = np.zeros((4,10),dtype = np.float32)

    output_weights_1 = np.zeros((10,10),dtype = np.float32)
    output_weights_2 = np.zeros((10,10),dtype = np.float32)
    #output_weights_3 = np.zeros((10,10),dtype = np.float32)

    output_baises_1 = np.zeros((10,1),dtype = np.float32)
    output_baises_2 = np.zeros((10,1),dtype = np.float32)
    #output_baises_3 = np.zeros((10,1),dtype = np.float32)
    output_loss = data_sheet2.cell_value(1,73)
    output_accuracy = 0.25

    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_weights_1[i,j] = data_sheet.cell_value(i+22,j+18)
            input_weights_2[i,j] = data_sheet.cell_value(i+35,j+18)
            #input_weights_3[i,j] = data_sheet.cell_value(i+48,j+18)
            output_weights_1[i,j] = data_sheet1.cell_value(i+15,j+191)
            output_weights_2[i,j] = data_sheet1.cell_value(i+33,j+191)
            #output_weights_3[i,j] = data_sheet1.cell_value(i+51,j+167)


    for i in range(10):
        input_baises_1[i,0] = data_sheet.cell_value(i+22,12)
        input_baises_2[i,0] = data_sheet.cell_value(i+35,12)
        #input_baises_3[i,0] = data_sheet.cell_value(i+48,12)

        output_baises_1[i,0] = data_sheet1.cell_value(11,i+191)
        output_baises_2[i,0] = data_sheet1.cell_value(29,i+191)
        #output_baises_3[i,0] = data_sheet1.cell_value(47,i+167)


    for i in range(10):
        for j in range(4):
            #print(i)
            #print(j)
            input_image_array[j,i] = data_sheet.cell_value(i+6,j+2)
            input_label_array[j,i] = data_sheet.cell_value(i+6,j+63)

    result_weights_1,result_baises_1,result_weights_2,result_baises_2,result_loss, result_accuracy = NN_1_layers_test(input_weights_1,input_baises_1,input_weights_2, input_baises_2, input_image_array, input_label_array, activation_1,reg,decay,optimizer)

    result_weights_1 = np.round(result_weights_1 ,3)
    output_weights_1 = np.round(output_weights_1, 3)
    result_weights_2 = np.round(result_weights_2 ,3)
    output_weights_2 = np.round(output_weights_2, 3)
    #result_weights_3 = np.round(result_weights_3 ,4)
    #output_weights_3 = np.round(output_weights_3, 4)


    result_baises_1 = np.round(result_baises_1 ,3)
    output_baises_1 = np.round(output_baises_1, 3) 
    result_baises_2 = np.round(result_baises_2 ,3)
    output_baises_2 = np.round(output_baises_2, 3) 
    #result_baises_3 = np.round(result_baises_3 ,4)
    #output_baises_3 = np.round(output_baises_3, 4)

    output_loss = round(output_loss, 4)
    result_loss = round(result_loss, 4)
    #print(result_accuracy)
    #print(result_loss)
    
    #if ((result_baises_3 == output_baises_3).all()):
    #    print("Fuction tested okay")
    #if ((result_baises_3 == output_baises_3).all()) == False:
    #    print("Fuction tested not okay")
        
    if (((result_weights_1 == output_weights_1).all()) and ((result_weights_2 == output_weights_2).all()) and ((result_baises_1 == output_baises_1).all()) and ((result_baises_2 == output_baises_2).all()) and result_loss == output_loss and result_accuracy == output_accuracy):
        print("Fuction tested okay")
    if (((result_weights_1 == output_weights_1).all()) and ((result_weights_2 == output_weights_2).all()) and  ((result_baises_1 == output_baises_1).all()) and ((result_baises_2 == output_baises_2).all()) and  result_loss == output_loss and result_accuracy == output_accuracy) == False:
        print("Fuction tested not okay")
    return None

def layer_0_momentum_opt_test():

    print("Function with zero hidden layer with Momentum optimizer with no reggression with no decaying learning rate is under testing")

    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet85')
    data_sheet1 = data.sheet_by_name('Sheet89')
    data_sheet2 = data.sheet_by_name('Sheet90')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    activation_1 = "no"
    reg = "n"
    decay = "n"
    optimizer = "momentum"
    #print(total_rows)
    #print(total_cols)
    input_weights_1 = np.zeros((10,10),dtype = np.float32)
    #input_weights_2 = np.zeros((10,10),dtype = np.float32)
    #input_weights_3 = np.zeros((10,10),dtype = np.float32)
    #output_loss = data_sheet.cell_value(1,73)

    input_baises_1 = np.zeros((10,1),dtype = np.float32)
    #input_baises_2 = np.zeros((10,1),dtype = np.float32)
    #input_baises_3 = np.zeros((10,1),dtype = np.float32)

    input_image_array = np.zeros((4,10),dtype = np.float32)
    input_label_array = np.zeros((4,10),dtype = np.float32)

    output_weights_1 = np.zeros((10,10),dtype = np.float32)
    #output_weights_2 = np.zeros((10,10),dtype = np.float32)
    #output_weights_3 = np.zeros((10,10),dtype = np.float32)

    output_baises_1 = np.zeros((10,1),dtype = np.float32)
    #output_baises_2 = np.zeros((10,1),dtype = np.float32)
    #output_baises_3 = np.zeros((10,1),dtype = np.float32)
    output_loss = data_sheet2.cell_value(1,73)
    output_accuracy = 0.0

    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_weights_1[i,j] = data_sheet.cell_value(i+22,j+18)
            #input_weights_2[i,j] = data_sheet.cell_value(i+35,j+18)
            #input_weights_3[i,j] = data_sheet.cell_value(i+48,j+18)
            output_weights_1[i,j] = data_sheet1.cell_value(i+15,j+191)
            #output_weights_2[i,j] = data_sheet1.cell_value(i+33,j+167)
            #output_weights_3[i,j] = data_sheet1.cell_value(i+51,j+167)


    for i in range(10):
        input_baises_1[i,0] = data_sheet.cell_value(i+22,12)
        #input_baises_2[i,0] = data_sheet.cell_value(i+35,12)
        #input_baises_3[i,0] = data_sheet.cell_value(i+48,12)

        output_baises_1[i,0] = data_sheet1.cell_value(11,i+191)
        #output_baises_2[i,0] = data_sheet1.cell_value(29,i+167)
        #output_baises_3[i,0] = data_sheet1.cell_value(47,i+167)


    for i in range(10):
        for j in range(4):
            #print(i)
            #print(j)
            input_image_array[j,i] = data_sheet.cell_value(i+6,j+2)
            input_label_array[j,i] = data_sheet.cell_value(i+6,j+63)

    result_weights_1,result_baises_1,result_loss, result_accuracy = NN_0_layers_test(input_weights_1,input_baises_1,input_image_array, input_label_array, activation_1,reg,decay,optimizer)

    result_weights_1 = np.round(result_weights_1 ,3)
    output_weights_1 = np.round(output_weights_1, 3)
    #result_weights_2 = np.round(result_weights_2 ,4)
    #output_weights_2 = np.round(output_weights_2, 4)
    #result_weights_3 = np.round(result_weights_3 ,4)
    #output_weights_3 = np.round(output_weights_3, 4)


    result_baises_1 = np.round(result_baises_1 ,3)
    output_baises_1 = np.round(output_baises_1, 3) 
    #result_baises_2 = np.round(result_baises_2 ,4)
    #output_baises_2 = np.round(output_baises_2, 4) 
    #result_baises_3 = np.round(result_baises_3 ,4)
    #output_baises_3 = np.round(output_baises_3, 4)

    output_loss = round(output_loss, 4)
    result_loss = round(result_loss, 4)
    #print(result_accuracy)
    #print(result_loss)
    
    #if ((result_baises_3 == output_baises_3).all()):
    #    print("Fuction tested okay")
    #if ((result_baises_3 == output_baises_3).all()) == False:
    #    print("Fuction tested not okay")
        
    if (((result_weights_1 == output_weights_1).all()) and ((result_baises_1 == output_baises_1).all()) and result_loss == output_loss and result_accuracy == output_accuracy ):
        print("Fuction tested okay")
    if (((result_weights_1 == output_weights_1).all()) and  ((result_baises_1 == output_baises_1).all()) and result_loss == output_loss and result_accuracy == output_accuracy) == False:
        print("Fuction tested not okay")
    return None

def layer_2_with_learning_rate_decay_test():

    print("Function with 2 hidden layer with Relu activation with SGD optimizer with no reggression with decaying learning rate is under testing")

    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet37')
    data_sheet1 = data.sheet_by_name('Sheet41')
    data_sheet2 = data.sheet_by_name('Sheet42')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    activation_1 = "relu"
    reg = "n"
    decay = "y"
    optimizer = "sgd"
    #print(total_rows)
    #print(total_cols)
    input_weights_1 = np.zeros((10,10),dtype = np.float32)
    input_weights_2 = np.zeros((10,10),dtype = np.float32)
    input_weights_3 = np.zeros((10,10),dtype = np.float32)
    #output_loss = data_sheet.cell_value(1,73)

    input_baises_1 = np.zeros((10,1),dtype = np.float32)
    input_baises_2 = np.zeros((10,1),dtype = np.float32)
    input_baises_3 = np.zeros((10,1),dtype = np.float32)

    input_image_array = np.zeros((4,10),dtype = np.float32)
    input_label_array = np.zeros((4,10),dtype = np.float32)

    output_weights_1 = np.zeros((10,10),dtype = np.float32)
    output_weights_2 = np.zeros((10,10),dtype = np.float32)
    output_weights_3 = np.zeros((10,10),dtype = np.float32)

    output_baises_1 = np.zeros((10,1),dtype = np.float32)
    output_baises_2 = np.zeros((10,1),dtype = np.float32)
    output_baises_3 = np.zeros((10,1),dtype = np.float32)
    output_loss = data_sheet2.cell_value(1,73)
    output_accuracy = 0.25

    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_weights_1[i,j] = data_sheet.cell_value(i+22,j+18)
            input_weights_2[i,j] = data_sheet.cell_value(i+35,j+18)
            input_weights_3[i,j] = data_sheet.cell_value(i+48,j+18)
            output_weights_1[i,j] = data_sheet1.cell_value(i+15,j+167)
            output_weights_2[i,j] = data_sheet1.cell_value(i+33,j+167)
            output_weights_3[i,j] = data_sheet1.cell_value(i+51,j+167)


    for i in range(10):
        input_baises_1[i,0] = data_sheet.cell_value(i+22,12)
        input_baises_2[i,0] = data_sheet.cell_value(i+35,12)
        input_baises_3[i,0] = data_sheet.cell_value(i+48,12)

        output_baises_1[i,0] = data_sheet1.cell_value(11,i+167)
        output_baises_2[i,0] = data_sheet1.cell_value(29,i+167)
        output_baises_3[i,0] = data_sheet1.cell_value(47,i+167)


    for i in range(10):
        for j in range(4):
            #print(i)
            #print(j)
            input_image_array[j,i] = data_sheet.cell_value(i+6,j+2)
            input_label_array[j,i] = data_sheet.cell_value(i+6,j+63)

    result_weights_1,result_baises_1,result_weights_2,result_baises_2,result_weights_3,result_baises_3,result_loss, result_accuracy = NN_2_layers_test(input_weights_1,input_baises_1,input_weights_2, input_baises_2,input_weights_3,input_baises_3, input_image_array, input_label_array, activation_1,reg,decay,optimizer)

    result_weights_1 = np.round(result_weights_1 ,4)
    output_weights_1 = np.round(output_weights_1, 4)
    result_weights_2 = np.round(result_weights_2 ,4)
    output_weights_2 = np.round(output_weights_2, 4)
    result_weights_3 = np.round(result_weights_3 ,4)
    output_weights_3 = np.round(output_weights_3, 4)


    result_baises_1 = np.round(result_baises_1 ,4)
    output_baises_1 = np.round(output_baises_1, 4) 
    result_baises_2 = np.round(result_baises_2 ,4)
    output_baises_2 = np.round(output_baises_2, 4) 
    result_baises_3 = np.round(result_baises_3 ,4)
    output_baises_3 = np.round(output_baises_3, 4)

    output_loss = round(output_loss, 4)
    result_loss = round(result_loss, 4)
    #print(result_baises_3)
    #print(output_baises_3)
    
    #if ((result_baises_3 == output_baises_3).all()):
    #    print("Fuction tested okay")
    #if ((result_baises_3 == output_baises_3).all()) == False:
    #    print("Fuction tested not okay")
        
    if (((result_weights_1 == output_weights_1).all()) and ((result_weights_2 == output_weights_2).all()) and ((result_weights_3 == output_weights_3).all()) and ((result_baises_1 == output_baises_1).all()) and ((result_baises_2 == output_baises_2).all()) and ((result_baises_3 == output_baises_3).all()) and result_loss == output_loss and result_accuracy == output_accuracy):
        print("Fuction tested okay")
    if (((result_weights_1 == output_weights_1).all()) and ((result_weights_2 == output_weights_2).all()) and ((result_weights_3 == output_weights_3).all()) and ((result_baises_1 == output_baises_1).all()) and ((result_baises_2 == output_baises_2).all()) and ((result_baises_3 == output_baises_3).all()) and result_loss == output_loss and result_accuracy == output_accuracy) == False:
        print("Fuction tested not okay")
    return None


def layer_1_learning_rate_decay_test():
    print("Function with 1 hidden layer with Relu activation with SGD optimizer with no reggression with decaying learning rate is under testing")

    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet67')
    data_sheet1 = data.sheet_by_name('Sheet71')
    data_sheet2 = data.sheet_by_name('Sheet72')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    activation_1 = "relu"
    reg = "n"
    decay = "y"
    optimizer = "sgd"
    #print(total_rows)
    #print(total_cols)
    input_weights_1 = np.zeros((10,10),dtype = np.float32)
    input_weights_2 = np.zeros((10,10),dtype = np.float32)
    #input_weights_3 = np.zeros((10,10),dtype = np.float32)
    #output_loss = data_sheet.cell_value(1,73)

    input_baises_1 = np.zeros((10,1),dtype = np.float32)
    input_baises_2 = np.zeros((10,1),dtype = np.float32)
    #input_baises_3 = np.zeros((10,1),dtype = np.float32)

    input_image_array = np.zeros((4,10),dtype = np.float32)
    input_label_array = np.zeros((4,10),dtype = np.float32)

    output_weights_1 = np.zeros((10,10),dtype = np.float32)
    output_weights_2 = np.zeros((10,10),dtype = np.float32)
    #output_weights_3 = np.zeros((10,10),dtype = np.float32)

    output_baises_1 = np.zeros((10,1),dtype = np.float32)
    output_baises_2 = np.zeros((10,1),dtype = np.float32)
    #output_baises_3 = np.zeros((10,1),dtype = np.float32)
    output_loss = data_sheet2.cell_value(1,73)
    output_accuracy = 0.0

    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_weights_1[i,j] = data_sheet.cell_value(i+22,j+18)
            input_weights_2[i,j] = data_sheet.cell_value(i+35,j+18)
            #input_weights_3[i,j] = data_sheet.cell_value(i+48,j+18)
            output_weights_1[i,j] = data_sheet1.cell_value(i+15,j+167)
            output_weights_2[i,j] = data_sheet1.cell_value(i+33,j+167)
            #output_weights_3[i,j] = data_sheet1.cell_value(i+51,j+167)


    for i in range(10):
        input_baises_1[i,0] = data_sheet.cell_value(i+22,12)
        input_baises_2[i,0] = data_sheet.cell_value(i+35,12)
        #input_baises_3[i,0] = data_sheet.cell_value(i+48,12)

        output_baises_1[i,0] = data_sheet1.cell_value(11,i+167)
        output_baises_2[i,0] = data_sheet1.cell_value(29,i+167)
        #output_baises_3[i,0] = data_sheet1.cell_value(47,i+167)


    for i in range(10):
        for j in range(4):
            #print(i)
            #print(j)
            input_image_array[j,i] = data_sheet.cell_value(i+6,j+2)
            input_label_array[j,i] = data_sheet.cell_value(i+6,j+63)

    result_weights_1,result_baises_1,result_weights_2,result_baises_2,result_loss, result_accuracy = NN_1_layers_test(input_weights_1,input_baises_1,input_weights_2, input_baises_2, input_image_array, input_label_array, activation_1,reg,decay,optimizer)

    result_weights_1 = np.round(result_weights_1 ,4)
    output_weights_1 = np.round(output_weights_1, 4)
    result_weights_2 = np.round(result_weights_2 ,4)
    output_weights_2 = np.round(output_weights_2, 4)
    #result_weights_3 = np.round(result_weights_3 ,4)
    #output_weights_3 = np.round(output_weights_3, 4)


    result_baises_1 = np.round(result_baises_1 ,4)
    output_baises_1 = np.round(output_baises_1, 4) 
    result_baises_2 = np.round(result_baises_2 ,4)
    output_baises_2 = np.round(output_baises_2, 4) 
    #result_baises_3 = np.round(result_baises_3 ,4)
    #output_baises_3 = np.round(output_baises_3, 4)

    output_loss = round(output_loss, 4)
    result_loss = round(result_loss, 4)
    #print(result_accuracy)
    #print(result_loss)
    
    #if ((result_baises_3 == output_baises_3).all()):
    #    print("Fuction tested okay")
    #if ((result_baises_3 == output_baises_3).all()) == False:
    #    print("Fuction tested not okay")
        
    if (((result_weights_1 == output_weights_1).all()) and ((result_weights_2 == output_weights_2).all()) and ((result_baises_1 == output_baises_1).all()) and ((result_baises_2 == output_baises_2).all()) and result_loss == output_loss and result_accuracy == output_accuracy):
        print("Fuction tested okay")
    if (((result_weights_1 == output_weights_1).all()) and ((result_weights_2 == output_weights_2).all()) and  ((result_baises_1 == output_baises_1).all()) and ((result_baises_2 == output_baises_2).all()) and  result_loss == output_loss and result_accuracy == output_accuracy) == False:
        print("Fuction tested not okay")
    return None

def layer_0_learning_rate_decay_test():
    print("Function with zero hidden layer with SGD optimizer with no reggression with decaying learning rate is under testing")

    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet91')
    data_sheet1 = data.sheet_by_name('Sheet95')
    data_sheet2 = data.sheet_by_name('Sheet96')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    activation_1 = "no"
    reg = "n"
    decay = "y"
    optimizer = "sgd"
    #print(total_rows)
    #print(total_cols)
    input_weights_1 = np.zeros((10,10),dtype = np.float32)
    #input_weights_2 = np.zeros((10,10),dtype = np.float32)
    #input_weights_3 = np.zeros((10,10),dtype = np.float32)
    #output_loss = data_sheet.cell_value(1,73)

    input_baises_1 = np.zeros((10,1),dtype = np.float32)
    #input_baises_2 = np.zeros((10,1),dtype = np.float32)
    #input_baises_3 = np.zeros((10,1),dtype = np.float32)

    input_image_array = np.zeros((4,10),dtype = np.float32)
    input_label_array = np.zeros((4,10),dtype = np.float32)

    output_weights_1 = np.zeros((10,10),dtype = np.float32)
    #output_weights_2 = np.zeros((10,10),dtype = np.float32)
    #output_weights_3 = np.zeros((10,10),dtype = np.float32)

    output_baises_1 = np.zeros((10,1),dtype = np.float32)
    #output_baises_2 = np.zeros((10,1),dtype = np.float32)
    #output_baises_3 = np.zeros((10,1),dtype = np.float32)
    output_loss = data_sheet2.cell_value(1,73)
    output_accuracy = 0.0

    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_weights_1[i,j] = data_sheet.cell_value(i+22,j+18)
            #input_weights_2[i,j] = data_sheet.cell_value(i+35,j+18)
            #input_weights_3[i,j] = data_sheet.cell_value(i+48,j+18)
            output_weights_1[i,j] = data_sheet1.cell_value(i+15,j+167)
            #output_weights_2[i,j] = data_sheet1.cell_value(i+33,j+167)
            #output_weights_3[i,j] = data_sheet1.cell_value(i+51,j+167)


    for i in range(10):
        input_baises_1[i,0] = data_sheet.cell_value(i+22,12)
        #input_baises_2[i,0] = data_sheet.cell_value(i+35,12)
        #input_baises_3[i,0] = data_sheet.cell_value(i+48,12)

        output_baises_1[i,0] = data_sheet1.cell_value(11,i+167)
        #output_baises_2[i,0] = data_sheet1.cell_value(29,i+167)
        #output_baises_3[i,0] = data_sheet1.cell_value(47,i+167)


    for i in range(10):
        for j in range(4):
            #print(i)
            #print(j)
            input_image_array[j,i] = data_sheet.cell_value(i+6,j+2)
            input_label_array[j,i] = data_sheet.cell_value(i+6,j+63)

    result_weights_1,result_baises_1,result_loss, result_accuracy = NN_0_layers_test(input_weights_1,input_baises_1,input_image_array, input_label_array, activation_1,reg,decay,optimizer)

    result_weights_1 = np.round(result_weights_1 ,4)
    output_weights_1 = np.round(output_weights_1, 4)
    #result_weights_2 = np.round(result_weights_2 ,4)
    #output_weights_2 = np.round(output_weights_2, 4)
    #result_weights_3 = np.round(result_weights_3 ,4)
    #output_weights_3 = np.round(output_weights_3, 4)


    result_baises_1 = np.round(result_baises_1 ,4)
    output_baises_1 = np.round(output_baises_1, 4) 
    #result_baises_2 = np.round(result_baises_2 ,4)
    #output_baises_2 = np.round(output_baises_2, 4) 
    #result_baises_3 = np.round(result_baises_3 ,4)
    #output_baises_3 = np.round(output_baises_3, 4)

    output_loss = round(output_loss, 4)
    result_loss = round(result_loss, 4)
    #print(result_accuracy)
    #print(result_loss)
    
    #if ((result_baises_3 == output_baises_3).all()):
    #    print("Fuction tested okay")
    #if ((result_baises_3 == output_baises_3).all()) == False:
    #    print("Fuction tested not okay")
        
    if (((result_weights_1 == output_weights_1).all()) and ((result_baises_1 == output_baises_1).all()) and result_loss == output_loss and result_accuracy == output_accuracy ):
        print("Fuction tested okay")
    if (((result_weights_1 == output_weights_1).all()) and  ((result_baises_1 == output_baises_1).all()) and result_loss == output_loss and result_accuracy == output_accuracy) == False:
        print("Fuction tested not okay")
    return None

def layer_2_with_reg_test():
    print("Function with 2 hidden layer relu activation with SGD optimizer with reggresion with no learning rate decay is under testing")
    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet43')
    data_sheet1 = data.sheet_by_name('Sheet47')
    data_sheet2 = data.sheet_by_name('Sheet48')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    activation_1 = "relu"
    reg = "y"
    decay = "n"
    optimizer = "sgd"
    #print(total_rows)
    #print(total_cols)
    input_weights_1 = np.zeros((10,10),dtype = np.float32)
    input_weights_2 = np.zeros((10,10),dtype = np.float32)
    input_weights_3 = np.zeros((10,10),dtype = np.float32)
    #output_loss = data_sheet.cell_value(1,73)

    input_baises_1 = np.zeros((10,1),dtype = np.float32)
    input_baises_2 = np.zeros((10,1),dtype = np.float32)
    input_baises_3 = np.zeros((10,1),dtype = np.float32)

    input_image_array = np.zeros((4,10),dtype = np.float32)
    input_label_array = np.zeros((4,10),dtype = np.float32)

    output_weights_1 = np.zeros((10,10),dtype = np.float32)
    output_weights_2 = np.zeros((10,10),dtype = np.float32)
    output_weights_3 = np.zeros((10,10),dtype = np.float32)

    output_baises_1 = np.zeros((10,1),dtype = np.float32)
    output_baises_2 = np.zeros((10,1),dtype = np.float32)
    output_baises_3 = np.zeros((10,1),dtype = np.float32)
    output_loss = data_sheet2.cell_value(1,73)
    output_accuracy = 0.25

    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_weights_1[i,j] = data_sheet.cell_value(i+22,j+18)
            input_weights_2[i,j] = data_sheet.cell_value(i+35,j+18)
            input_weights_3[i,j] = data_sheet.cell_value(i+48,j+18)
            output_weights_1[i,j] = data_sheet1.cell_value(i+15,j+167)
            output_weights_2[i,j] = data_sheet1.cell_value(i+33,j+167)
            output_weights_3[i,j] = data_sheet1.cell_value(i+51,j+167)


    for i in range(10):
        input_baises_1[i,0] = data_sheet.cell_value(i+22,12)
        input_baises_2[i,0] = data_sheet.cell_value(i+35,12)
        input_baises_3[i,0] = data_sheet.cell_value(i+48,12)

        output_baises_1[i,0] = data_sheet1.cell_value(11,i+167)
        output_baises_2[i,0] = data_sheet1.cell_value(29,i+167)
        output_baises_3[i,0] = data_sheet1.cell_value(47,i+167)


    for i in range(10):
        for j in range(4):
            #print(i)
            #print(j)
            input_image_array[j,i] = data_sheet.cell_value(i+6,j+2)
            input_label_array[j,i] = data_sheet.cell_value(i+6,j+63)

    result_weights_1,result_baises_1,result_weights_2,result_baises_2,result_weights_3,result_baises_3,result_loss, result_accuracy = NN_2_layers_test(input_weights_1,input_baises_1,input_weights_2, input_baises_2,input_weights_3,input_baises_3, input_image_array, input_label_array, activation_1,reg,decay,optimizer)

    result_weights_1 = np.round(result_weights_1 ,4)
    output_weights_1 = np.round(output_weights_1, 4)
    result_weights_2 = np.round(result_weights_2 ,4)
    output_weights_2 = np.round(output_weights_2, 4)
    result_weights_3 = np.round(result_weights_3 ,4)
    output_weights_3 = np.round(output_weights_3, 4)


    result_baises_1 = np.round(result_baises_1 ,4)
    output_baises_1 = np.round(output_baises_1, 4) 
    result_baises_2 = np.round(result_baises_2 ,4)
    output_baises_2 = np.round(output_baises_2, 4) 
    result_baises_3 = np.round(result_baises_3 ,4)
    output_baises_3 = np.round(output_baises_3, 4)

    output_loss = round(output_loss, 4)
    result_loss = round(result_loss, 4)
    #print(result_baises_3)
    #print(output_baises_3)
    
    #if ((result_baises_3 == output_baises_3).all()):
    #    print("Fuction tested okay")
    #if ((result_baises_3 == output_baises_3).all()) == False:
    #    print("Fuction tested not okay")
        
    if (((result_weights_1 == output_weights_1).all()) and ((result_weights_2 == output_weights_2).all()) and ((result_weights_3 == output_weights_3).all()) and ((result_baises_1 == output_baises_1).all()) and ((result_baises_2 == output_baises_2).all()) and ((result_baises_3 == output_baises_3).all()) and result_loss == output_loss and result_accuracy == output_accuracy):
        print("Fuction tested okay")
    if (((result_weights_1 == output_weights_1).all()) and ((result_weights_2 == output_weights_2).all()) and ((result_weights_3 == output_weights_3).all()) and ((result_baises_1 == output_baises_1).all()) and ((result_baises_2 == output_baises_2).all()) and ((result_baises_3 == output_baises_3).all()) and result_loss == output_loss and result_accuracy == output_accuracy) == False:
        print("Fuction tested not okay")
        return None

def layer_1_with_reg_test():
    print("Function with 1 hidden layer relu activation with SGD optimizer with reggresion with no learning rate decay is under testing")

    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet73')
    data_sheet1 = data.sheet_by_name('Sheet77')
    data_sheet2 = data.sheet_by_name('Sheet78')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    activation_1 = "relu"
    reg = "y"
    decay = "n"
    optimizer = "sgd"
    #print(total_rows)
    #print(total_cols)
    input_weights_1 = np.zeros((10,10),dtype = np.float32)
    input_weights_2 = np.zeros((10,10),dtype = np.float32)
    #input_weights_3 = np.zeros((10,10),dtype = np.float32)
    #output_loss = data_sheet.cell_value(1,73)

    input_baises_1 = np.zeros((10,1),dtype = np.float32)
    input_baises_2 = np.zeros((10,1),dtype = np.float32)
    #input_baises_3 = np.zeros((10,1),dtype = np.float32)

    input_image_array = np.zeros((4,10),dtype = np.float32)
    input_label_array = np.zeros((4,10),dtype = np.float32)

    output_weights_1 = np.zeros((10,10),dtype = np.float32)
    output_weights_2 = np.zeros((10,10),dtype = np.float32)
    #output_weights_3 = np.zeros((10,10),dtype = np.float32)

    output_baises_1 = np.zeros((10,1),dtype = np.float32)
    output_baises_2 = np.zeros((10,1),dtype = np.float32)
    #output_baises_3 = np.zeros((10,1),dtype = np.float32)
    output_loss = data_sheet2.cell_value(1,73)
    output_accuracy = 0.0

    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_weights_1[i,j] = data_sheet.cell_value(i+22,j+18)
            input_weights_2[i,j] = data_sheet.cell_value(i+35,j+18)
            #input_weights_3[i,j] = data_sheet.cell_value(i+48,j+18)
            output_weights_1[i,j] = data_sheet1.cell_value(i+15,j+167)
            output_weights_2[i,j] = data_sheet1.cell_value(i+33,j+167)
            #output_weights_3[i,j] = data_sheet1.cell_value(i+51,j+167)


    for i in range(10):
        input_baises_1[i,0] = data_sheet.cell_value(i+22,12)
        input_baises_2[i,0] = data_sheet.cell_value(i+35,12)
        #input_baises_3[i,0] = data_sheet.cell_value(i+48,12)

        output_baises_1[i,0] = data_sheet1.cell_value(11,i+167)
        output_baises_2[i,0] = data_sheet1.cell_value(29,i+167)
        #output_baises_3[i,0] = data_sheet1.cell_value(47,i+167)


    for i in range(10):
        for j in range(4):
            #print(i)
            #print(j)
            input_image_array[j,i] = data_sheet.cell_value(i+6,j+2)
            input_label_array[j,i] = data_sheet.cell_value(i+6,j+63)

    result_weights_1,result_baises_1,result_weights_2,result_baises_2,result_loss, result_accuracy = NN_1_layers_test(input_weights_1,input_baises_1,input_weights_2, input_baises_2, input_image_array, input_label_array, activation_1,reg,decay,optimizer)

    result_weights_1 = np.round(result_weights_1 ,4)
    output_weights_1 = np.round(output_weights_1, 4)
    result_weights_2 = np.round(result_weights_2 ,4)
    output_weights_2 = np.round(output_weights_2, 4)
    #result_weights_3 = np.round(result_weights_3 ,4)
    #output_weights_3 = np.round(output_weights_3, 4)


    result_baises_1 = np.round(result_baises_1 ,4)
    output_baises_1 = np.round(output_baises_1, 4) 
    result_baises_2 = np.round(result_baises_2 ,4)
    output_baises_2 = np.round(output_baises_2, 4) 
    #result_baises_3 = np.round(result_baises_3 ,4)
    #output_baises_3 = np.round(output_baises_3, 4)

    output_loss = round(output_loss, 4)
    result_loss = round(result_loss, 4)
    #print(result_accuracy)
    #print(result_loss)
    
    #if ((result_baises_3 == output_baises_3).all()):
    #    print("Fuction tested okay")
    #if ((result_baises_3 == output_baises_3).all()) == False:
    #    print("Fuction tested not okay")
        
    if (((result_weights_1 == output_weights_1).all()) and ((result_weights_2 == output_weights_2).all()) and ((result_baises_1 == output_baises_1).all()) and ((result_baises_2 == output_baises_2).all()) and result_loss == output_loss and result_accuracy == output_accuracy):
        print("Fuction tested okay")
    if (((result_weights_1 == output_weights_1).all()) and ((result_weights_2 == output_weights_2).all()) and  ((result_baises_1 == output_baises_1).all()) and ((result_baises_2 == output_baises_2).all()) and  result_loss == output_loss and result_accuracy == output_accuracy) == False:
        print("Fuction tested not okay")
    return None

def layer_0_with_reg_test():

    print("Function with zero hidden layer with SGD optimizer with reggresion with no learning rate decay is under testing")

    data = xlrd.open_workbook('valid_excel.xlsx')
    data_sheet = data.sheet_by_name('Sheet97')
    data_sheet1 = data.sheet_by_name('Sheet101')
    data_sheet2 = data.sheet_by_name('Sheet102')
    #prob = 0.8
    total_rows = data_sheet.nrows
    total_cols = data_sheet.ncols
    #print(total_rows)
    #print(total_cols)
    activation_1 = "no"
    reg = "y"
    decay = "n"
    optimizer = "sgd"
    input_weights_1 = np.zeros((10,10),dtype = np.float32)
    #input_weights_2 = np.zeros((10,10),dtype = np.float32)
    #input_weights_3 = np.zeros((10,10),dtype = np.float32)
    #output_loss = data_sheet.cell_value(1,73)

    input_baises_1 = np.zeros((10,1),dtype = np.float32)
    #input_baises_2 = np.zeros((10,1),dtype = np.float32)
    #input_baises_3 = np.zeros((10,1),dtype = np.float32)

    input_image_array = np.zeros((4,10),dtype = np.float32)
    input_label_array = np.zeros((4,10),dtype = np.float32)

    output_weights_1 = np.zeros((10,10),dtype = np.float32)
    #output_weights_2 = np.zeros((10,10),dtype = np.float32)
    #output_weights_3 = np.zeros((10,10),dtype = np.float32)

    output_baises_1 = np.zeros((10,1),dtype = np.float32)
    #output_baises_2 = np.zeros((10,1),dtype = np.float32)
    #output_baises_3 = np.zeros((10,1),dtype = np.float32)
    output_loss = data_sheet2.cell_value(1,73)
    output_accuracy = 0.0

    for i in range(10):
        for j in range(10):
            #print(i)
            #print(j)
            input_weights_1[i,j] = data_sheet.cell_value(i+22,j+18)
            #input_weights_2[i,j] = data_sheet.cell_value(i+35,j+18)
            #input_weights_3[i,j] = data_sheet.cell_value(i+48,j+18)
            output_weights_1[i,j] = data_sheet1.cell_value(i+15,j+167)
            #output_weights_2[i,j] = data_sheet1.cell_value(i+33,j+167)
            #output_weights_3[i,j] = data_sheet1.cell_value(i+51,j+167)


    for i in range(10):
        input_baises_1[i,0] = data_sheet.cell_value(i+22,12)
        #input_baises_2[i,0] = data_sheet.cell_value(i+35,12)
        #input_baises_3[i,0] = data_sheet.cell_value(i+48,12)

        output_baises_1[i,0] = data_sheet1.cell_value(11,i+167)
        #output_baises_2[i,0] = data_sheet1.cell_value(29,i+167)
        #output_baises_3[i,0] = data_sheet1.cell_value(47,i+167)


    for i in range(10):
        for j in range(4):
            #print(i)
            #print(j)
            input_image_array[j,i] = data_sheet.cell_value(i+6,j+2)
            input_label_array[j,i] = data_sheet.cell_value(i+6,j+63)

    result_weights_1,result_baises_1,result_loss, result_accuracy = NN_0_layers_test(input_weights_1,input_baises_1,input_image_array, input_label_array, activation_1,reg,decay,optimizer)

    result_weights_1 = np.round(result_weights_1 ,4)
    output_weights_1 = np.round(output_weights_1, 4)
    #result_weights_2 = np.round(result_weights_2 ,4)
    #output_weights_2 = np.round(output_weights_2, 4)
    #result_weights_3 = np.round(result_weights_3 ,4)
    #output_weights_3 = np.round(output_weights_3, 4)


    result_baises_1 = np.round(result_baises_1 ,4)
    output_baises_1 = np.round(output_baises_1, 4) 
    #result_baises_2 = np.round(result_baises_2 ,4)
    #output_baises_2 = np.round(output_baises_2, 4) 
    #result_baises_3 = np.round(result_baises_3 ,4)
    #output_baises_3 = np.round(output_baises_3, 4)

    output_loss = round(output_loss, 4)
    result_loss = round(result_loss, 4)
    #print(result_accuracy)
    #print(result_loss)
    
    #if ((result_baises_3 == output_baises_3).all()):
    #    print("Fuction tested okay")
    #if ((result_baises_3 == output_baises_3).all()) == False:
    #    print("Fuction tested not okay")
        
    if (((result_weights_1 == output_weights_1).all()) and ((result_baises_1 == output_baises_1).all()) and result_loss == output_loss and result_accuracy == output_accuracy ):
        print("Fuction tested okay")
    if (((result_weights_1 == output_weights_1).all()) and  ((result_baises_1 == output_baises_1).all()) and result_loss == output_loss and result_accuracy == output_accuracy) == False:
        print("Fuction tested not okay")
    return None
if __name__=='__main__':
    repeat =0
    while(repeat==0):
        digit = input("Enter the digit associated with the test function ")
        #digit = int(digit)
        if digit=="1":
            relu_activation_test()
            repeat =1
        if digit=="2":
            relu_activation_back_test()
            repeat =1
        if digit=="3":
            sigmoid_activation_test()
            repeat =1
        if digit=="4":
            sigmoid_activation_back_test()
            repeat =1
        if digit=="5":
            SGD_optimizer_test_for_2_layers()
            repeat =1
        if digit=="6":
            SGD_optimizer_test_for_1_layers()
            repeat =1
        if digit=="7":
            SGD_optimizer_test_for_0_layers()
            repeat =1
        if digit=="8":
            Momentum_optimizer_test_for_2_layers()
            repeat =1
        if digit=="9":
            Momentum_optimizer_test_for_1_layers()
            repeat =1
        if digit=="10":
            Momentum_optimizer_test_for_0_layers()
            repeat =1
        if digit=="11":
            reg_loss_test_2_layers()
            repeat =1
        if digit=="12":
            reg_loss_test_1_layers()
            repeat =1
        if digit=="13":
            reg_loss_test_0_layers()
            repeat =1
        if digit=="14":
            learning_rate_decay_test()
            repeat =1
        if digit=="15":
            Normal_normalization_test()
            repeat =1
        if digit=="16":
            Simple_normalization_test()
            repeat =1
        if digit=="17":
            Gaussian_initialization_test()
            repeat =1
        if digit=="18":
            Xavier_initialization_test()
            repeat =1
        if digit=="19":
            Dropout_forward_test()
            repeat =1
        if digit=="20":
            Loss_function_with_reg_2_layer_test()
            repeat =1
        if digit=="21":
            Loss_function_with_reg_1_layer_test()
            repeat =1
        if digit=="22":
            Loss_function_with_reg_0_layer_test()
            repeat =1
        if digit=="23":
            Loss_function_with_no_reg_test()
            repeat =1
        if digit=="24":
            Forward_prop_for_loss_with_relu_test()
            repeat =1
        if digit=="25":
            Forward_prop_for_loss_no_reg_with_relu_test()
            repeat =1
        if digit=="26":
            Forward_prop_for_loss_with_sigmoid_test()
            repeat =1
        if digit=="27":
            Forward_prop_for_loss_no_reg_with_sigmoid_test()
            repeat =1
        if digit=="28":
            Forward_prop_for_loss_with_reg_with_relu_layer_1_test()
            repeat =1
        if digit=="29":
            Forward_prop_for_loss_with_no_reg_with_relu_layer_1_test()
            repeat =1
        if digit=="30":
            Forward_prop_for_loss_with_reg_with_sigmoid_layer_1_test()
            repeat =1
        if digit=="31":
            Forward_prop_for_loss_with_no_reg_with_sigmoid_layer_1_test()
            repeat =1
        if digit=="32":
            Forward_prop_for_loss_with_reg_layer_0_test()
            repeat =1
        if digit=="33":
            Forward_prop_for_loss_with_no_reg_layer_0_test()
            repeat =1
        if digit=="34":
            Accuracy_with_relu_test()
            repeat =1
        if digit=="35":
            Accuracy_with_sigmoid_test()
            repeat =1
        if digit=="36":
            Accuracy_with_relu_layer_1_test()
            repeat =1
        if digit=="37":
            Accuracy_with_sigmoid_layer_1_test()
            repeat =1
        if digit=="38":
            Accuracy_with_layer_0_test()
            repeat =1
        if digit=="39":
            Data_pre_processing_test()
            repeat =1
        if digit=="40":
            layer_2_relu_test()
            repeat =1
        if digit=="41":
            layer_1_relu_test()
            repeat =1
        if digit=="42":
            layer_0_test()
            repeat =1
        if digit=="43":
            layer_2_sigmoid_test()
            repeat =1
        if digit=="44":
            layer_1_sigmoid_test()
            repeat =1
        if digit=="45":
            layer_2_momentum_opt_test()
            repeat =1
        if digit=="46":
            layer_1_momentum_opt_test()
            repeat =1
        if digit=="47":
            layer_0_momentum_opt_test()
            repeat =1
        if digit=="48":
            layer_2_with_learning_rate_decay_test()
            repeat =1
        if digit=="49":
            layer_1_learning_rate_decay_test()
            repeat =1
        if digit=="50":
            layer_0_learning_rate_decay_test()
            repeat =1
        if digit=="51":
            layer_2_with_reg_test()
            repeat =1
        if digit=="52":
            layer_1_with_reg_test()
            repeat =1
        if digit=="53":
            layer_0_with_reg_test()
            repeat =1
        if (int(digit)>53 or int(digit)<1):
            print("Please enter the correct input ")
        if (digit.isdigit())==False:
            print("Please enter the correct input ")
