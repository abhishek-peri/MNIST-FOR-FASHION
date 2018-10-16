# Libraries used:
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

############################
#Programming Assignment 1
# Asit Tarsode; ME15B087
###########################
# Notes:
# Throughout the code, the vectors are horizontal by default
###############################################  Parameters  ###############################################

# No of inputs in base layer
base_num = 784

# No of hidden layers
num_hidden = 3

# No of elements in the final layer
k_num = 10

# size is the array that has the length of hidden layers.
# If the length for each hidden layer is same, size can be set directly by common_length parameter
a = np.zeros(num_hidden)
common_length = 6
size = [common_length for i in a]

if len(size) != num_hidden:
    raise SystemExit('Error!Please check the compatibility of size with num_hidden')

# Sigmoid parameters:
sigmoid_a = 1
sigmoid_b = 0

# Different file paths:
train_path = '/home/at/Documents/DL Assgns/Prog_Assignment1/PA_1/train.csv'
test_path = '/home/at/Documents/DL Assgns/Prog_Assignment1/PA_1/test.csv'
validation_path = '/home/at/Documents/DL Assgns/Prog_Assignment1/PA_1/val.csv'

######################################################################################

# H includes all the h_i's in a multidimensional array. A holds all the a_i's in a similar multidimensional array.
# The ith element of both the arrays holds the respective values for a_i and h_i
# H[0] is equal to the list of 784 inputs. A[0] is an extra row.
# y is the list of actual outputs while y_cap is the list of predicted outputs
# H[l] is nothing but y_cap
# W_tensor has all the weights. W[0] is a NaN element. W[i] corresponds to the weights' array between a_i and h_i-1
# bias contains all the respective biases
# All first level elements of H,A,W_tensor,bias are nd-arrays for ease of calculations later

L = num_hidden + 1
size_bar = [base_num] + size
print(size_bar)
A = H = [[0.01 for x in range(size_bar[i])] for i in range(L)]
y = [[int(i+0.01) for i in np.zeros(k_num)]]
a_L = [[int(i+0.01) for i in np.zeros(k_num)]]
y_cap = [[int(i+0.01) for i in np.zeros(k_num)]]

H = H + y_cap
A  = A + a_L

H = ([np.matrix(i) for i in H])
A = ([np.matrix(i) for i in A])

W_tensor = [[[np.nan]]]+ [[[0 for i in range(len(H[k-1][0]))]for j in range(0,len(A[k][0]))] for k in range(1,L+1)]
for i  in range(len(W_tensor)):
    print(W_tensor[i])
print('Length is: ',(A[1][0]))
W_tensor = [np.matrix(i) for i in W_tensor]

bias = [[np.nan]] + [[np.zeros(len(A[i]))] for i in range(1, L+1)]
bias = [np.matrix(i) for i in bias]

W_tensor_update = [[[np.nan]]]+ [[[0 for i in range(len(H[k-1]))]for j in range(0,len(A[k]))] for k in range(1,L+1)]
W_tensor_update = [np.matrix(i) for i in W_tensor]

bias_update = [[np.nan]] + [[np.zeros(len(A[i]))] for i in range(1, L+1)]
bias_update = [np.matrix(i) for i in bias]

#############################################################################################################
print('A is:',A,'\n')
#print('H is:',H,'\n')
print('W_tensor is:',W_tensor,'\n')
print(bias)

###############################################################################################################
# Read in the data
# train_values represent the feature values while train_label contains the true labels. Similarly for validation

train_values = pd.read_csv(train_path,index_col= 'id')
train_label = train_values['label']
train_values.drop(labels = ['label'],axis= 1,inplace= True)

vald_values = pd.read_csv(validation_path,index_col= 'id')
vald_label = vald_values['label']
vald_values.drop(labels = ['label'],axis= 1,inplace= True)

test_values = pd.read_csv(test_path,index_col= 'id')

if (base_num != len(train_values.iloc[0])):
    raise SystemExit('Error! Please check the compatibility of input data with base_num')

###############################################################################################################
# Defining the numerical functions necessary

def sigmoid(x):
    return (1/(1+ np.power(np.e,-1*(sigmoid_a*x+sigmoid_b))))

def softmax(X):
    X = np.matrix(X)
    a_1 = np.power(np.e, X)
    sum = np.sum(a_1)
    a_1 = a_1/float(sum)
    return a_1

def cross_entropy_loss(y):
    return (-1*np.log(y))

###############################################################################################################

def feedforward(values):
    global H
    global A
    global W_tensor
    global bias

    H[0] = np.matrix(list(values))
    for i in range(1,L):
        A[i] = np.matmul(W_tensor[i],np.transpose(np.matrix(H[(i-1)]))) + bias[i]
        print('A[i]',A[i])
        H[i] = sigmoid(A[i])
    A[L] = np.matmul(W_tensor[L],np.transpose(H[(L-1)])) + np.transpose(bias[L])
    H[L] = softmax(A[L])

#############################################################################################################
# Defining the functions that generate derivative matrices(i.e. Jacobian and gradients) required for backprop

# d_top is the highest level k*1 matrix. Differentiates loss with y's
def d_top_cross_entropy(true_label):
    global H
    a_1 = np.matrix([0 for i in range(k_num)])
    a_1[true_label] = -1/float(H[L][0])
    return np.asmatrix(a_1)

#d_2 is the k*k Jacobian matrix just after d_1. Differentiates H[l] wrt A[l]
def d_2_softmax():
    global H
    print(H[L])
    b_1 = np.matrix([[(float(H[L][i])-float(H[L][i])**2) if(i==j) else (-1*float(H[L][i])*float(H[L][j])) for i in range(k_num)] for j in range(k_num)])
    #b_1 = np.array([[1 for i in range(k_num)]for j in range(k_num)])
    print('b_1',b_1)
    return b_1

# d_h_a matrix is for derivatives between h_i and a_i
def d_hi_ai_sigmoid(i):
    global A
    global H
    if i not in range(1,len(size_bar)):
        raise SystemError('Check the hidden layer being passed. Cannot do hi_ai derivative for i = ',i)
    b_1 = np.array([[0 if(p!=j) else (sigmoid_a*H[i][j]*(1-H[i][j]))] for j in range(size_bar[i]) for p in range(size_bar[i])])

def d_ai_h_lower(i):
    global W_tensor
    if i not in range(1, len(size_bar)):
        raise SystemError('Check the hidden layer being passed. Cannot do ai_h_lower derivative for i = ',i)
    return W_tensor[i]

# The following function gets gradient for Wi and basis i with gradients for basis as the last coloumn
def contribute_grad(i,d):
    global A
    global H
    global W_tensor_update
    global bias_update

    if i not in range(1, len(size_bar)):
        raise SystemError('Check the hidden layer being passed. Cannot do contribute_grad for i = ',i)

    if len(d)!= size_bar[i]:
        raise SystemError('Check the d being passed in backprop. Cannot do contribute_grad for i = ',i)

    b_1 = np.array([[0 for p in range(size_bar[i-1] )] for q in range(size_bar[i])])
    b_2 = np.array([0 for i in range(size_bar[i])])
    for p in range(size_bar[i]):
        for q in range(size_bar[i-1]):
            b_1[p][q] = d[p]*H[i-1][q]
        b_2[p] = H[i-1][q]

    W_tensor_update[i] = W_tensor_update[i] + b_1
    bias_update[i] = bias_update[i] + b_2
#########################################################
#########################################################



def backprop(true_value):

    d_1_T = d_top_cross_entropy(true_value)
    d_2 = d_2_softmax()
    print('Shape of d_1 transpose iS: ',d_1_T.shape,'d_2 shape is: ',d_2.shape)
    d = np.matmul(d_1_T, d_2)
    contribute_grad(L, d)

    for i in range(L-1,0,-1):
        d = np.matmul(d,d_ai_h_lower(i+1))
        d = np.matmul(d, d_hi_ai_sigmoid)
        contribute_grad(i,d)

#print(H)
#print('Hi',W_tensor)
#print('initial',H)
#feedforward(train_values.iloc[0])
#print('final',H)
#backprop(1)
#print(W_tensor_update)
#print(bias_update)


