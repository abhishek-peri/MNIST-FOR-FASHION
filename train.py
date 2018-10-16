
# coding: utf-8

# In[ ]:


import pandas as pd
import math
import numpy as np
from operator import add
#b = np.ones((55000,1))


# In[ ]:


class makelayer:
      def __init__(self, name,size):
            self.name = name
            self.size = size

# class weights:
#     def __init__(self, name):
#         self.name =name
        
#     def initialize(self,a,b):
#         return np.random.random((a,b))
                   
class NN:
    def loaddata(self,path):
        df1 = pd.read_csv(path,header=0)
        M = df1.as_matrix()
        #Y =Y[:,785]
#         X = df1.as_matrix()
        #M = M[:,1:786].astype(float)
        return M
    
    
    def makeNN(self,layers):
        self.layers =layers
        instancenames = []
        #size =30
        for i in range(0,layers+2):
            instancenames.append(i)
        layer = {name: makelayer(name=name,size=50) for name in instancenames}    
        #w = {name: weights(name=name) for name in instancenames}
        e = []
        f = []
        layer[0].size = 784
        layer[layers+1].size = 10
        for i in range(1,layers+2):
            a =layer[i].size 
            c= layer[i-1].size
            e.append(0.001*np.random.randn(c,a))
            f.append(np.zeros((a,1)))
        return e,f
    
    def sigmoid(self,Q):
        M = []
        for i in Q:
            try:
                res = 1.0 /(1.0 + math.exp(-i))
            except OverflowError:
                res = 0.0
            M.append(res)
        return np.transpose(np.asmatrix(M))
    
    def softmax(self,U):
        return np.exp(U)/float(np.sum(np.exp(U)))
    
 
    def feedforward(self,K,W,B):
        #print(K)
        instancenames = []
        for i in range(0,layers+2):
            instancenames.append(i)
        layer = {name: makelayer(name=name,size=100) for name in instancenames}   
        H = []
        A = []
        layer[0].size = 784
        layer[layers+1].size = 10
        for i in range(1,layers+2):
            a =layer[i].size 
            c= layer[i-1].size
            H.append(np.zeros((c,1)))
            A.append(np.zeros((a,1)))
        H.append(np.zeros((10,1)))
        H[0] = np.transpose(np.asmatrix(K))
        for i in range(0,layers):
            A[i] = B[i] + np.matmul(np.transpose(W[i]),H[i])
            H[i+1] = self.sigmoid((A[i]))
        A[layers] = B[layers] + np.matmul(np.transpose(W[layers]),H[layers])
        Y1 = self.softmax((A[layers])) 
        return Y1,H,A
            
    def feature_normalize(self,R):
        mean = np.mean(R)
        range_val = np.amax(R)-np.amin(R)
        R = (R-mean)/float(np.sqrt(np.var(R)))
        return R
    
    def preprocessing(self,Y):
        #G = self.feature_normalize(S)
        Y2 = (np.arange(0,10)==Y).astype(float)      
        return np.transpose(np.asmatrix(Y2))    
    
    #def tanh(self,z):
     #   np.tanh(z)
        
    def sig(self,z):
        try:
            res = 1 / float(1 + math.exp(-z))
        except OverflowError:
            res = 0.0
        return res

    def grad_sig(self,S):
        L = []
        for i in S:
            L.append(self.sig(i)*(1 - self.sig(i)))
        return np.transpose(np.asmatrix(L))    
    
    def makegr(self,layers):
        self.layers =layers
        instancenames = []
        #size =30
        for i in range(0,layers+2):
            instancenames.append(i)
        layer = {name: makelayer(name=name,size=50) for name in instancenames}    
        #w = {name: weights(name=name) for name in instancenames}
        e = []
        f = []
        layer[0].size = 784
        layer[layers+1].size = 10
        for i in range(1,layers+2):
            a =layer[i].size 
            c= layer[i-1].size
            e.append(np.zeros((c,a)))
            f.append(np.zeros((a,1)))
        return e,f
    
    def back_prop(self,Y3,Y4,H,A,W,B):
        da = [0]*(layers+1)
        db = [0]*(layers+1)
        dw = [0]*(layers+1)
        dh = [0]*(layers+1)
        da[layers] = (Y3-Y4)
        for i in range(layers,-1,-1):
            dw[i] = np.transpose(np.matmul(da[i],np.transpose(H[i])))
            db[i] = da[i]
            dh[i] = np.matmul(W[i],da[i])
            if(i!=0): 
                da[i-1] = np.multiply(dh[i],self.grad_sig(A[i-1]))
        return dw,db
    
    def accuracy(self,U,T):
        acc = np.sum(U == T)/float(len(U))
        return acc
        
    def list_mul(self, scalar,C):
        l =[]
        for i in range(len(C)):
            l.append(np.multiply(scalar,C[i]))
        return l
        
    def list_square(self, C):
        l =[]
        for i in range(len(C)):
            l.append(np.square(C[i]))
            #C[i] = [np.square(x) for x in C[i]]
        return l
        
    def list_sqrt(self,C):
        l =[]
        for i in range(len(C)):
            l.append(np.sqrt(np.add(C[i],1e-08)))
        return l    
    
    def list_div(self,O1,O2):
        new_list = []
        for i in range(len(O1)):
            new_list.append(np.divide((O1[i]),(O2[i])))
        return new_list
    
    def loss(self,i):
        return (-math.log(i))
    
    def optimization(self,eta,epochs,batch_size):
        DT = self.loaddata(path)
        tr_loss = []
        for i in range(0,785):
            DT[:,i] = self.feature_normalize(DT[:,i])
        W,B = self.makeNN(layers)
        beta1 = 0.9 
        beta2 = 0.999
        m_w,m_b = self.makegr(layers)
        v_w,v_b = self.makegr(layers)
        for i in range(epochs):
            np.random.shuffle(DT)
            X = DT[:,1:785].astype(float)
            Y = DT[:,785].astype(float)
            batches = (55000/(batch_size))+1
            train_loss =0
            steps = 0
            errors = 0
            t=0
            for o in range(batches-1):
                db = [0]*(layers+1)
                dw = [0]*(layers+1)
                for j in range(o*batch_size,(o+1)*batch_size):
                    Y2 = self.preprocessing(Y[j])
                    Y1,I,J  = self.feedforward(X[j,:],W,B)
                    #print(np.argmax(Y1))
                    steps = steps+1
                    if(np.argmax(Y1) != np.argmax(Y2)):
                        errors = errors+1
                        err_per = 100*errors/float(steps)
                    train_loss = train_loss+(self.loss(Y1[np.argmax(Y2)]))
                    #print("epoch {},steps {},error {},loss {},lr {}\n".format(i,steps,err_per,train_loss,eta))
                    if(steps % 100 == 0):
                        print("epoch {},steps {},error {},loss {},lr {}\n".format(i,steps,err_per,train_loss,eta))
                        with open("log_train_3.txt", "a") as myfile:
                            myfile.write("epoch {},steps {},error {},loss {},lr {}\n".format(i,steps,err_per,train_loss,eta))
#ed
                    dx,dy= self.back_prop(Y1,Y2,I,J,W,B)
                    #dw = [sum(x) for x in zip(dw,dx)]
                    #db = [sum(x) for x in zip(db,dy)]
                    dw = map(add,dw,dx)
                    #print(type(dw[0]))
                    db = map(add,db,dy)                 
                m_w = map(add,self.list_mul(beta1,m_w),self.list_mul((1-beta1),dw)) 
                #print(type(m_w[0]))
                #print(len(m_w))
                m_b = map(add,self.list_mul(beta1,m_b),self.list_mul((1-beta1),db))
                v_w = map(add,self.list_mul(beta2,v_w),self.list_mul((1-beta2),self.list_square(dw)))
                v_b = map(add,self.list_mul(beta2,v_b),self.list_mul((1-beta2),self.list_square(db)))
                m_tw = self.list_mul((1.0/(1-math.pow(beta1,t+1))),m_w) 
                m_tb = self.list_mul((1.0/(1-math.pow(beta1,t+1))),m_b)
                v_tw = self.list_mul((1.0/(1-math.pow(beta2,t+1))),v_w)
                v_tb = self.list_mul((1.0/(1-math.pow(beta2,t+1))),v_b)
                v_tw = self.list_sqrt(v_tw)
                v_tb = self.list_sqrt(v_tb)
                dw = self.list_div(m_tw,v_tw)
                db = self.list_div(m_tb,v_tb)
                for k in range(0,layers+1):
                    W[k] = W[k] - eta*dw[k]
                    B[k] = B[k] - eta*db[k]           
                t =t+1
            dw,db = self.makegr(layers)
            for m in range(batches*batch_size,len(X)+1):
                Y2 = self.preprocessing(Y[m])
                Y1,I,J   = self.feedforward(X[j,:],W,B)
                dl,dk= self.back_prop(Y1,Y2,I,J,W,B)
                #print(np.argmax(Y1))
                steps = steps+1
                if(np.argmax(Y1) != np.argmax(Y2)):
                    errors = errors+1
                    err_per = 100*errors/float(steps)
                train_loss = train_loss+(self.loss(Y1[np.argmax(Y2)]))
                #print("epoch {},steps {},error {},loss {},lr {}\n".format(i,steps,err_per,train_loss,eta))
                if(steps % 100 == 0):
                    print("epoch {},steps {},error {%.2f},loss {},lr {}\n".format(i,steps,err_per,train_loss,eta))
                    with open("log_train_3.txt", "a") as myfile:
                        myfile.write("epoch {},steps {},error {},loss {},lr {}\n".format(i,steps,err_per,train_loss,eta))
                dw = map(add,dw,dl)
                db = map(add,db,dk)
            m_w = map(add,self.list_mul(beta1,m_w),self.list_mul((1-beta1),dw)) 
            m_b = map(add,self.list_mul(beta1,m_b),self.list_mul((1-beta1),db))
            v_w = map(add,self.list_mul(beta2,v_w),self.list_mul((1-beta2),self.list_square(dw)))
            v_b = map(add,self.list_mul(beta2,v_b),self.list_mul((1-beta2),self.list_square(db)))
            m_w_cap = self.list_mul((1.0/(1-math.pow(beta1,t+1))),m_w) 
            m_b_cap = self.list_mul((1.0/(1-math.pow(beta1,t+1))),m_b)
            v_w_cap = self.list_mul((1.0/(1-math.pow(beta2,t+1))),v_w)
            v_b_cap = self.list_mul((1.0/(1-math.pow(beta2,t+1))),v_b)
            v_w_sq = self.list_sqrt(v_w_cap)
            v_b_sq = self.list_sqrt(v_b_cap)
            dw = self.list_div(m_w_cap,v_w_sq)
            db = self.list_div(m_b_cap,v_b_sq) 
            for k in range(0,layers+1):
                W[k] = W[k] - eta*dw[k]
                B[k] = B[k] - eta*db[k]
            tr_loss.append(train_loss)
            print("epoch {}".format(i+1))
        return W,B   

# class deep_model:
#     def __init__(self,W_tensor,bias,train_err,val_err,describe):
#         self.W_tensor = W_tensor
#         self.bias = bias
#         self.train_err = train_err
#         self.val_err = val_err
#         self.num_hidden = num_hidden
#         self.common_length = common_length
#         self.describe = describe
#     def Dump():
#         D1 = deep_model(W_tensor,bias,train_err,val_err,'Cross-entropy,soft-max,sigmoid,ADAM')
#         filename = save_dir +'/'+'D1.sav'
#         pickle.dump(D1, open(filename, 'wb'))

#     def Load:
#         save_dir = '/home/at/Documents/DL Assgns/Prog_Assignment1/PA_1'
#         filename = save_dir +'/' + 'D1.sav'
#         loaded_model = pickle.load(open(filename, 'rb'))


# In[ ]:


path = "C:/Users/abhi123/Downloads/test.csv"
lays = NN()
layers = 1
#lays.optimization(eta=0.1,epochs=20,batch_size=200)


# In[ ]:


df1 = pd.read_csv(path,header=0)
M = df1.as_matrix()
M =M[:,0]
N = df1.as_matrix()
N = N[:,1:785].astype(float)
M


# In[ ]:


W,B = lays.optimization(eta=0.001,epochs=10,batch_size=20)


# In[ ]:


for i in range(0,784):
    N[:,i] = lays.feature_normalize(N[:,i])
#l,m,n = lays.feedforward(N[1001,:],W,B)
#np.argmax(l)


# In[ ]:


Q = N[:,11]
#Q = np.transpose(np.asmatrix(Q))#l,m,n = lays.feedforward(Q,W,B)
Q.shape
#np.transpose(np.matmul(Q.T,np.zeros((784,30))))
#E = (((np.matmul(np.transpose(W[0]),np.transpose(np.asmatrix(Q))))))
#F = lays.sigmoid(B[1]+((np.matmul(np.transpose(W[1]),(np.asmatrix(E))))))
#G = lays.sigmoid(B[2]+((np.matmul(np.transpose(W[2]),(np.asmatrix(F))))))
np.sqrt(np.var(Q))
Q = (Q-np.amin(Q))/float(np.sqrt(np.var(Q)))
Q
#(lays.feature_normalize((np.matmul(np.transpose(W[1]),(np.asmatrix(E))))))


# In[ ]:


with open("test.txt", "a") as myfile:
    myfile.write("appended text\n")


# In[ ]:


for i in range(10,-1,-1):
    print(1-np.square(np.tanh(i)))


# In[ ]:


pred = []
for i in range(len(N)):
    P,Q,R = lays.feedforward(N[i,:],W,B)
    pred.append(np.argmax(P))
accur = lays.accuracy(M,np.asmatrix(pred))    
print(accur)


# In[ ]:


pred = []
for i in range(len(N)):
    P,Q,R = lays.feedforward(N[i,:],W,B)
    pred.append(np.argmax(P))


# In[ ]:


import csv
#Assuming res is a flat list
csvfile = "C:/Users/abhi123/Desktop/submission1.csv"
with open(csvfile, "a") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in M:
        writer.writerow([val]) 
with open(csvfile, "a") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in pred:
        writer.writerow([val])        


# In[ ]:


import pickle

with open('biases_1_100', 'wb') as fp:
    pickle.dump(B, fp)


# In[ ]:


import pickle
with open('weights_1.0_100') as fp:
    W1 = pickle.load(W, fp)
# W1 = pickle.load(fp)
# B1 = pickle.load(fps)

