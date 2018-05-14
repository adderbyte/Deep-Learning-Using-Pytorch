import torch
#from torch.autograd import Variable
#from torch import nn
#from torch.nn import functional as F
import numpy as np
import math
import dlc_practical_prologue as prologue

class linear():
	'''
	Put Comments here !!!

	'''

	def __init__(self,start_layer_size,end_layer_size):
        assert type(start_layer_size)== int 
        assert type(end_layer_size) == int
        if (start_layer_size > 0 and end_layer_size > 0):
            
            self.m = start_layer_size
            self.n = end_layer_size
            self.w = torch.Tensor(self.m,self.n).normal_()
            self.b = torch.Tensor(self.m,1).normal_()
        else :
            raise "error"
        
    def forward(self, x):
    	'''

        Put comments here
    	'''
       
        return torch.mm(self.w , x)+self.b
    
    def backward(self,x):
        print("backward_linear called")
        return x,self.w
    
class Relu():
	'''

    Put comments Here

	'''
     
    def forward(self,x):
        y = torch.zeros_like(x)
        return torch.max(x,y)
    
    def backward(self,x):
        u=x.clone()
        u[u>0]=1
        u[u<0]=0
        return u



class Identity():
	'''

   Put the comments here!!!

	'''
    
    def forward(self,x):
        return x
    
    def backward(self,x):
        return torch.ones_like(x)


 def __structurize__(activations):
        que=[]
        l=len(activations)
        print(l)
        for i in range(l):
            que.append(activations[i])
            if(i<l-1):
                if(type(activations[i]).__name__=='linear' and type(activations[i+1]).__name__=='linear'):
                    que.append(Identity())
        
        if (type(activations[-1]).__name__=='linear'):
            que.append(Identity())
        
        return que


class sequential(): #list of operation
	'''
     Put Comments Here
   
	'''
    def __init__(self,*operations):
        self.l = len(operations)
        self.result = []
        self.initial_operations = operations
        self.operations = __structurize__(self.initial_operations )
#         linear.__init__(self,in_node,out_node)
#         function.__init__(self)
        self.delta =[] # dl_ds
        self.forward_flag=False
#         self.w =[]
#         self.w.append(0)
#         self.w.append(weight)
#         self.b=bias
#         self.l = int(num_layers)
#         self.s = list(np.zeros(self.l + 1) ) # first element remain 0  each time
#         self.x = list(np.zeros(self.l + 1) )
#         #derivatives
#         self.dl_dx = list(np.zeros(self.l+1) ) # first element remain 0  each time
          
        self.dl_dw = []
        self.dl_db = [] 
        
    '''
    def structrize(self,activations):
        que=[]
        l=len(activations)
        print(l)
        for i in range(l):
            que.append(activations[i])
            if(i<l-1):
                if(type(activations[i]).__name__=='linear' and type(activations[i+1]).__name__=='linear'):
                    que.append(Identity())
        
        if (type(activations[-1]).__name__=='linear'):
            que.append(Identity())
        
        return que

    self.operations= structrize(self.initial_operations)
    '''
    def sigma(self , inp):
        return torch.tanh(inp)
    def dsigma(x):
        
        return 1 - torch.mul(torch.tanh(x),torch.tanh(x))
    
    def loss(self ,output,target):
        return torch.sum(torch.pow(output-target,2))
    
    
    def dloss(output,target): # dloss is derivative of loss
    #print("dloss_shape(v-t) = ",v.shape , "-",t.shape)
        return 2*(output-target)
    
    def forward(self,x):
        self.forward_flag=True
        self.result.append(x)
        for op in self.operations:
            self.result.append(op.forward(self.result[-1]))
            #print('sequential forward called ,result[i]:',self.result[-1])
        return self.result   
        
    def backward_pass(self,target):
        
        if(self.forward_flag==False):
            raise ValueError("forward hasn't been called.")
        self.delta.append(-2*(target-self.result[-1]))
        
        print("result",len(self.result))
        
        
        for i in range (self.l,0,-1): # calculating deltas
            #print("i = ",i)
            #print("delta length",len(self.delta))
            
            #print('delta = ',self.delta[-1])
            #print('current op= ',self.operations[i-1])
            self.delta.append(torch.mm(self.delta[-1],self.operations[i-1].backward(self.result[i-1])[1])) # [1] is to select w not x from linear_backward()
       
        self.delta.reverse()
        for i in range (self.l,0,-1): #calculating dl_dw's 
            print("i = ",i,"-------------------------------------------------------------------------")
            print("x_(l-1).shape" , self.result[i-1].shape)
            print("delta[",i,"]",self.delta[i].shape)
            #self.dl_dw.append(torch.mm(self.delta[i],torch.t(self.result[i-1])))
            self.dl_dw.append(torch.mm(self.result[i-1],self.delta[i]))
        
        self.dl_dw.reverse()
    
        self.dl_db=self.delta
        return self.delta, self.dl_dw, self.dl_db
            
            
#         self.dl_dx[self.l] = dloss(self.x[self.l] , target)
#         self.dl_ds[self.l] = torch.mul(dl_dx[self.l],dsigma(s[self.l]) )
        
#         torch.mul( dl_dx2 , dsigma(s2) )
        
#         for i in range(self.l-1,0,-1):  # it goes from l-1 to 1
        
#             #print("dl_dx2-->size = " , (dloss(x2,t)).shape)
#             #print('x2--->size' ,  x2.shape )
#             #print('x2 = ' ,x2)
#             dl_dx[i] = dloss(x[-1],t)
    

    
#             #print("dl_dx1-->size = " , (w2.t()).shape,' * ',dl_ds2.shape)
#             dl_dx[i] = torch.mm(w[i+1].t(),dl_ds[i+1])
    
#             #print("dl_ds1-->size = " , dl_dx1.shape,' .* ',dsigma(s1).shape)
#             dl_ds[i] = torch.mul(dl_dx[i] ,dsigma(s[i]) )
    
#             #print("dl_dw1-->size = " , dl_ds1.view(-1,1).shape,' * ',(x0.view(1,-1).shape))
                #self.result.append(op.forward(self.result[-1]))
            
    
    
    
    
#             dl_dw[i] = torch.mm(dl_ds1.view(-1,1),x0.view(1,-1))
#             dl_dw2 = dl_ds2.view(-1,1).mm(x1.view(1,-1))
    
    
#             dl_db1=dl_ds1;
#             dl_db2=dl_ds2;
    
#             #print(dl_dw1.shape,dl_dw2.shape,dl_db1.shape,dl_db2.shape)
#             #print("(dl_dw1,dl_dw2,dl_db1,dl_db2)")
#         return self.dl_dw,self.dl_db