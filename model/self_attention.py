# Muti-head Attention 机制的实现
from math import sqrt
import torch
import torch.nn as nn


class Self_Attention(nn.Module):
    # input : batch_size * seq_len * input_dim
    # q : batch_size * input_dim * dim_k
    # k : batch_size * input_dim * dim_k
    # v : batch_size * input_dim * dim_v
    def __init__(self,input_dim,dim_k,dim_v):
        super(Self_Attention,self).__init__()
        self.q = nn.Linear(input_dim,dim_k)
        self.k = nn.Linear(input_dim,dim_k)
        self.v = nn.Linear(input_dim,dim_v)
        self._norm_fact = 1 / sqrt(dim_k)
        
    
    def forward(self,x):
        Q = self.q(x) # Q: batch_size * seq_len * dim_k
        K = self.k(x) # K: batch_size * seq_len * dim_k
        V = self.v(x) # V: batch_size * seq_len * dim_v
        print(Q.shape)
        print(K.shape)
        print(V.shape)
        atten = nn.Softmax(dim=-1)(torch.bmm(Q,K.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len
        print(atten.shape)
        output = torch.bmm(atten,V) # Q * K.T() * V # batch_size * seq_len * dim_v
        print(output.shape)
        return output

if __name__=='__main__':
    x=torch.randn([3,224,224])
    x=Self_Attention(3,3,64)(x)
    x=Self_Attention(64,3,128)(x)
    x=Self_Attention(128,3,256)(x)
    x=Self_Attention(256,3,512)(x)
    x=Self_Attention(512,3,1024)(x)