#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


def cost(x,y,m):
    total_cost=0
    N=379
    for i in range(N):
        total_cost+=(1/N)*((x[i]*m).sum()-y[i])**2
    return total_cost


# In[9]:


def step_gardient(x,y,learning_rate,m):
    M=x.shape[1]
    N=x.shape[0]
    m_slope=np.zeros(M)
    for i in range(N):
        for j in range(M):
            m_slope[j]+=(1/N)*((x[i]*m).sum()-y[i])*x[i][j]
    m=m-learning_rate*m_slope;
    return m


# In[13]:


def gd(x,y,learning_rate,num_iterations):
    m=np.zeros(14)
    for i in range(num_iterations):
        m=step_gardient(x,y,learning_rate,m)
        print(i,"cost: ",cost(x,y,m))
        print("hello",i)
    return m


# In[14]:


def run(x,y):
    learning_rate=0.3
    num_iterations=500
    m=gd(x,y,learning_rate,num_iterations)
    print(m)
    return m


# In[15]:


def score(y_truth, y_pred):
    u = ((y_truth - y_pred)**2).sum()
    v = ((y_truth - y_truth.mean())**2).sum()
    return 1 - u/v


# In[16]:


# Training
data=np.loadtxt("Bostontrain.csv",delimiter=",")
data=pd.DataFrame(data)
print(data.shape)
x=data.iloc[0:,0:13]
x['13']=1
x=np.array(x)
print(x.shape[1])
y=np.array(data.iloc[:,13]);

m=run(x,y) 
Ypred=x.dot(m)
y_pred=np.array(Ypred);
print(y.shape)
print(y_pred.shape)
print("Train Score: ",score(y, Ypred))


# In[20]:


# Testing
data_test=np.loadtxt("Bostontest.csv",delimiter=",")
data_test=pd.DataFrame(data_test)
print(data_test.shape)
data_test['13']=1
x_test=np.array(data_test)
y_test_pred=np.array(x_test.dot(m))
print(y_test_pred)

