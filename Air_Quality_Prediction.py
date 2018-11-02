# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 17:34:19 2018

@author: shl12
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Importing the Training dataset
df = pd.read_csv('dataset/Train.csv')

x = df.values
x = x[:,:-1]
y = df.target
x0 = np.ones((x.shape[0],1))
x = np.append(x0,x,axis = 1)

# Importing the Testing dataset
dt = pd.read_csv('dataset/Test.csv')

x_test = dt.values
x_test = x_test[:,:-1]
y_test = dt.target
x0 = np.ones((x_test.shape[0],1))
x_test = np.append(x0,x_test,axis = 1)

"""
    Number of features = 5
    
"""

def hypothesis(x,theta):
    hyp = 0;
    for i in range(theta.shape[0]):
        hyp+=theta[i]*x[i]
    return hyp

def error_function(theta,x,y):
    error = 0
    
    for i in range(x.shape[0]):
        error += (hypothesis(x[i],theta)-y[i])**2
    error/=2
    return error

def gradient(theta,x,y):
    grad = np.zeros((x.shape[1]))
    
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            grad[j]+=(hypothesis(x[i],theta)-y[i])*x[i][j]
    return grad

def gradient_descent(x,y,learning_rate = 0.001,batch_size = 1):
    theta = np.zeros((x.shape[1],),dtype = np.float64)
    error_rate = []
    for i in range(100):
        grad = gradient(theta,x,y)
        for j in range(x.shape[1]):
            theta[j] -= learning_rate*grad[j]
            error_rate.append(error_function(theta,x,y))
    return theta,error_rate

def y_predict(x,theta):
    y_pred = np.zeros((x.shape[0],),dtype = np.float64)
    for i in range(x.shape[0]):
        y_pred[i] = hypothesis(x[i],theta)
    return y_pred

def coeff_of_determination(yp,ya):
    num = 0
    den = 0
    ym = ya.mean()
    for i in range(yp.shape[0]):
        num += (ya[i]-yp[i])**2
        den += (ya[i]-ym)**2
    return 1-(num/den)

final_theta,error_rate = gradient_descent(x,y)

print(final_theta)

plt.plot(error_rate)

# Result on Training Set

y_pred = y_predict(x,final_theta)

print(y_pred.shape)
print("Training set score : %.4f "%coeff_of_determination(y_pred,y))

# Result on Testing Set

y_pred = y_predict(x_test,final_theta)

print(y_pred.shape)
print("Testing set score : %.4f "%coeff_of_determination(y_pred,y_test))









