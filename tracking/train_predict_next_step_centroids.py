# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 22:16:01 2021

@author: Utente
"""



import numpy as np
# from scipy.ndimage.interpolation import shift
from pandas import read_csv
from sklearn.metrics import mean_squared_error 
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from loadData import loadD
from model_struct import gen_fit_model

 

def split_dataset(dataset):
	# split into standard weeks
	train, test = dataset[0:-100], dataset[-100:-10]
	trainC, testC = dataC[0:-100], dataC[-100:-10]
	# restructure into windows of weekly data
	train = np.array(np.split(train, len(train)//30))
	test = np.array(np.split(test, len(test)/30))
	trainC = np.array(np.split(trainC, len(trainC/30)))
	testC = np.array(np.split(testC, len(testC)/30))
	return train, test, trainC, testC

def gen_gt(train, trainC, n_input, n_out = 5):
    """here we use dataC to generate the ground truth as we expect the network to
      predict the next center for each true and faked subjet

      n_input: is the number of inputs the net has to see in order to make a prediction
      n_out: is the number of next time step steps for redicting the centroid
    """
    dataX = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
    dataXC = trainC.reshape((trainC.shape[0]*trainC.shape[1], trainC.shape[2]))
    X, y = list(), list()
    startx = 0

    for _ in range(len(dataX)):
     
        endx = startx + n_input
        out_end = endx + n_out
        # ensure we have enough data for this instance
        if out_end <= len(dataX):
            X.append(dataX[startx : endx, :])
            y.append(dataXC[endx : out_end, :])
        
            startx += 1
    return np.array(X), np.array(y)

 

def model_predict(model, test_data, n_input, j):
    #it is called iteratively the forward prediction "predict model"
    input_x = test_data[j:n_input+j,:] 
    input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
  
    yhat = model.predict(input_x, verbose=1)
    
    yhat = yhat[0]
    return yhat, input_x


def evaluate_predictions(gt, predicted):
    rmse = np.zeros_like(gt)
    mse = np.zeros_like(gt)
    crmse = np.zeros((gt.shape[0], gt.shape[1]//2))
    cmse = np.zeros_like(crmse)
    
    # calculate an RMSE score for each day
    
    s = 0
    for gt_vals, preds in zip(gt, predicted):
        
        # calculate accuracy
         
        mse[s] = mean_squared_error(gt_vals, preds)
        
        # calculate rmse
        rmse[s] = mean_squared_error(gt_vals, preds, squared =False)
        
        cmse[s] = mean_squared_error(gt_vals[0:4], preds[0:4])
        crmse[s] = mean_squared_error(gt_vals[0:4], preds[0:4], squared =False)
        s += 1
        
    ##    
    mseS = np.sum(mse,axis =1)/gt.shape[1]
    rmseS = np.sum(rmse, axis =1)/gt.shape[1]
    
    
    
    plt.figure(1)
    x = np.arange(0,gt.shape[0])
    plt.plot(x, mseS, 'go--', linewidth=2, markersize=2, label = 'mseS')
    plt.plot(x, rmseS, 'ro--', linewidth=2, markersize=2, label = 'rmseS')
    plt.legend(loc="upper left")
    plt.ylim(0, 0.2)
    plt.show()
    
    # adding weights to the wrong data
    mseSc = np.sum(cmse,axis =1)/(gt.shape[1]/2)
    rmseSc = np.sum(crmse, axis =1)/(gt.shape[1]/2)
    
    plt.figure(2)
    x = np.arange(0,gt.shape[0])
    plt.plot(x, mseSc, 'go--', linewidth=2, markersize=2, label = 'mseSc')
    plt.plot(x, rmseSc, 'ro--', linewidth=2, markersize=2, label = 'rmseSc')
    plt.legend(loc="upper left")
    plt.ylim(0, 0.2)
    plt.show()
    
    return mse, rmse


def predict_model(model, train, test, trainC, testC, n_input):

    test_data = test.reshape((test.shape[0]*test.shape[1], test.shape[2]))
    target = testC.reshape((testC.shape[0]*testC.shape[1], testC.shape[2]))
    # walk-forward for each sequence
    predictions = np.zeros((test_data.shape[0]-n_input, testC.shape[2]))
    ver_data_test = np.zeros((test_data.shape[0]-n_input, testC.shape[2]))
    for ii in range(len(test_data)-n_input):
        # print(ii)
        # predict the current sequence:
        # if seq = 10 input of bb + cc at time tk : tk+10
        #        it predicts the cc at time tk+11 
        # ii is the line of prediction
        yhat_next, input_x = model_predict(model, test_data, n_input, ii)
        # store the predictions
        predictions[ii,:] =  yhat_next[0,:]
        # update the gt
        ver_data_test[ii,:] = target[ii + n_input,:]
        # 
    return predictions, ver_data_test


### main 

n_input = 10
dataset, datax, dataC = loadD()

## split dataset in train and test considering that the ground truth is just dataC
train, test, trainC, testC  = split_dataset(dataset)

## Here we generate the appropriate structure for train data and labeling according to n_input 
## we assume that the output time step  is just of length 1
train_x, train_y = gen_gt(train, trainC, n_input) 
model = gen_fit_model(train_x, train_y)
model.summary()

# make prediction with the test and forward W is the ground truth of the centroids
predictions, forwardV = predict_model(model, train, test, trainC, testC, n_input)

mse, rmse = evaluate_predictions(forwardV, predictions)

