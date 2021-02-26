# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 13:28:21 2021

@author: Utente
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.activations import elu

def gen_fit_model(train_x, train_y):
    verbose, epochs, batch_size = 1, 10, 10
    rep = 5
    n_timesteps, n_features, n_outputs, n_feat_out = train_x.shape[1], train_x.shape[2], train_y.shape[1], train_y.shape[2]
    # reshape output into [samples, timesteps, features]
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], train_y.shape[2]))
    # define model
    model = keras.Sequential()
    model.add(layers.LSTM(220, activation='elu', input_shape=(n_timesteps, n_features)))
    model.add(layers.RepeatVector(n_outputs))
    model.add(layers.LSTM(220, activation='elu', return_sequences=True))
    
    model.add(layers.TimeDistributed(layers.Dense(100, activation='relu')))
    model.add(layers.TimeDistributed(layers.Dense(n_feat_out)))
    model.compile(loss='mse', optimizer='adam')
    # fit network
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    
    return model