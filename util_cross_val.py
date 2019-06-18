"""Util file for tuning the hyperparameters. Used by Snakefile_new_top_layer"""
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras import initializers
from keras.optimizers import *
#import keras.optimizer
from keras.models import load_model
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from keras import initializers

def create_new_model(bottleneck_model):
	def bm():
		model = Sequential()
		model.add(bottleneck_model)
		model.add(Flatten())
		model.add(Dense(1))
		model.compile(optimizer = "Adam", loss = "mean_squared_error")
		return model
	return bm
def create_top_model_fn(y_true):
	def bm():
		model = Sequential()
		model.add(Dense(1,input_dim=384,bias_initializer = initializers.Constant(value = np.mean(y_true))))
		Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00, amsgrad=False)
		model.compile(optimizer = "Adam", loss = "mean_squared_error")
		return model
	return bm

def create_top_model(y_true):
	model = Sequential()
	model.add(Dense(1,input_dim=384,bias_initializer = initializers.Constant(value = np.mean(y_true))))
	Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00, amsgrad=False)
	model.compile(optimizer = "Adam", loss = "mean_squared_error")
	return model


#def train_and_evaluate_model_stitch(model, data_train, y_train, data_val, y_val):
	#model.fit(data_train,y_train,batch_size=128, epochs=25,validation_data=(data_val,y_val))

#def train_and_evaluate_model_top_layer(model, data_train, y_train, data_val, y_val):
	#model.fit(data_train,y_true_train,batch_size=64,
                      #epochs=30,validation_data=(data_val,y_val))
