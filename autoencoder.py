# -*- coding: utf-8 -*-
"""Autoencoder.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sFSk5kjG7n74NquoJEn6KMVZAFWMOFmj
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import pandas as pd
import scipy as sp
import sklearn
import math
from sklearn.preprocessing import StandardScaler
import os
import glob
import random
import matplotlib.pyplot as plt
from google.colab import drive
drive.mount('/content/drive')
import keras
from keras.utils import to_categorical
from keras import optimizers
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv1D, MaxPool1D, Dropout
from tensorflow.keras import Model
from tqdm import tqdm

def Normalise_data(training, testing):

    # Normalise the data for pca
    scaler = StandardScaler()
    scaler.fit(training)
    normalised_training = scaler.transform(training)
    normalised_testing = scaler.transform(testing)
    
    return normalised_training, normalised_testing

def n_all_markers(base_dir, train_subjects, test_subjects, model): # This function creates a model using all marker input data available

    training_y = []
    testing_y = []
    train_labels_list = []
    test_labels_list = []
    n_training_y = []
    n_testing_y = []
    n_train_labels_list = []
    n_test_labels_list = []


    os.chdir(base_dir)

    # Find all matrices created by MATLAB programme, first do right leg stance for training data subjects
    for subject in train_subjects:
        os.chdir(base_dir + subject)
        for name in glob.glob('r_?'):
            x_in = pd.read_csv(name, sep = '\t', index_col=False, skiprows = 0, usecols = [i for i in range(96)])            

            matrix_y = []
            
            if len(x_in) > 60:
                r = random.randint(0,len(x_in)-60)

                # Create features for model to use
                for i in range(r, r+60):
                    matrix_y = x_in[i:i+1]
                    matrix_y = np.squeeze(matrix_y.values)
                    train_labels_list.append(matrix_y)


        # Now repeat for left leg stance
        for name in glob.glob('l_?'):
            x_in = pd.read_csv(name, sep = '\t', index_col=False, skiprows = 0, usecols = [i for i in range(96)])

            matrix_y = []
            
            if len(x_in) > 60:
                r = random.randint(0,len(x_in)-60)

                # Create features for model to use
                for i in range(r, r+60):
                    matrix_y = x_in[i:i+1]
                    matrix_y = np.squeeze(matrix_y.values)
                    train_labels_list.append(matrix_y)
                

        os.chdir("..")

    # Create stack of these lists, containing all features, at all times, from all trials, in one matrix. This will serve as data to train the model.
    training_y = np.stack(train_labels_list)


    # Now repeat as above for remaining subjects to use as testing data
    for subject in test_subjects:
        os.chdir(base_dir + subject)
        for name in glob.glob('r_?'):
            x_in = pd.read_csv(name, sep = '\t', index_col=False, skiprows = 0, usecols = [i for i in range(96)])

            matrix_y = []
            
            if len(x_in) > 60:
                r = random.randint(0,len(x_in)-60)

                # Create features for model to use
                for i in range(r, r+60):
                    matrix_y = x_in[i:i+1]
                    matrix_y = np.squeeze(matrix_y.values)
                    test_labels_list.append(matrix_y)


        # Now repeat for left leg stance
        for name in glob.glob('l_?'):
            x_in = pd.read_csv(name, sep = '\t', index_col=False, skiprows = 0, usecols = [i for i in range(96)])

            matrix_y = []
            
            if len(x_in) > 60:
                r = random.randint(0,len(x_in)-60)

                # Create features for model to use           
                for i in range(r, r+60):
                    matrix_y = x_in[i:i+1]
                    matrix_y = np.squeeze(matrix_y.values)
                    test_labels_list.append(matrix_y)


        os.chdir("..")

    # Create stack of these lists, containing all features, at all times, from all trials, in one matrix. This will serve as data to test the model.
    testing_y = np.stack(test_labels_list)


    # Normalise data
    (normalised_training_y, normalised_testing_y) = Normalise_data(training_y, testing_y)

    
    # Create stacks of normalised matrices which are each of 60 rows; i.e. dimensions (no. of trials, 60, 96)  
    for i in range(0, len(normalised_training_y), 60):
        matrix_y = normalised_training_y[i:i+60]
        n_train_labels_list.append(matrix_y)
    
    for i in range(0, len(normalised_testing_y), 60):
        matrix_y = normalised_testing_y[i:i+60]
        n_test_labels_list.append(matrix_y) 
  
    
    # Give extra dimension to these matrices for 2D convolution
    n_training_y = np.expand_dims(np.stack(n_train_labels_list), axis=3)   
    n_testing_y = np.expand_dims(np.stack(n_test_labels_list), axis=3)

    
    # Create uncorrupted features matrices by copying labels matrices
    n_training_x = n_training_y.copy()
    n_testing_x = n_testing_y.copy()
    
    
    # Run model on data and return results
    (predictions, mse) = conv_neural_network(n_training_x, n_training_y, n_testing_x, n_testing_y)
    
    
    # Plot graphs comparing predictions and labels
    graphs(predictions, n_testing_y)
    
    
    return mse

def corrupt(features): # Corrupt data statically
    
    # Randomly corrupt 10% of data
    for i in range(len(features)):
        for k in range(96):
            for j in range(60):
                r = random.randint(0,60)
                if r < 6: 
                    features[i,j,k,0] = 0
                
    return features

def gen_corrupt(features): # Dynamically corrupt data for generator
    
    features2 = features.copy()
    
    # Randomly corrupt 10% of data
    for k in range(96):
        for j in range(60):
            r = random.randint(0,60)
            if r < 6: 
                features2[j,k,0] = 0
                
    return features2

def generator(features, labels, batch_size): # Generate dynamically changing data
  
  # Create empty arrays to contain batch of features and labels
  batch_features = np.zeros((batch_size, 60, 96, 1))
  batch_labels = np.zeros((batch_size, 60, 96, 1))
  
  while True:    
      for i in range(batch_size):
         
          # Choose random index in features
          index = random.randint(0,len(features)-1)
          batch_features[i,:,:,:] = gen_corrupt(features[index,:,:,:])
          batch_labels[i,:,:,:] = labels[index,:,:,:]
        
      yield batch_features, batch_labels

def conv_neural_network(n_training_x, n_training_y, n_testing_x, n_testing_y):

    # Layers of model
    model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(filters=128, kernel_size=2, strides=1, padding='same', activation='relu'),
      tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='same'),
      tf.keras.layers.Conv2D(filters=64, kernel_size=2, strides=1, padding='same', activation='relu'),
      tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='same'),

      tf.keras.layers.Conv2D(filters=64, kernel_size=2, strides=1, padding='same', activation='relu'),
      
      tf.keras.layers.UpSampling2D((2,2)),
      tf.keras.layers.Conv2D(filters=64, kernel_size=2, strides=1, padding='same', activation='relu'),
      tf.keras.layers.UpSampling2D((2,2)),
      tf.keras.layers.Conv2D(filters=128, kernel_size=2, strides=1, padding='same', activation='relu'),
        
      tf.keras.layers.Conv2D(filters=1, kernel_size=2, strides=1, padding='same'),
      
      tf.keras.layers.Reshape((60, 96, 1))
    ])

    
    # Decaying learning rate for optimizer
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        0.001,
        decay_steps = 1000,
        decay_rate = 0.95,
        staircase = True)
    
    opt_adam = tf.keras.optimizers.Adam(lr_schedule)
    
    # Compile model
    model.compile(loss='mse', metrics = ['mse'], optimizer=opt_adam)
   
  
    # Corrupt data, then fit model using this static data
    n_training_x = corrupt(n_training_x)
    n_testing_x = corrupt(n_testing_x)
    model.fit(n_training_x, n_training_y, batch_size=64, epochs=300, validation_data=[n_testing_x, n_testing_y], verbose=0, shuffle=True)
    results = model.evaluate(n_testing_x, n_testing_y)


    # Fit model using dynamically corrupted data
#     BS = 8  
#     EPOCHS = 10
#     n_training_x = n_training_x[:,:]
#     n_training_y = n_training_y[:,:]
#     train_gen = generator(n_training_x, n_training_y, batch_size=BS)
#     valid_gen = generator(n_testing_x, n_testing_y, batch_size=BS)
#     model.fit_generator(train_gen, steps_per_epoch=n_training_x.shape[0] // BS, validation_data=valid_gen, validation_steps=n_testing_x.shape[0] // BS, epochs=EPOCHS, verbose=2, use_multiprocessing=True)
#     results = model.evaluate_generator(valid_gen, steps=n_testing_x.shape[0] // BS, use_multiprocessing=True)
    
    
    predictions = model.predict(n_testing_x)
    mse = results[0]
    
    return predictions, mse

def graphs(predictions, n_testing_y):

    # Create 2D arrays that can be plotted
    predictions = np.reshape(predictions, (60*len(predictions), 96))
    n_testing_y = np.reshape(n_testing_y, (60*len(n_testing_y), 96))
    
    # Plot graphs comparing predictions vs. testing results
    fig= plt.figure(figsize=(15,5))
    plt.plot(predictions[:180,19], 'b')
    plt.plot(n_testing_y[:180,19], 'y')
    plt.plot(predictions[:180,61], 'b')
    plt.plot(n_testing_y[:180,61], 'y')
    plt.show()

def cross_validate(base_dir, folds, model, markers): # This function uses cross-validation to determine the true performance of the model using data from selected markers as inputs
    
    print('Cross-validating')
    
    start = 0
    total = len(os.listdir(base_dir))
    end = subs_per_fold = total // folds

    mse_scores = []


    # Progress bar which updates with each fold
    with tqdm(total=folds) as pbar:
      
        for k in range(folds):

            pbar.update(1)

            # Separate training and testing subjects
            test_subjects = os.listdir(base_dir)[start:end]
            train_subjects = [i for i in os.listdir(base_dir) if i not in test_subjects]

            # Trains and tests for each fold, returning scores, using desired markers
            (mse) = n_all_markers(base_dir, train_subjects, test_subjects, model)

            mse_scores.append(mse)

            start += subs_per_fold
            end += subs_per_fold
              

    # Find mean scores, finds cross-validated scores
    cv_mse = np.mean(mse_scores)
    
    return cv_mse

def settings(base_dir, folds, model, markers): # Take in settings and apply desired model
    
    # Print settings information
    print('Settings')
    print('Model:\t\t' + model)
    print('Base directory:\t' + str(base_dir))
    print('Folds:\t\t' + str(folds))
    print('Markers:\t' + markers)
    print('---------------------------------------')
    
    (cv_mse) = cross_validate(base_dir, folds, model, markers)
    
    # Print cross-validated scores
    print('---------------------------------------')
    print('CV MSE:\t' + str(cv_mse))

# Machine learning model to use; 'logistic regression', 'random forest', 'extra trees', 'boosted trees', 'ff neural network' or 'conv neural network'
model = 'conv neural network'

# Location of processed matrices containing kinematics and forceplate data
if model == 'ff neural network' or 'conv neural network':
  base_directory = '/content/drive/My Drive/Colab Notebooks/Processed Force and Kinematics Data/'
else:
  base_directory = 'D:/UROP/Processed Force and Kinematics Data/'

# Number of folds to use in k-fold cross-validation
num_folds = 5

# Which markers to use; 'all' or 'selected'
markers = 'all'


settings(base_directory, num_folds, model, markers)

"""CV MSE:	0.013508321146073327"""

