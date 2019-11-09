# -*- coding: utf-8 -*-
"""AutoDetect.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wGIXU65xT9WIdZNHw0c3iOxWGgtrUZKA
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import pandas as pd
import scipy as sp
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.exceptions import DataConversionWarning
import warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
import os
import glob
import matplotlib.pyplot as plt
from google.colab import drive
drive.mount('/content/drive')
from keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv1D, MaxPool1D, Dropout
from tensorflow.keras import Model

def Normalise_data(training_x, testing_x):

    # Normalise the data for pca
    scaler = StandardScaler()
    scaler.fit(training_x)
    normalised_training_x = scaler.transform(training_x)
    normalised_testing_x = scaler.transform(testing_x)
    
    return normalised_training_x, normalised_testing_x

def n_all_markers(base_dir, train_subjects, test_subjects, hp, n, model): # This function creates a model using all marker input data available

    training_x = []
    training_y = []
    testing_x = []
    testing_y = []
    train_vector_list = []
    train_labels_list = []
    test_vector_list = []
    test_labels_list = []
    n_training_x = []
    n_training_y = []
    n_testing_x = []
    n_testing_y = []
    n_train_vector_list = []
    n_train_labels_list = []
    n_test_vector_list = []
    n_test_labels_list = []


    os.chdir(base_dir)

    # Find all matrices created by MATLAB programme, first do right leg stance for training data subjects
    for subject in train_subjects:
        os.chdir(base_dir + subject)
        for name in glob.glob('r_?'):
            x_in = pd.read_csv(name, sep = '\t', index_col=False, skiprows = 0, usecols = [i for i in range(96)])
            y_in = pd.read_csv(name, sep = '\t', index_col=False, skiprows = 0, usecols = ['FORCE PLATE'])

            matrix_x = []

            # Create features for model to use
            for i in range(len(x_in)):
                matrix_x = x_in[i:i+1]
                matrix_x = np.squeeze(matrix_x.values)
                train_vector_list.append(matrix_x)
                train_labels_list.append(y_in.iloc[i])


        # Now repeat for left leg stance
        for name in glob.glob('l_?'):
            x_in = pd.read_csv(name, sep = '\t', index_col=False, skiprows = 0, usecols = [i for i in range(96)])
            y_in = pd.read_csv(name, sep = '\t', index_col=False, skiprows = 0, usecols = ['FORCE PLATE'])

            matrix_x = []

            # Create features for model to use
            for i in range(len(x_in)):
                matrix_x = x_in[i:i+1]
                matrix_x = np.squeeze(matrix_x.values)
                train_vector_list.append(matrix_x)
                train_labels_list.append(y_in.iloc[i])

        os.chdir("..")

    # Create stack of these lists, containing all features, at all times, from all trials, in one matrix. This will serve as data to train the model.
    training_x = np.stack(train_vector_list)
    training_y = np.stack(train_labels_list)


    # Now repeat as above for remaining subjects to use as testing data
    for subject in test_subjects:
        os.chdir(base_dir + subject)
        for name in glob.glob('r_?'):
            x_in = pd.read_csv(name, sep = '\t', index_col=False, skiprows = 0, usecols = [i for i in range(96)])
            y_in = pd.read_csv(name, sep = '\t', index_col=False, skiprows = 0, usecols = ['FORCE PLATE'])

            matrix_x = []

            # Create features for model to use
            for i in range(len(x_in)):
                matrix_x = x_in[i:i+1]
                matrix_x = np.squeeze(matrix_x.values)
                test_vector_list.append(matrix_x)
                test_labels_list.append(y_in.iloc[i])


        # Now repeat for left leg stance
        for name in glob.glob('l_?'):
            x_in = pd.read_csv(name, sep = '\t', index_col=False, skiprows = 0, usecols = [i for i in range(96)])
            y_in = pd.read_csv(name, sep = '\t', index_col=False, skiprows = 0, usecols = ['FORCE PLATE'])

            matrix_x = []

            # Create features for model to use           
            for i in range(len(x_in)):
                matrix_x = x_in[i:i+1]
                matrix_x = np.squeeze(matrix_x.values)
                test_vector_list.append(matrix_x)
                test_labels_list.append(y_in.iloc[i])


        os.chdir("..")

    # Create stack of these lists, containing all features, at all times, from all trials, in one matrix. This will serve as data to test the model.
    testing_x = np.stack(test_vector_list)
    testing_y = np.stack(test_labels_list)


    (normalised_training_x, normalised_testing_x) = Normalise_data(training_x, testing_x)


    if model == 'ff neural network':
        for i in range(n, len(normalised_training_x), n):
            matrix_x = normalised_training_x[i-n:i]
            n_train_vector_list.append(matrix_x)
            n_train_labels_list.append(np.squeeze(training_y[i]))

        n_training_x = np.expand_dims(np.stack(n_train_vector_list), axis=3)
        n_training_y = np.stack(n_train_labels_list)

        for i in range(n, len(normalised_testing_x), n):
            matrix_x = normalised_testing_x[i-n:i]
            n_test_vector_list.append(matrix_x)
            n_test_labels_list.append(np.squeeze(testing_y[i]))

        n_testing_x = np.expand_dims(np.stack(n_test_vector_list), axis=3)
        n_testing_y = np.stack(n_test_labels_list)
        
        (predictions, accuracy_score, auc_score) = ff_neural_network(n_training_x, n_training_y, n_testing_x, n_testing_y)
        
        graphs(predictions, n_testing_y)
    
    
    if model == 'conv neural network':
        for i in range(n, len(normalised_training_x)):
            matrix_x = normalised_training_x[i-n:i]
            n_train_vector_list.append(matrix_x)
            n_train_labels_list.append(np.squeeze(training_y[i]))

        n_training_x = np.stack(n_train_vector_list)
        n_training_y = np.stack(n_train_labels_list)

        for i in range(n, len(normalised_testing_x), n):
            matrix_x = normalised_testing_x[i-n:i]
            n_test_vector_list.append(matrix_x)
            n_test_labels_list.append(np.squeeze(testing_y[i]))

        n_testing_x = np.stack(n_test_vector_list)
        n_testing_y = np.stack(n_test_labels_list)
        
        (predictions, accuracy_score, auc_score) = conv_neural_network(n_training_x, n_training_y, n_testing_x, n_testing_y)
    
        graphs(predictions, n_testing_y)
      
    return accuracy_score, auc_score

def ff_neural_network(n_training_x, n_training_y, n_testing_x, n_testing_y):

    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(10, 96, 1)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.25),
      tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'AUC'])

    model.fit(n_training_x, to_categorical(n_training_y), batch_size=64, epochs=5, 
              validation_data=[n_testing_x, to_categorical(n_testing_y)], verbose=0, shuffle=True)

    raw_predictions = model.predict(n_testing_x)
    
    predictions = []
    for i in range (len(raw_predictions)):
        if raw_predictions[i,0] > 0.5:
            predictions.append(0)
        else:
            predictions.append(1)
    
    results = model.evaluate(n_testing_x, to_categorical(n_testing_y))
    accuracy_score = (results[1])
    auc_score = (results[2])
    
    return predictions, accuracy_score, auc_score

def conv_neural_network(n_training_x, n_training_y, n_testing_x, n_testing_y):

    model = tf.keras.models.Sequential([
      tf.keras.layers.Conv1D(16, 3, activation='relu'),
      tf.keras.layers.Conv1D(16, 3, activation='relu'),
      tf.keras.layers.Conv1D(16, 3, activation='relu'),
      tf.keras.layers.MaxPool1D(pool_size = 2),
      tf.keras.layers.Flatten(input_shape=(1, 96)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.25),
      tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'AUC'])

    model.fit(n_training_x, to_categorical(n_training_y), batch_size=64, epochs=5, 
              validation_data=[n_testing_x, to_categorical(n_testing_y)], verbose=0, shuffle=True)

    raw_predictions = model.predict(n_testing_x)
    
    predictions = []
    for i in range (len(raw_predictions)):
        if raw_predictions[i,0] > 0.5:
            predictions.append(0)
        else:
            predictions.append(1)
    
    results = model.evaluate(n_testing_x, to_categorical(n_testing_y))
    accuracy_score = (results[1])
    auc_score = (results[2])
    
    
    return predictions, accuracy_score, auc_score

def graphs(predictions, testing_y):

    # Plot graphs comparing predictions vs. testing results
    plt.plot(predictions[:100], 'b')
    plt.plot(testing_y[:100], 'y')
    plt.show()

def cross_validate(base_dir, hp, folds, n, model, markers): # This function uses cross-validation to determine the true performance of the model using data from selected markers as inputs
    
    print('Cross-validating')
    
    start = 0
    total = len(os.listdir(base_dir))
    end = subs_per_fold = total // folds

    accuracy_scores = []
    auc_scores = []


    for k in range(folds):
    
        print('Completed: '+ str(k) +'/' + str(folds) + ' folds')
        
        test_subjects = os.listdir(base_dir)[start:end]
        train_subjects = [i for i in os.listdir(base_dir) if i not in test_subjects]
    
        # Trains and tests for each fold, returning scores, using desired markers
        if markers == 'all':
          if model == 'ff neural network' or 'conv neural network':
            (accuracy_score, auc_score) = n_all_markers(base_dir, train_subjects, test_subjects, hyperparameter, n, model)
          else:
            (training_x, training_y, testing_x, testing_y, accuracy_score, auc_score) = all_markers(base_dir, train_subjects, test_subjects, hyperparameter, n, model)
        
        if markers == 'selected':
            (training_x, training_y, testing_x, testing_y, accuracy_score, auc_score) = selected_markers(base_dir, train_subjects, test_subjects, hyperparameter, n, model)
        
        accuracy_scores.append(accuracy_score)
        auc_scores.append(auc_score)
    
        start += subs_per_fold
        end += subs_per_fold
        

    print('Completed: '+ str(folds) +'/' + str(folds) + ' folds')
    
    # Find mean scores, finds cross-validated scores
    cv_accuracy = np.mean(accuracy_scores)
    cv_auc = np.mean(auc_scores)
    
    return cv_accuracy, cv_auc

def settings(base_dir, hp, folds, n, model, markers): # Take in settings and apply desired model
    
    # Print settings information
    print('Settings')
    print('Model:\t\t' + model)
    print('Base directory:\t' + str(base_dir))
    print('Folds:\t\t' + str(folds))
    print('n:\t\t' + str(n))
    if model == 'logistic regression':
        print('Hyperparameter:\t' + str(hp))
    print('Markers:\t' + markers)
    print('---------------------------------------')
    
    (cv_accuracy, cv_auc) = cross_validate(base_dir, hp, folds, n, model, markers)
    
    # Print cross-validated scores
    print('---------------------------------------')
    print('CV Accuracy:\t' + str(cv_accuracy))
    print('CV AUC:\t\t' + str(cv_auc))

# Machine learning model to use; 'logistic regression', 'random forest', 'extra trees', 'boosted trees', 'ff neural network' or 'conv neural network'
model = 'conv neural network'

# Location of processed matrices containing kinematics and forceplate data
if model == 'ff neural network' or 'conv neural network':
  base_directory = '/content/drive/My Drive/Colab Notebooks/Processed Force and Kinematics Data/'
else:
  base_directory = 'D:/UROP/Processed Force and Kinematics Data/'

# Number of lines to use for calculating features
n = 10

# Hyperparameter - controls 'power' of logistic regression model
hyperparameter = 0.5

# Number of folds to use in k-fold cross-validation
num_folds = 5

# Which markers to use; 'all' or 'selected'
markers = 'all'


settings(base_directory, hyperparameter, num_folds, n, model, markers)
