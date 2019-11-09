#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd
import scipy as sp
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import DataConversionWarning
import warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
import os
import glob
import matplotlib.pyplot as plt


# In[661]:


def all_markers(base_dir, train_subjects, test_subjects, hp, n, model): # This function creates a model using all marker input data available
  
    training_x = []
    training_y = []
    testing_x = []
    testing_y = []
    train_vector_list = []
    train_labels_list = []
    test_vector_list = []
    test_labels_list = []

    os.chdir(base_dir)

    # Find all matrices created by MATLAB programme, first do right leg stance for training data subjects
    for subject in train_subjects:
        os.chdir(base_dir + subject)
        for name in glob.glob('r_?'):
            x_in = pd.read_csv(name, sep = '\t', index_col=False, skiprows = 0, usecols = [i for i in range(96)])
            y_in = pd.read_csv(name, sep = '\t', index_col=False, skiprows = 0, usecols = ['FORCE PLATE'])
        
            matrix_x = []

            # Create features for model to use
            for i in range(n, len(x_in)):
                matrix_x = x_in[i-n:i]
                delta = (np.max(matrix_x, axis=0) - np.min(matrix_x, axis=0))
                gradient = np.squeeze(np.gradient(matrix_x, axis=0).reshape((-1,1)))
                std_d = np.std(matrix_x, axis=0)

                # Add these features into a vector representing one time frame
                feats = np.hstack((delta, gradient, std_d))
                # Now create a list of these horizontal vectors, adding up to a matrix of features for all time stamps for one trial
                train_vector_list.append(feats)
                train_labels_list.append(y_in.iloc[i])

#             for i in range(len(x_in)):
#                 matrix_x = x_in[i:i+1]
#                 matrix_x = np.squeeze(matrix_x.values)
#                 train_vector_list.append(matrix_x)
#                 train_labels_list.append(y_in.iloc[i])
                
        
        # Now repeat for left leg stance
        for name in glob.glob('l_?'):
            x_in = pd.read_csv(name, sep = '\t', index_col=False, skiprows = 0, usecols = [i for i in range(96)])
            y_in = pd.read_csv(name, sep = '\t', index_col=False, skiprows = 0, usecols = ['FORCE PLATE'])

            matrix_x = []

            # Create features for model to use
            for i in range(n, len(x_in)):
                matrix_x = x_in[i-n:i]
                delta = (np.max(matrix_x, axis=0) - np.min(matrix_x, axis=0))
                gradient = np.squeeze(np.gradient(matrix_x, axis=0).reshape((-1,1)))
                std_d = np.std(matrix_x, axis=0)

                # Add these features into a vector representing one time frame
                feats = np.hstack((delta, gradient, std_d))
                # Now create a list of these horizontal vectors, adding up to a matrix of features for all time stamps for one trial
                train_vector_list.append(feats)
                train_labels_list.append(y_in.iloc[i])

#             for i in range(len(x_in)):
#                 matrix_x = x_in[i:i+1]
#                 matrix_x = np.squeeze(matrix_x.values)
#                 train_vector_list.append(matrix_x)
#                 train_labels_list.append(y_in.iloc[i])
            
        os.chdir("..")
    
    # Create stack of these lists, containing all features, at all times, from all trials, in one matrix. This will serve as data to train the model.
    training_x = np.vstack(train_vector_list)
    training_y = np.stack(train_labels_list)
    
    
    # Now repeat as above for remaining subjects to use as testing data
    for subject in test_subjects:
        os.chdir(base_dir + subject)
        for name in glob.glob('r_?'):
            x_in = pd.read_csv(name, sep = '\t', index_col=False, skiprows = 0, usecols = [i for i in range(96)])
            y_in = pd.read_csv(name, sep = '\t', index_col=False, skiprows = 0, usecols = ['FORCE PLATE'])
        
            matrix_x = []

            # Create features for model to use
            for i in range(n, len(x_in)):
                matrix_x = x_in[i-n:i]
                delta = (np.max(matrix_x, axis=0) - np.min(matrix_x, axis=0))
                gradient = np.squeeze(np.gradient(matrix_x, axis=0).reshape((-1,1)))
                std_d = np.std(matrix_x, axis=0)

                # Add these features into a vector representing one time frame
                feats = np.hstack((delta, gradient, std_d))
                # Now create a list of these horizontal vectors, adding up to a matrix of features for all time stamps for one trial
                test_vector_list.append(feats)
                test_labels_list.append(y_in.iloc[i])
    
#             for i in range(len(x_in)):
#                 matrix_x = x_in[i:i+1]
#                 matrix_x = np.squeeze(matrix_x.values)
#                 test_vector_list.append(matrix_x)
#                 test_labels_list.append(y_in.iloc[i])
    
    
        # Now repeat for left leg stance
        for name in glob.glob('l_?'):
            x_in = pd.read_csv(name, sep = '\t', index_col=False, skiprows = 0, usecols = [i for i in range(96)])
            y_in = pd.read_csv(name, sep = '\t', index_col=False, skiprows = 0, usecols = ['FORCE PLATE'])

            matrix_x = []

            # Create features for model to use
            for i in range(n, len(x_in)):
                matrix_x = x_in[i-n:i]
                delta = (np.max(matrix_x, axis=0) - np.min(matrix_x, axis=0))
                gradient = np.squeeze(np.gradient(matrix_x, axis=0).reshape((-1,1)))
                std_d = np.std(matrix_x, axis=0)

                # Add these features into a vector representing one time frame
                feats = np.hstack((delta, gradient, std_d))
                # Now create a list of these horizontal vectors, adding up to a matrix of features for all time stamps for one trial
                test_vector_list.append(feats)
                test_labels_list.append(y_in.iloc[i])
            
#             for i in range(len(x_in)):
#                 matrix_x = x_in[i:i+1]
#                 matrix_x = np.squeeze(matrix_x.values)
#                 test_vector_list.append(matrix_x)
#                 test_labels_list.append(y_in.iloc[i])
    
    
        os.chdir("..")
    
    # Create stack of these lists, containing all features, at all times, from all trials, in one matrix. This will serve as data to test the model.
    testing_x = np.stack(test_vector_list)
    testing_y = np.stack(test_labels_list)
    
    if model == 'logistic regression':
        # Use p. c. analysis to reduce the number of coefficients used in logistic regression, making the process more efficient
        (pca_training_x, pca_testing_x) = p_c_analysis(training_x, testing_x)
        # Create model using logistic regression and return accuracy and area-under-curve scores
        (accuracy_score, auc_score, lr, predictions) = logistic_regression(pca_training_x, training_y, pca_testing_x, testing_y, hp)
        #lr_graphs(lr, predictions, testing_y)
        
    if model == 'random forest':
        # Create model using decision trees and return accuracy and area-under-curve scores
        (accuracy_score, auc_score, predictions) = decision_trees(training_x, training_y, testing_x, testing_y)
        #graphs(predictions, testing_y)
        
    if model == 'extra trees':
        # Create model using decision trees and return accuracy and area-under-curve scores
        (accuracy_score, auc_score, predictions) = extra_trees(training_x, training_y, testing_x, testing_y)
        #graphs(predictions, testing_y)
        
    if model == 'boosted trees':
        # Create model using decision trees and return accuracy and area-under-curve scores
        #(accuracy_score, auc_score, predictions) = gradient_boosting(training_x, training_y, testing_x, testing_y)
        (accuracy_score, auc_score, predictions) = gradient_boosting(training_x, training_y, training_x, training_y)
        graphs(predictions, training_y)
        

    return training_x, training_y, testing_x, testing_y, accuracy_score, auc_score


# In[662]:


def selected_markers(base_dir, train_subjects, test_subjects, hp, n, model): # This function creates a model using only the selected markers as input data
  
    training_x = []
    training_y = []
    testing_x = []
    testing_y = []
    train_vector_list = []
    train_labels_list = []
    test_vector_list = []
    test_labels_list = []

    os.chdir(base_dir)

    # Find all matrices created by MATLAB programme, first do right leg stance for training data subjects
    for subject in train_subjects:
        os.chdir(base_dir + subject)
        for name in glob.glob('r_?'):
            x_in = pd.read_csv(name, sep = '\t', index_col=False, skiprows = 0, usecols = ['RFAM(y)', 'RFAM(z)', 'RTAM(y)', 'RTAM(z)', 'RFCC(y)', 'RTF(y)', 'RFMT(y)', 'LFAM(y)', 'LTAM(y)', 'LTAM(z)', 'LFCC(x)', 'LFCC(y)', 'LTF(y)','LFMT(x)', 'LFMT(y)'])
            y_in = pd.read_csv(name, sep = '\t', index_col=False, skiprows = 0, usecols = ['FORCE PLATE'])
        
            matrix_x = []

            # Create features for model to use
            for i in range(n, len(x_in)):
                matrix_x = x_in[i-n:i]
                delta = (np.max(matrix_x, axis=0) - np.min(matrix_x, axis=0))
                gradient = np.squeeze(np.gradient(matrix_x, axis=0).reshape((-1,1)))
                gradient_2 = np.gradient(gradient)
                std_d = np.std(matrix_x, axis=0)
                heel_markers_ydifference = delta[5] - delta[11] # Relative differences between specific markers
                heel_toe_ydifference = delta[5] - delta[14]
                ankles_zdifference = delta[2] - delta[9]

                # Add these features into a vector representing one time frame
                feats = np.hstack((delta, gradient, gradient_2, std_d, heel_markers_ydifference, heel_toe_ydifference, ankles_zdifference))                
                # Now create a list of these horizontal vectors, adding up to a matrix of features for all time stamps for one trial
                train_vector_list.append(feats)
                train_labels_list.append(y_in.iloc[i])
        
        # Now repeat for left leg stance
        for name in glob.glob('l_?'):
            x_in = pd.read_csv(name, sep = '\t', index_col=False, skiprows = 0, usecols = ['LFAM(y)', 'LFAM(z)', 'LTAM(y)', 'LTAM(z)', 'LFCC(y)', 'LTF(y)', 'LFMT(y)', 'RFAM(y)', 'RTAM(y)', 'RTAM(z)', 'RFCC(x)', 'RFCC(y)', 'RTF(y)','RFMT(x)', 'RFMT(y)'])
            y_in = pd.read_csv(name, sep = '\t', index_col=False, skiprows = 0, usecols = ['FORCE PLATE'])

            matrix_x = []

            # Create features for model to use
            for i in range(n, len(x_in)):
                matrix_x = x_in[i-n:i]
                delta = (np.max(matrix_x, axis=0) - np.min(matrix_x, axis=0))
                gradient = np.squeeze(np.gradient(matrix_x, axis=0).reshape((-1,1)))
                gradient_2 = np.gradient(gradient)
                std_d = np.std(matrix_x, axis=0)
                heel_markers_ydifference = delta[5] - delta[11] # Relative differences between specific markers
                heel_toe_ydifference = delta[5] - delta[14]
                ankles_zdifference = delta[2] - delta[9]

                # Add these features into a vector representing one time frame
                feats = np.hstack((delta, gradient, gradient_2, std_d, heel_markers_ydifference, heel_toe_ydifference, ankles_zdifference))                
                # Now create a list of these horizontal vectors, adding up to a matrix of features for all time stamps for one trial
                train_vector_list.append(feats)
                train_labels_list.append(y_in.iloc[i])
            
        os.chdir("..")
    
    # Create stack of these lists, containing all features, at all times, from all trials, in one matrix. This will serve as data to train the model.
    training_x = np.stack(train_vector_list)
    training_y = np.stack(train_labels_list)
    

    # Now repeat as above for remaining subjects to use as testing data
    for subject in test_subjects:
        os.chdir(base_dir + subject)
        for name in glob.glob('r_?'):
            x_in = pd.read_csv(name, sep = '\t', index_col=False, skiprows = 0, usecols = ['RFAM(y)', 'RFAM(z)', 'RTAM(y)', 'RTAM(z)', 'RFCC(y)', 'RTF(y)', 'RFMT(y)', 'LFAM(y)', 'LTAM(y)', 'LTAM(z)', 'LFCC(x)', 'LFCC(y)', 'LTF(y)','LFMT(x)', 'LFMT(y)'])
            y_in = pd.read_csv(name, sep = '\t', index_col=False, skiprows = 0, usecols = ['FORCE PLATE'])
        
            matrix_x = []

            # Create features for model to use
            for i in range(n, len(x_in)):
                matrix_x = x_in[i-n:i]
                delta = (np.max(matrix_x, axis=0) - np.min(matrix_x, axis=0))
                gradient = np.squeeze(np.gradient(matrix_x, axis=0).reshape((-1,1)))
                gradient_2 = np.gradient(gradient)
                std_d = np.std(matrix_x, axis=0)
                heel_markers_ydifference = delta[5] - delta[11] # Relative differences between specific markers
                heel_toe_ydifference = delta[5] - delta[14]
                ankles_zdifference = delta[2] - delta[9]

                # Add these features into a vector representing one time frame
                feats = np.hstack((delta, gradient, gradient_2, std_d, heel_markers_ydifference, heel_toe_ydifference, ankles_zdifference))                
                # Now create a list of these horizontal vectors, adding up to a matrix of features for all time stamps for one trial
                test_vector_list.append(feats)
                test_labels_list.append(y_in.iloc[i])
        
        # Now repeat for left leg stance
        for name in glob.glob('l_?'):
            x_in = pd.read_csv(name, sep = '\t', index_col=False, skiprows = 0, usecols = ['LFAM(y)', 'LFAM(z)', 'LTAM(y)', 'LTAM(z)', 'LFCC(y)', 'LTF(y)', 'LFMT(y)', 'RFAM(y)', 'RTAM(y)', 'RTAM(z)', 'RFCC(x)', 'RFCC(y)', 'RTF(y)','RFMT(x)', 'RFMT(y)'])
            y_in = pd.read_csv(name, sep = '\t', index_col=False, skiprows = 0, usecols = ['FORCE PLATE'])

            matrix_x = []

            # Create features for model to use
            for i in range(n, len(x_in)):
                matrix_x = x_in[i-n:i]
                delta = (np.max(matrix_x, axis=0) - np.min(matrix_x, axis=0))
                gradient = np.squeeze(np.gradient(matrix_x, axis=0).reshape((-1,1)))
                gradient_2 = np.gradient(gradient)
                std_d = np.std(matrix_x, axis=0)
                heel_markers_ydifference = delta[5] - delta[11] # Relative differences between specific markers
                heel_toe_ydifference = delta[5] - delta[14]
                ankles_zdifference = delta[2] - delta[9]

                # Add these features into a vector representing one time frame
                feats = np.hstack((delta, gradient, gradient_2, std_d, heel_markers_ydifference, heel_toe_ydifference, ankles_zdifference))                
                # Now create a list of these horizontal vectors, adding up to a matrix of features for all time stamps for one trial
                test_vector_list.append(feats)
                test_labels_list.append(y_in.iloc[i])
            
        os.chdir("..")
    
    # Create stack of these lists, containing all features, at all times, from all trials, in one matrix. This will serve as data to test the model.
    testing_x = np.stack(test_vector_list)
    testing_y = np.stack(test_labels_list)
    
    
    if model == 'logistic regression':
        # Use p. c. analysis to reduce the number of coefficients used in logistic regression, making the process more efficient
        (pca_training_x, pca_testing_x) = p_c_analysis(training_x, testing_x)
        # Create model using logistic regression and return accuracy and area-under-curve scores
        (accuracy_score, auc_score, lr, predictions) = logistic_regression(pca_training_x, training_y, pca_testing_x, testing_y, hp)
        #lr_graphs(lr, predictions, testing_y)
        
    if model == 'random forest':
        # Create model using decision trees and return accuracy and area-under-curve scores
        (accuracy_score, auc_score, predictions) = random_forest(training_x, training_y, testing_x, testing_y)
        #graphs(predictions, testing_y)
        
    if model == 'extra trees':
        # Create model using decision trees and return accuracy and area-under-curve scores
        (accuracy_score, auc_score, predictions) = extra_trees(training_x, training_y, testing_x, testing_y)
        #graphs(predictions, testing_y)
        
    if model == 'boosted trees':
        # Create model using decision trees and return accuracy and area-under-curve scores
        (accuracy_score, auc_score, predictions) = gradient_boosting(training_x, training_y, testing_x, testing_y)
        #graphs(predictions, testing_y)

        
    return training_x, training_y, testing_x, testing_y, accuracy_score, auc_score


# In[663]:


def p_c_analysis(training_x, testing_x): # Use p. c. analysis to reduce the number of coefficients used in logistic regression, making the process more efficient

    # Normalise the data for pca
    scaler = StandardScaler()
    scaler.fit(training_x)
    normalised_training_x = scaler.transform(training_x)
    normalised_testing_x = scaler.transform(testing_x)
    
    # Perform pca on the normalised data
    pca = PCA(n_components=30)
    pca.fit(normalised_training_x)
    pca_training_x = pca.transform(normalised_training_x)
    pca_testing_x = pca.transform(normalised_testing_x)

#     plt.plot(np.cumsum(pca.explained_variance_ratio_))
#     plt.xlabel('number of components')
#     plt.ylabel('cumulative explained variance')
#     plt.show()
    
    return pca_training_x, pca_testing_x


# In[664]:


def logistic_regression(training_x, training_y, testing_x, testing_y, hp): # Create a model using logistic regression

    lr = LogisticRegression(C = hp, solver='lbfgs', max_iter = 100000)
    lr.fit(training_x, training_y)
    
    # Score the model's accuracy and area-under-curve by comparing predictions from the model and testing data
    accuracy_score = lr.score(testing_x, testing_y)
    #np.sum(testing_y) / len(testing_y)
    predictions = lr.predict(testing_x)
    auc_score = roc_auc_score(predictions, testing_y)
    
    return accuracy_score, auc_score, lr, predictions


# In[665]:


def lr_graphs(lr, predictions, testing_y): # Plot graphs showing the weightings of coefficients and comparing predictions vs. testing results

    weights = np.squeeze(lr.coef_)
    plt.bar(range(len(weights)),np.abs(weights))
    plt.show()
    
    plt.plot(predictions[750:1500], 'b')
    plt.plot(testing_y[750:1500], 'y')
    plt.show()


# In[666]:


def random_forest(training_x, training_y, testing_x, testing_y): # Create a model using random forests

    clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=0)
    clf.fit(training_x, training_y)
    
    # Score the model's accuracy and area-under-curve by comparing predictions from the model and testing data
    accuracy_score = clf.score(testing_x, testing_y)
    np.sum(testing_y) / len(testing_y)
    predictions = clf.predict(testing_x)
    auc_score = roc_auc_score(predictions, testing_y)
    
    return accuracy_score, auc_score, predictions


# In[667]:


def extra_trees(training_x, training_y, testing_x, testing_y): # Create a model using extra trees

    clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
    clf.fit(training_x, training_y)
    
    # Score the model's accuracy and area-under-curve by comparing predictions from the model and testing data
    accuracy_score = clf.score(testing_x, testing_y)
    np.sum(testing_y) / len(testing_y)
    predictions = clf.predict(testing_x)
    auc_score = roc_auc_score(predictions, testing_y)
    
    return accuracy_score, auc_score, predictions


# In[668]:


def gradient_boosting(training_x, training_y, testing_x, testing_y): # Create a model using boosted trees

    clf = GradientBoostingClassifier(n_estimators=40, min_samples_leaf=10, random_state=0)
    clf.fit(training_x, training_y)
    
    # Score the model's accuracy and area-under-curve by comparing predictions from the model and testing data
    accuracy_score = clf.score(testing_x, testing_y)
    np.sum(testing_y) / len(testing_y)
    predictions = clf.predict(testing_x)
    auc_score = roc_auc_score(predictions, testing_y)
    
    return accuracy_score, auc_score, predictions


# In[669]:


def gradient_boosting_training(training_x, training_y, testing_x, testing_y): # Create a model using boosted trees

    clf = GradientBoostingClassifier(n_estimators=40, min_samples_leaf=10, random_state=0)
    clf.fit(training_x, training_y)
    
    # Score the model's accuracy and area-under-curve by comparing predictions from the model and testing data
    accuracy_score = clf.score(testing_x, testing_y)
    np.sum(testing_y) / len(testing_y)
    predictions = clf.predict(testing_x)
    auc_score = roc_auc_score(predictions, testing_y)
    
    return accuracy_score, auc_score, predictions


# In[670]:


def graphs(predictions, testing_y):

    # Plot graphs comparing predictions vs. testing results
    plt.plot(predictions[750:1500], 'b')
    plt.plot(testing_y[750:1500], 'y')
    plt.show()


# In[671]:


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


# In[672]:


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


# In[673]:


# Location of processed matrices containing kinematics and forceplate data
base_directory = 'D:/UROP/Processed Force and Kinematics Data/'

# Number of lines to use for calculating features
n = 3

# Hyperparameter - controls 'power' of logistic regression model
hyperparameter = 0.5

# Number of folds to use in k-fold cross-validation
num_folds = 5

# Machine learning model to use; 'logistic regression', 'random forest', 'extra trees' or 'boosted trees'
model = 'boosted trees'

# Which markers to use; 'all' or 'selected'
markers = 'all'


settings(base_directory, hyperparameter, num_folds, n, model, markers)


# # SELECTION OF MARKERS

# #### Using only stance foot markers:
# 
# CV Accuracy: 0.9741732177233782
# CV AUC: 0.9643417577326275
# 
# 
# 
# stance foot markers, without FM2(x)
# 
# CV Accuracy: 0.9741732177233782
# CV AUC: 0.9643417577326275
# 
# identical performance
# 
# 
# 
# stance foot markers, without FM2(x), TF(z)
# 
# CV Accuracy: 0.9743113949640163
# CV AUC: 0.9648734436537708
# 
# better
# 
# 
# 
# stance foot markers, without FM2(x), TF(z), TAM(x), FM2(z)
# 
# CV Accuracy: 0.9746496929510453
# CV AUC: 0.9657321965983574
# 
# better
# 
# 
# 
# stance foot markers, without FM2(x), TF(z), TAM(x), FM2(z), TAM(z)
# 
# CV Accuracy: 0.9746205739307484
# CV AUC: 0.9656713927192053
# 
# worse
# 
# 
# 
# stance foot markers, without FM2(x), TF(z), TAM(x), FM2(z), FMT(x)
# 
# CV Accuracy: 0.9744047717517796
# CV AUC: 0.9651971344587084
# 
# worse
# 
# 
# 
# stance foot markers, without FM2(x), TF(z), TAM(x), FM2(z), FMT(z)
# 
# CV Accuracy: 0.9745101255361386
# CV AUC: 0.965754460763182
# 
# better
# 
# 
# 
# stance foot markers, without FM2(x), TF(z), TAM(x), FM2(z), FMT(z), FM2(y)
# 
# CV Accuracy: 0.9746008025367796
# CV AUC: 0.9659480315441723
# 
# better
# 
# 
# 
# stance foot markers, without FM2(x), TF(z), TAM(x), FM2(z), FMT(z), FM2(y), TAM(y)
# 
# CV Accuracy: 0.9745983690644515
# CV AUC: 0.965613590593799
# 
# worse
# 
# 
# 
# stance foot markers, without FM2(x), TF(z), TAM(x), FM2(z), FMT(z), FM2(y), TF(x)
# 
# CV Accuracy: 0.9747100908639178
# CV AUC: 0.9662127437232808
# 
# better
# 
# 
# 
# stance foot markers, without FM2(x), TF(z), TAM(x), FM2(z), FMT(z), FM2(y), TF(x), FAM(x)
# 
# CV Accuracy: 0.9746274000690663
# CV AUC: 0.966095591059925
# 
# worse
# 
# 
# 
# 
# Using stance and non-stance foot markers:
# 
# stance and non-stance foot markers, without FM2(x), TF(z), TAM(x), FM2(z), FMT(z), FM2(y), TF(x), FAM(x), oFM2(x), oTF(z), oTAM(x), oFM2(z), oFMT(z), oFM2(y), oTF(x), oFAM(x)
# 
# CV Accuracy: 0.9770439628491902
# CV AUC: 0.9687835296029661
# 
# better
# 
# 
# 
# stance and non-stance foot markers, without FM2(x), TF(z), TAM(x), FM2(z), FMT(z), FM2(y), TF(x), FAM(x), oFM2(x), oTF(z), oTAM(x), oFM2(z), oFMT(z), oFM2(y), oTF(x), oFAM(x), oFCC(z)
# 
# CV Accuracy: 0.9769612720543387
# CV AUC: 0.968790419740628
# 
# better
# 
# 
# 
# stance and non-stance foot markers, without FM2(x), TF(z), TAM(x), FM2(z), FMT(z), FM2(y), TF(x), FAM(x), oFM2(x), oTF(z), oTAM(x), oFM2(z), oFMT(z), oFM2(y), oTF(x), oFAM(x), oFCC(z), FMT(x)
# 
# CV Accuracy: 0.9769014455511978
# CV AUC: 0.9685962621327097
# 
# worse
# 
# 
# 
# stance and non-stance foot markers, without FM2(x), TF(z), TAM(x), FM2(z), FMT(z), FM2(y), TF(x), FAM(x), oFM2(x), oTF(z), oTAM(x), oFM2(z), oFMT(z), oFM2(y), oTF(x), oFAM(x), oFCC(z), oTAM(z)
# 
# CV Accuracy: 0.9769815474551091
# CV AUC: 0.9687178024644938
# 
# worse
# 
# 
# 
# stance and non-stance foot markers, without FM2(x), TF(z), TAM(x), FM2(z), FMT(z), FM2(y), TF(x), FAM(x), oFM2(x), oTF(z), oTAM(x), oFM2(z), oFMT(z), oFM2(y), oTF(x), oFAM(x), oFCC(z), oFAM(z)
# 
# CV Accuracy: 0.9770560062195379
# CV AUC: 0.9689343323067604
# 
# better
# 
# 
# 
# stance and non-stance foot markers, without FM2(x), TF(z), TAM(x), FM2(z), FMT(z), FM2(y), TF(x), FAM(x), oFM2(x), oTF(z), oTAM(x), oFM2(z), oFMT(z), oFM2(y), oTF(x), oFAM(x), oFCC(z), oFAM(z), FAM(z)
# 
# CV Accuracy: 0.9767520371022966
# CV AUC: 0.9686471782585038
# 
# worse
# 
# 
# 
# stance and non-stance foot markers, without FM2(x), TF(z), TAM(x), FM2(z), FMT(z), FM2(y), TF(x), FAM(x), oFM2(x), oTF(z), oTAM(x), oFM2(z), oFMT(z), oFM2(y), oTF(x), oFAM(x), oFCC(z), oFAM(z), FCC(x)
# 
# CV Accuracy: 0.9770007013867439
# CV AUC: 0.9690318607320305
# 
# better
# 
# 
# 
# stance and non-stance foot markers, without FM2(x), TF(z), TAM(x), FM2(z), FMT(z), FM2(y), TF(x), FAM(x), oFM2(x), oTF(z), oTAM(x), oFM2(z), oFMT(z), oFM2(y), oTF(x), oFAM(x), oFCC(z), oFAM(z), FCC(x), FCC(z)
# CV Accuracy: 0.9768934436992998
# CV AUC: 0.9691555683520676
# better
# 
# stance and non-stance foot markers, without FM2(x), TF(z), TAM(x), FM2(z), FMT(z), FM2(y), TF(x), FAM(x), oFM2(x), oTF(z), oTAM(x), oFM2(z), oFMT(z), oFM2(y), oTF(x), oFAM(x), oFCC(z), oFAM(z), FCC(x), FCC(z), FMT(x)
# 
# CV Accuracy: 0.9768779070397787
# CV AUC: 0.9691853837603978
# 
# better
# 
# 
# 
# stance and non-stance foot markers, without FM2(x), TF(z), TAM(x), FM2(z), FMT(z), FM2(y), TF(x), FAM(x), oFM2(x), oTF(z), oTAM(x), oFM2(z), oFMT(z), oFM2(y), oTF(x), oFAM(x), oFCC(z), oFAM(z), FCC(x), FCC(z), FMT(x), oFCC(x)
# 
# CV Accuracy: 0.9767215754440859
# CV AUC: 0.9690448211215952
# 
# worse
# 
# 
# 
# stance and non-stance foot markers, without FM2(x), TF(z), TAM(x), FM2(z), FMT(z), FM2(y), TF(x), FAM(x), oFM2(x), oTF(z), oTAM(x), oFM2(z), oFMT(z), oFM2(y), oTF(x), oFAM(x), oFCC(z), oFAM(z), FCC(x), FCC(z), FMT(x), oFMT(x)
# 
# CV Accuracy: 0.9766087354176362
# CV AUC: 0.9691793774468784
# 
# worse
# 
# 
# 
# 
# 
# Random Forest - selected markers
# 
# CV Accuracy: 0.9784934222909767
# CV AUC: 0.9682335704259708
# 
# 
# 
# Random Forest - all markers
# 
# CV Accuracy: 0.9808945806959078
# CV AUC: 0.9719111962588887
# 
# 
# 
# Extra trees - selected markers
# 
# CV Accuracy: 0.9781856703384495
# CV AUC: 0.9668894039349303
# 
# 
# 
# Extra trees - all markers
# 
# CV Accuracy: 0.979275395853435
# CV AUC: 0.9694766476317124
# 
# 
# 
# Boosted trees - all markers, 40 estimators
# 
# CV Accuracy: 0.9804626807809615
# CV AUC: 0.9713841831501728

# In[ ]:




