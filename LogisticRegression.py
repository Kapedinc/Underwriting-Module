# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 18:19:17 2022

@author: michael de guzman
"""

# import libraries
from typing import Any

from pandas import DataFrame
from pandas.io.parsers import TextFileReader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from collections import OrderedDict
plt.rc("font", size=14)


class IndividualPredictions:
    
    def __init__(self, training_data, individual_testing_data, individual_testing_data_features, probability_approval,
                 find_features, overdraft):
        self.training_data = training_data
        self.individual_testing_data_features = individual_testing_data_features
        self.individual_testing_data = individual_testing_data
        self.y_training = training_data['Credit_Approval']
        self.x_training = training_data.drop('Credit_Approval', axis=1)
        self.x_training = self.x_training.iloc[:, 1:]
        self.os_x_train = pd.DataFrame()
        self.os_y_train = pd.DataFrame()
        self.os_x_test = pd.DataFrame()
        self.os_y_test = pd.DataFrame()
        self.os_x_training = pd.DataFrame()
        self.os_x_testing = pd.DataFrame()
        self.model = pd.DataFrame()
        self.columns = pd.DataFrame()
        self.predictions = np.zeros(shape=(90, 1))
        self.confusion_matrix_training = np.zeros(shape=(2, 2))
        self.rescaled_x = np.zeros(shape=(10, 15))
        self.ind_pred = ""
        self.probability_approval = probability_approval
        self.preliminary = ""
        self.prel_var = 0
        self.find_features = find_features
        self.overdraft = overdraft
        
        # Over-sampling using SMOTE

        os = SMOTE(random_state=0)
        self.os_x_train, self.os_x_test, self.os_y_train, self.os_y_test = train_test_split(self.x_training, 
                                                                                            self.y_training,
                                                                                            test_size=0.4,
                                                                                            random_state=0)

        scaler = MinMaxScaler()
        
        self.os_x_training = pd.DataFrame(scaler.fit_transform(self.os_x_train), columns=self.os_x_train.columns)

        self.os_x_testing = pd.DataFrame(scaler.fit_transform(self.os_x_test), columns=self.os_x_test.columns)
        
        self.model = LogisticRegression(random_state=0)
        
        self.model.fit(self.os_x_training, self.os_y_train)

        if self.find_features == 1:
            importance = pd.DataFrame(data={
                'Attribute': self.os_x_training.columns,
                'Importance': self.model.coef_[0]})

            importance = importance.sort_values(by='Importance', ascending=False)
            print(importance)

            plt.bar(x=importance['Attribute'], height=importance['Importance'], color='#087E8B')
            plt.title('Feature importance obtained from coefficients', size=15)
            plt.xticks(rotation='vertical')
            plt.xlabel('xlabel', fontsize=8)
            plt.ylabel('ylabel', fontsize=8)
            plt.show()

        self.columns = self.os_x_training.columns

        self.predictions = (self.model.predict(self.os_x_testing) >= self.probability_approval).astype(int)
        
        # self.predictions = self.model.predict_proba(self.os_x_testing)
        
        '''
        scikit-learn has an excellent built-in module called classification_report 
        that makes it easy to measure the performance of a classification 
        machine learning model. 
        '''
        print(classification_report(self.os_y_test, self.predictions))
        print(confusion_matrix(self.os_y_test, self.predictions))
        self.confusion_matrix_training = confusion_matrix( self.os_y_test, self.predictions)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(self.confusion_matrix_training)
        ax.grid(False)
        ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted Approved', 'Predicted Not Approved'))
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual Approved', 'Actual Not Approved'))
        ax.set_ylim(1.5, -0.5)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, self.confusion_matrix_training[i, j], ha='center', va='center', color='red')
        plt.show()
        
        '''
        # Grid searching and making the model perform better

        scikit-learn's implementation of logistic regression consists of 
        different hyperparameters but we will grid search over the following two:
        -tol
        -max_iter
        '''
        
        # Define the grid of values for tol and max_iter
        tol = [0.01, 0.001, 0.0001]
        max_iter = [100, 150, 200]

        # Create a dictionary where tol and max_iter are keys and 
        # the lists of their values are corresponding values
        param_grid = dict(tol=tol, max_iter=max_iter)
        
        '''
        Finding the best performing model
        We have defined the grid of hyperparameter 
        values and converted them into a single dictionary format 
        which GridSearchCV() expects as one of its parameters. 
        Now, we will begin the grid search to see which values perform best.

        We will instantiate GridSearchCV() with our earlier 
        logreg model with all the data we have. Instead of passing train 
        and test sets separately, we will supply X (scaled version) and y. 
        We will also instruct GridSearchCV() to perform a cross-validation of five folds.
        We'll end the notebook by storing the best-achieved score and the 
        respective best parameters.

        While building this credit card predictor, we tackled some of the
        most widely-known preprocessing steps such as scaling, label encoding, 
        and missing value imputation. We finished with some machine learning to 
        predict if a person's application for a credit card would get approved 
        or not given some information about that person.
        '''
        
        grid_model = GridSearchCV(estimator=self.model, param_grid = param_grid, cv=5)
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Use scaler to rescale X and assign it to rescaled_x
        self.rescaled_x = scaler.fit_transform(self.os_x_training)
        
        # Fit data to grid_model
        grid_model_result = grid_model.fit(self.rescaled_x, self.os_y_train)
        
        # Summarize results
        best_score, best_params = grid_model_result.best_score_, grid_model_result.best_params_
        print("Best: %f using %s" % (best_score, best_params))
       
        print('accuracy score: ', accuracy_score(y_true=self.os_y_test, y_pred=self.predictions))

    def preliminary_check(self):
        
        if self.individual_testing_data_features["Balance_2d_Mo"] - self.individual_testing_data_features["Limit_Requested"] <= 0:
            self.preliminary = "Manual Underwriting Required.  Limit Should be Lowered."
        else:
            self.preliminary = "Proceed to Automatic Underwriting"
            self.prel_var = 1
            
        return self.preliminary

    def individual_prediction(self):

        individual_predictions = self.model.predict(self.individual_testing_data)

        if self.prel_var:
            if individual_predictions:
                  self.ind_pred = "Approved"
            else:
                  self.ind_pred = "Disapproved"
            return self.ind_pred
        else:
             return self.preliminary
    
    
overdraft = 0

if overdraft:
    training_data = pd.read_csv('credit_default_training_set.csv')
    new_customer = OrderedDict([('Overdraft_Protection', 1), ('Balance_1st_Mo', 9880),
                                ('Balance_2d_Mo', 1740), ('Expenses_1st_Mo', 44460),
                                ('Expenses_2d_Mo', 17060), ('Deposit_Credit_1st_Mo', 54340),
                                ('Deposit_Credit_2d_Month', 18800),
                                ('Two_Mo_Expenses_Select_Categories', 61520),
                                ('Two_Mo_Income', 73140), ('Two_Mo_DTI_Ratio', .80),
                                ('Limit_Requested', 5400),
                                ('Free_Cash_Flow', 5810), ('Limit_Free_Cash_Ratio', 1.08)])
else:
    training_data = pd.read_csv('credit_default_training_set_no_overdraft.csv')
    new_customer = OrderedDict([('Balance_1st_Mo', 9880), ('Balance_2d_Mo', 1740),
                                ('Expenses_1st_Mo', 44460), ('Expenses_2d_Mo', 17060),
                                ('Deposit_Credit_1st_Mo', 54340), ('Deposit_Credit_2d_Month', 18800),
                                ('Two_Mo_Expenses_Select_Categories', 61520),
                                ('Two_Mo_Income', 73140), ('Two_Mo_DTI_Ratio', .80),
                                ('Limit_Requested', 5400),
                                ('Free_Cash_Flow', 5810), ('Limit_Free_Cash_Ratio', 1.08)])


individual_testing_data = pd.Series(new_customer)

individual_testing_data_features = individual_testing_data

individual_testing_data = individual_testing_data.values.reshape(1, -1)

probability_approval = 0.4

find_features = 1

individualPrediction = IndividualPredictions(training_data, individual_testing_data, individual_testing_data_features,
                                             probability_approval, find_features, overdraft)

preliminary_status = individualPrediction.preliminary_check()

underwriting_prediction = individualPrediction.individual_prediction()

print(underwriting_prediction)



