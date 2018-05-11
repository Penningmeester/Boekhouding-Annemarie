#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main.py

Purpose:
    Assignment 2 of Data Mining course from the Vrije Universiteit Amsterdam

Date:
    2018/05/09

@author: 	Pepijn Meewis
			Tijn Wijdoogen
			Winsen Duker
"""
#==============================================================================
# #import own code modules
#==============================================================================
from Data_exploration import *



###########################################################
### Imports
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import scipy.optimize as opt
# from pandas_datareader import data as web

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

#==============================================================================
# Import data
#==============================================================================
def importdatabase():
    
    df_train = pd.read_hdf("Data/train.hdf")
    df_test = pd.read_hdf("Data/test.hdf")
    
    df_test['date_time'] = pd.to_datetime(df_test['date_time'])
    df_train['date_time'] = pd.to_datetime(df_train['date_time'])
    
    return df_test, df_train



###########################################################
### main
def main():
    # Import data
    #test,train  = importdatabase()
    
    Data_exploration(train)
  
    print('\nFinished code')

###########################################################
### start main
if __name__ == "__main__":
    main()
