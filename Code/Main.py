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
###########################################################
### Imports
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import scipy.optimize as opt
# from pandas_datareader import data as web

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
    # Magic numbers
    #train,test  = importdatabase()

    # Initialisation
    # Output
    print('Finished code')

###########################################################
### start main
if __name__ == "__main__":
    main()
