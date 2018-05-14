# -*- coding: utf-8 -*-
"""
Created on Sun May 13 22:07:07 2018

@author: tijnw
"""

#Purpose is to check wether price is daily or weekly

import numpy as np
import pandas as pd

def Convert_daily_rates(train):
    Price_set = train[['srch_id', 'price_usd', 'srch_adults_count', 'srch_children_count']]
    Total_persons = list(train['srch_adults_count'] + train['srch_children_count'])
    Prices = train['price_usd']
    Prices = list(Prices)
    Price_pp = np.zeros(len(Total_persons))
    for i in range(0,len(Total_persons)):
        Price_pp[i] = Prices[i] / Total_persons[i] 
    return Price_set, Price_pp

def main():
    print('Finished code')
    Price_set, Prices_pp = Convert_daily_rates(train)
   
###########################################################
### start main
if __name__ == "__main__":
    main()