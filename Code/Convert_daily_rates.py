# -*- coding: utf-8 -*-
"""
Created on Sun May 13 22:07:07 2018

@author: tijnw
"""

#Purpose is to check wether price is daily or weekly

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


def Convert_daily_rates(train):
    Price_set = train[['srch_id', 'price_usd', 'srch_adults_count', 'srch_children_count']]
    Price_pp_total = ToPP(train)
    
    suspect1 = train[train['prop_country_id'] == 219]
    Prices_1 = ToPP(suspect1)
    return Prices_1

def ToPP(price_set):
    Total_persons = price_set['price_usd'] / (price_set['srch_adults_count'] + price_set['srch_children_count'])
    #StayDays = price_set['srch_length_of_stay']
    #Prices = price_set['price_usd']
    #Prices = list(Prices)
    #Price_pp = np.zeros(len(Total_persons))
    #for i in range(0,len(Total_persons)):
        #Price_pp[i] = Prices[i] / Total_persons[i] 
    return Total_persons

def main():
    
    Price_pp = Convert_daily_rates(train)
    max_val = Price_pp.argsort()[-1000000:][::-1]
    max_countries = np.zeros(len(max_val))
    Countries = list(train['prop_country_id'])
    for i in range(0, len(max_val)):
        max_countries[i] = Countries[max_val[i]]
    
    cnt = Counter(max_countries)
    print(cnt.most_common(10))
    
   
        
    
    
    
    #plt.hist(Price_pp, bins = 100)
    #plt.show()
    #plt.hist(StayDays)
    #plt.show()
    
    print('Finished code')
###########################################################
### start main
if __name__ == "__main__":
    main()