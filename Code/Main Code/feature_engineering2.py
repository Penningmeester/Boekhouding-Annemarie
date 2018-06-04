import pandas as pd
import numpy as np
from datetime import datetime as dt

def get_features(train, isBook=True):
    feature_names = list(train.columns)[:27]

    if "comp1_rate" in feature_names:
        # only true in the test set
        feature_names.remove("comp1_rate")
    if "position" in feature_names:
        # only true in the training set
        feature_names.remove("position")

    feature_names.remove("date_time")
    feature_names.remove("srch_id")
    feature_names.remove("visitor_hist_starrating")
    feature_names.remove("visitor_hist_adr_usd") #new
    if isBook:
        feature_names.append("bool_visitor_hist_starrating")
        feature_names.append("bool_visitor_hist_adr_usd") #new
    feature_names.append("sum_comp_rate")
    feature_names.append("sum_comp_inv")
    return feature_names

def feature_eng(train):

    # deal with NAs in hotels's infor
    train['prop_review_score'].fillna(train['prop_review_score'].median(), inplace=True)
    train.loc[train['prop_review_score']==0,'prop_review_score']=train.prop_review_score.median()
    train["prop_location_score2"].fillna(0, inplace=True)
    train["srch_query_affinity_score"].fillna(train["srch_query_affinity_score"].mean(), inplace=True)
    train["orig_destination_distance"].fillna(train["orig_destination_distance"].mean(),inplace=True) 
    #train["visitor_hist_adr_usd"].fillna(0, inplace=True)
    train['bool_visitor_hist_adr_usd'] = train['visitor_hist_starrating'].notnull() #new
    train['bool_visitor_hist_starrating'] = train['visitor_hist_starrating'].notnull()

    # add feature: sum_comp_rate
    for i in range(1,9):
        train['comp'+str(i)+'_rate'].fillna(0, inplace=True)
    train['sum_comp_rate'] = train['comp1_rate']
    for i in range(2,9):
        train['sum_comp_rate'] += train['comp'+str(i)+'_rate']

    # add feature: sum_comp_rate
    for i in range(1,9):
        train['comp'+str(i)+'_inv'].fillna(0, inplace=True)
    train['sum_comp_inv'] = train['comp1_inv']
    for i in range(2,9):
        train['sum_comp_inv'] += train['comp'+str(i)+'_inv']

def main(train):
    print('Starting with engineering features')
    feature_eng(train)
    print('Done engineering')

    return train

if __name__=="__main__":
    main()