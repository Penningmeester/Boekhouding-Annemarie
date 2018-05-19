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
    if isBook:
        feature_names.append("visitor_hist_starrating_bool")
    feature_names.append("comp_rate_sum")
    feature_names.append("comp_inv_sum")
    return feature_names


def feature_eng(train):

    # deal with NAs in hotels's infor
    train['prop_review_score'].fillna(3, inplace=True)
    train['prop_review_score'][train['prop_review_score']==0]=train.prop_review_score.mean()
    train["prop_location_score2"].fillna(0, inplace=True)
    avg_srch_score = train["srch_query_affinity_score"].mean()
    train["srch_query_affinity_score"].fillna(avg_srch_score, inplace=True)
    train["orig_destination_distance"].fillna(1509,inplace=True)
    train["visitor_hist_adr_usd"].fillna(0, inplace=True)
    train['visitor_hist_starrating_bool'] = pd.notnull(train['visitor_hist_starrating'])

    # add feature: comp_rate_sum
    for i in range(1,9):
        train['comp'+str(i)+'_rate'].fillna(0, inplace=True)
    train['comp_rate_sum'] = train['comp1_rate']
    for i in range(2,9):
        train['comp_rate_sum'] += train['comp'+str(i)+'_rate']

    # add feature: comp_rate_sum
    for i in range(1,9):
        train['comp'+str(i)+'_inv'].fillna(0, inplace=True)
        train['comp'+str(i)+'_inv'][train['comp'+str(i)+'_inv']==1] = 10
        train['comp'+str(i)+'_inv'][train['comp'+str(i)+'_inv']==-1] = 1
        train['comp'+str(i)+'_inv'][train['comp'+str(i)+'_inv']==0] = -1
        train['comp'+str(i)+'_inv'][train['comp'+str(i)+'_inv']==10] = 0
    train['comp_inv_sum'] = train['comp1_inv']
    for i in range(2,9):
        train['comp_inv_sum'] += train['comp'+str(i)+'_inv']

def main(train):

    feature_eng(train)
    print('done engineering')

    return train

if __name__=="__main__":
    main()