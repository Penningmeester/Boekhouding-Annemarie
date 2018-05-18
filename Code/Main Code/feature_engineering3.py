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
    feature_names.remove("price_usd")


    if isBook:
        feature_names.append("visitor_hist_starrating_bool")
    feature_names.append("comp_rate_sum")
    feature_names.append("comp_inv_sum")
    feature_names.append("Price_pp_usd")
    feature_names.append('Month')
    feature_names.append('Day')

    return feature_names


def feature_eng(train):
    # deal with NAs in hotels's infor
    train['prop_review_score'].fillna(0, inplace=True)
        
    train['Price_pp_usd'] = train['price_usd'] / (train['srch_adults_count'] + train['srch_children_count'])

    # Winsen
    train['visitor_hist_starrating'] = train['visitor_hist_starrating'].fillna(0, inplace=True)
    train['visitor_hist_starrating_bool'] = train["visitor_hist_starrating"].notnull()
    train["visitor_hist_adr_usd"].fillna(0, inplace=True)
    train['visitor_hist_adr_usd_bool'] = train['visitor_hist_adr_usd'].notnull()
    train['prop_review_score']=train['prop_review_score'].fillna(3, inplace=True)
    train.loc[train['prop_review_score']==0,'prop_review_score']=2.5
    train["prop_location_score2"]=train["prop_location_score2"].fillna(0, inplace=True)
    train['prop_location_score1']=train['prop_location_score1'].fillna(0, inplace=True)
    train['prop_log_historical_price']=train['prop_log_historical_price'].fillna(train['prop_log_historical_price'].mean(),inplace=True)
    train["orig_destination_distance_log"] = np.log(train.orig_destination_distance)
    train['srch_query_affinity_score']=train['srch_query_affinity_score'].fillna(train.srch_query_affinity_score.mean(),inplace=True)


    train['comp_sum'] = train['comp1_rate'] + train['comp1_inv'] + 2
    for i in range(2,9):
        train['comp_sum'] += train['comp'+str(i)+'_rate'] + train['comp'+str(i)+'_inv']    

    for i in range(1,9):
        train['comp'+str(i)+'_inv'].fillna(0, inplace=True)
        train['comp'+str(i)+'_inv'][train['comp'+str(i)+'_inv']==1] = 2
        train['comp'+str(i)+'_inv'][train['comp'+str(i)+'_inv']==-1] = 1
        train['comp'+str(i)+'_inv'][train['comp'+str(i)+'_inv']==0] = -1
        train['comp'+str(i)+'_inv'][train['comp'+str(i)+'_inv']==2] = 0
        train['comp_inv_sum'] = train['comp1_inv']
        for i in range(2,9):
            train['comp_inv_sum'] += train['comp'+str(i)+'_inv']
            

    for i in range(1,9):
        train['comp'+str(i)+'_rate'].fillna(0, inplace=True)
    train['comp_rate_sum'] = train['comp1_rate']
    for i in range(2,9):
        train['comp_rate_sum'] += train['comp'+str(i)+'_rate']
    train['date_time'] = pd.to_datetime(train['date_time'])
    train['Month'] = train.date_time.dt.month
    train['Day'] = train.date_time.dt.weekday


def main(train):

    feature_eng(train)
    print('done engineering')

    return train

if __name__=="__main__":
    main()
