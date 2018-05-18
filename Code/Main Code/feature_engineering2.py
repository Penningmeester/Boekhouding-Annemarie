import pandas as pd
import numpy as np

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

    return feature_names


def feature_eng(train):

    # deal with NAs in hotels's infor
    train['prop_review_score'].fillna(0, inplace=True)
    
    train["prop_location_score2"].fillna(0, inplace=True)
    
    avg_srch_score = train["srch_query_affinity_score"].mean()
    train["srch_query_affinity_score"].fillna(avg_srch_score, inplace=True)
    train["orig_destination_distance"].fillna(1509,inplace=True)
    train["visitor_hist_adr_usd"].fillna(0, inplace=True)
    train['visitor_hist_starrating_bool'] = pd.notnull(train['visitor_hist_starrating'])
    train['Price_pp_usd'] = train['price_usd'] / (train['srch_adults_count'] + train['srch_children_count'])

    


def main():

    feature_eng(train)
    print('done engineering')

    return train

if __name__=="__main__":
    main()
