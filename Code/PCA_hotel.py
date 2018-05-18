# -*- coding: utf-8 -*-
"""
Created on Wed May 16 15:48:07 2018

@author: tijnw
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def main():

    features = "site_id	visitor_location_country_id	visitor_hist_starrating	visitor_hist_adr_usd	prop_country_id	prop_id	prop_starrating	prop_review_score	prop_brand_bool	prop_location_score1	prop_location_score2	prop_log_historical_price	position	price_usd	promotion_flag	srch_destination_id	srch_length_of_stay	srch_booking_window	srch_adults_count	srch_children_count	srch_room_count	srch_saturday_night_bool	srch_query_affinity_score	orig_destination_distance	random_bool"
    features = features.split()
    x = train[features]
    evenanders = x.dropna()
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    x = imp.fit_transform(x)
    x = StandardScaler().fit_transform(x)
    
    
    print(evenanders.shape)
    evenanders = StandardScaler().fit_transform(evenanders)
    
    pca = PCA(n_components=10)
    principalComponents = pca.fit_transform(evenanders)
    principalDf = pd.DataFrame(data = principalComponents)
    scree = pca.explained_variance_ratio_
    print(scree)
if __name__ == "__main__":
    main()