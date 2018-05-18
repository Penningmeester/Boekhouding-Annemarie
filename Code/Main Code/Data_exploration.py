"""
User_explorer.py

Purpose:
    Explore habits of visitors
    Assignment 2 of Data Mining course from the Vrije Universiteit Amsterdam

Date:
    2018/05/12

@author: 	Pepijn Meewis
			Tijn Wijdoogen
			Winsen Duker
"""

import Data_importer
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sns.set_style("darkgrid")
sns.set_palette("Set1", n_colors=8, desat=.5)
#==============================================================================
# Explore rates
#==============================================================================
def nanrate(train, valname):
    ## train, the dataset name
    ## valname, the feature name to study
    train[valname+"_na"] = pd.isnull(train[valname])
    book_rate=[]
    click_rate=[]
    c_summary=[]
    b_summary=[]
    cond = []
    for i, gb in train.groupby(valname+"_na"):
        if i:
            cond.append(1)
        else:
            cond.append(0)
        book_rate.append(gb["booking_bool"].mean())
        click_rate.append(gb["click_bool"].mean())
        c_summary.append(gb["click_bool"].describe())
        b_summary.append(gb["booking_bool"].describe())
    df = pd.DataFrame(np.array([cond, click_rate, book_rate]), index=["Condition","Click Rate", "Book Rate"])
    df = df.transpose()
    #print(df)
    df.plot( x="Condition",kind="bar")
    locs, labels = plt.xticks()
    plt.xticks(locs, ["Not NULL", "NULL"], size='small', rotation='horizontal')
    plt.title("Click and booking rate of non-NA samples and Null features, wrt:"+valname)
    plt.show()
    
    
def ratedist(train, name, steps, navalue):
    ## train, dataset name
    ## name, the feature name
    ## steps, the interval size we are going to split the feature data with
    ## navalue, a special value for the na samples, can be -1, or -100
    print("Here the NULL values are classed as "+str(navalue))
    train[name+"_step"] = np.round(train[name]/steps)
    train.loc[pd.isnull(train[name]),'%s_%s'] = navalue %(name,steps)
    rate_list = []
    c_per_list = []
    b_per_list = []
    for rate, gp in train.groupby(name+"_step"):
        rate_list.append(rate)
        c_per_list.append(1.0*gp["click_bool"].mean())
        b_per_list.append(1.0*gp["booking_bool"].mean())
    df = pd.DataFrame(np.array([rate_list, c_per_list, b_per_list]), index=["levels","c_per", "b_per"])
    df = df.transpose()
    df.plot(x="levels", y="c_per", kind="bar")
    plt.title("Click rate against "+name)
    plt.show()
    df.plot(x="levels", y="b_per", kind="bar")
    plt.title("Booking rate against "+name)
    plt.show()

def User_explorer(train): 
    print('User concentration')
    users_per_country = train[['visitor_location_country_id', 'price_usd']].groupby(['visitor_location_country_id'], as_index=False).mean()
    
    vist_clicked = train.loc[train['click_bool'] == 1]
    vist_booked = train.loc[train['booking_bool'] == 1]
    vist_notclicked = train.loc[train['click_bool'] == 0]
    vist_notbooked = train.loc[train['booking_bool'] == 0]
    vist_total_clicked = train["click_bool"].notnull().sum()
    vist_total_booked = train["booking_bool"].notnull().sum()
    
## Investigate whether each booking is clicked first    
    if (len(vist_booked["click_bool"]) - vist_booked["click_bool"].sum()) == 0:
        print("Bookings are clicked first")
    else:
        print("Bookings are not clicked first")    
    
## Investigate Search IDs
    sns.distplot(vist_clicked["srch_id"].dropna(), kde = False)
    sns.distplot(vist_notclicked["srch_id"].dropna(), kde = False)  
    sns.distplot(vist_booked["srch_id"].dropna(), kde = False)
    sns.distplot(vist_notbooked["srch_id"].dropna(), kde = False)  
    # Result: No specific behaviour per search ID

## Investigate site_id (.com, .jp)
    sns.distplot(vist_clicked["site_id"].dropna(), kde = False)
    sns.distplot(vist_notclicked["site_id"].dropna(), kde = False)  
    sns.distplot(vist_booked["site_id"].dropna(), kde = False)
    sns.distplot(vist_notbooked["site_id"].dropna(), kde = False)  
    # Result: No specific behaviour per site ID

## Investigate Visitor Location Country ID 
    sns.distplot(vist_clicked["visitor_location_country_id"].dropna(), kde = False)
    sns.distplot(vist_notclicked["visitor_location_country_id"].dropna(), kde = False)  
    sns.distplot(vist_booked["visitor_location_country_id"].dropna(), kde = False)
    sns.distplot(vist_notbooked["visitor_location_country_id"].dropna(), kde = False)  
    # Result: No specific behaviour per country ID
    
## Investigate visitor_hist_starrating    
    vist_hist_rating = train["visitor_hist_starrating"]
    #Completeness of feature
    print("Percentage missing Visitor_hist_starrating:", vist_hist_rating.isnull().sum()/len(train))
    
    # Starrating of clicked and booked
    sns.distplot(vist_clicked["visitor_hist_starrating"].dropna(), kde = False, label = "Clicked", hist_kws={"histtype":"step", "linewidth": 3}).set_title("Distribution of historical visitor rating if clicked or booked", horizontalalignment = 'center')  
    sns.distplot(vist_booked["visitor_hist_starrating"].dropna(), kde = False, label = "Booked")
    plt.legend()
    plt.show()
    # Statistics
    vist_clicked["visitor_hist_starrating"].describe()
    vist_booked["visitor_hist_starrating"].describe()
    
    # Starrating of not clicked
    sns.distplot(vist_notclicked["visitor_hist_starrating"].dropna(), kde = False, label = "Clicked", hist_kws={"histtype":"step", "linewidth": 3}).set_title("Distribution of historical visitor rating if not clicked", horizontalalignment = 'center')  
    plt.legend()
    plt.show()
    # Statistics
    vist_notclicked["visitor_hist_starrating"].describe()
    # Result: Obviously, no clicked is no booked
    
    # No clue how to properly change function to get seaborn yet
    nanrate(train, "visitor_hist_starrating")
    ratedist(train, "visitor_hist_starrating", 1, -1)
    # Result: Hight of rating doesn't matter as long as there is rated before, booking will be higher. No difference in clicking. 
    # Hence fill all NAs with zeros
    vist_hist_rating = vist_hist_rating.fillna(0, inplace=True)
    
# NEW FEATURE - If rated before, probability to book is higher
    ## Create visitor_hist-starrating_bool
    train['visitor_hist_starrating_bool'] = train["visitor_hist_starrating"].notnull()
    ratedist(train, "visitor_hist_starrating_bool", 1, -1)
    
## Investigate visitor_hist_adr_usd = mean usd spending    
    vist_hist_spend = train["visitor_hist_adr_usd"]
    #Completeness of feature
    print("Percentage missing Visitor_hist_adr_usd:", vist_hist_spend.isnull().sum()/len(train))
    sns.distplot(vist_hist_spend.dropna(), kde = False).set_title("The visitor_hist_adr_usd Distribution", horizontalalignment = 'center')
    
    # Spending if clicked vs no clicked
    sns.distplot(vist_clicked["visitor_hist_adr_usd"].dropna(), kde = False, label = "Clicked", hist_kws={"histtype":"step", "linewidth": 3}).set_title("Historical spending if clicked or booked")
    sns.distplot(vist_booked["visitor_hist_adr_usd"].dropna(), kde = False, label = "Booked")
    plt.legend()
    plt.show()
    # Statistics
    vist_clicked["visitor_hist_adr_usd"].describe()
    vist_booked["visitor_hist_adr_usd"].describe()
    
    # Spending if booked vs no booked
    sns.distplot(vist_notclicked["visitor_hist_adr_usd"].dropna(), kde = False, label = "Clicked", hist_kws={"histtype":"step", "linewidth": 3}).set_title("Historical spending if not clicked")
    plt.legend()
    plt.show()
    # Statistics
    vist_notclicked["visitor_hist_adr_usd"].describe()
    # Not clicked = not booked
    
    # Funtions should be adapted to seaborn
    nanrate(train, "visitor_hist_adr_usd" )
    ratedist(train, "visitor_hist_adr_usd", 50, -1)
    # Result: If booked before, more likely to book again. No differnence on clicking
    # As value of historical booking has no impact, change NAs to zero
    
# NEW FEATURE - If booked before, probability to book is higher
    ## Create visitor_hist_adr_usd_bool
    train['visitor_hist_adr_usd_bool'] = vist_hist_spend.notnull()
    ratedist(train, "visitor_hist_adr_usd_bool", 1, -1)
        
## Investigate Property Review score
    prop_rev_score = train['prop_review_score']
    #Completeness of feature
    print("Percentage missing Prop_review_score:", prop_rev_score.isnull().sum()/len(train))
    
    nanrate(train, "prop_review_score")
    ratedist(train, "prop_review_score", 0.5, -1)
    # Result: If a property is rated, its more likely to be clicked and booked. 
    
    # Statistics
    # To smooth curve, Xu sets 0.0 (no reviews) to 2.5 and NaN to 3.0. 
    prop_rev_score.fillna(3, inplace=True)
    train['prop_review_score'][train['prop_review_score']==0]=2.5
    ratedist(train, "prop_review_score", 0.5, -1)
    # No clue how to do differently. Rating of 3 has higher click prob. No impact on book

## CHECK IF MISSING DATA, ELSE REDUNDANT - Investigate Property Location score1
    prop_location1 = train["prop_location_score1"]
    #Completeness of feature
    print("Percentage missing Prop_location_score1:", prop_location1.isnull().sum()/len(train))

    nanrate(train, "prop_location_score1")
    prop_location1.describe()
    ratedist(train, "prop_location_score1", 0.1, -0.1)
    # Result: As hight of rating impact probability little, replace NA with zeros

## Investigate Property Location score2
    prop_location2 = train["prop_location_score2"]
    #Completeness of feature
    print("Percentage missing Prop_location_score2:", prop_location2.isnull().sum()/len(train))

    nanrate(train, "prop_location_score2")
    prop_location2.describe()
    ratedist(train, "prop_location_score2", 0.1, -0.1)
    # Result: As hight of rating impact probability little, replace NA with zeros

## Investigate Property Log Historical Price - IS THERE REALLY SOMETHING MISSING?
    prop_log_histprice = train["prop_log_historical_price"]
    #Completeness of feature
    print("Percentage missing Prop_log_historical_price:", prop_log_histprice.isnull().sum()/len(train))

    nanrate(train, "prop_log_historical_price")
    prop_log_histprice.describe()
    ratedist(train, "prop_log_historical_price", 0.1, -0.1)
    # Result: Only little missing replace NA with mean
    prop_log_histprice.fillna(prop_log_histprice.mean(),inplace=True)

## Investigate Price_USD
    price_usd = train["price_usd"]
    #Completeness of feature
    print("Percentage missing Price_usd:", price_usd.isnull().sum()/len(train))

## NEW FEATURE - 
#==============================================================================
#     mean_spend_country = train[['prop_country_id', 'price_usd']].groupby(['prop_country_id'], as_index=False).mean()
#     
#     test = train[['prop_id','prop_country_id','price_usd']]
#     if test['price_usd'] > 
#     
#     price_per_property = train[['prop_id', 'price_usd']].groupby(['prop_id'], as_index=False)
#     for i in range(0,len(mean_spend_country)):
#         check = train['prop_country_id'] == mean_spend_country['prop_country_id'][i]
#         if check['price_usd'] > mean_spend_country['price_usd'][i]:
#             train['new'] = 1
#         else:
#             train['new'] = 0
# 
#==============================================================================

## Investigate Distance user and location
    orig_dest_dist = train["orig_destination_distance"]
    nanrate(train, "orig_destination_distance")
    orig_dest_dist.describe()
    # No relation w.r.t. booking or clicking
    # Take log
    train["orig_destination_distance_log"] = np.log(orig_dest_dist)
    train["orig_destination_distance_log"].describe()
    ratedist(train, "orig_destination_distance_log", 1, -6)
    # Replace NAs with median
    orig_dest_dist.fillna(orig_dest_dist.median(),inplace=True)
    train["orig_destination_distance_log"] = np.log(train["orig_destination_distance"])
    ratedist(train, "orig_destination_distance_log", 1, -6)
    # Result: No relation w.r.t. click/book, only filled NAs with median
    
## Investigate Probability to be clicked feature
    srch_query_score = train["srch_query_affinity_score"]
    #Completeness of feature
    print("Percentage missing Srch_query_affinity_score:", srch_query_score.isnull().sum()/len(train))

    nanrate(train, "srch_query_affinity_score")
    srch_query_score.describe()
    # High click probability provides higher probability of being books
    train["srch_query_affinity_score_log"] = np.log(-srch_query_score)
    train["srch_query_affinity_score_log"].describe()
    ratedist(train, "srch_query_affinity_score_log", 0.5, 0)
    # Result: Level of score slightly higher for bookings, can be disregarded
    # Hence replace NAs with mean value as no difference in click/book is seen
    srch_query_score.fillna(srch_query_score.mean(),inplace=True)
    
## Investigate Competitor info
    nanrate(train, "comp5_rate")
    print(train["comp5_rate"].describe())
    ratedist(train, "comp5_rate", 1, -2)
    
    nanrate(train, "comp5_inv")
    print(train["comp5_inv"].describe())
    ratedist(train, "comp5_inv", 1, -2)
    
    ratedist(train, "comp2_inv", 1, -2)
    ratedist(train, "comp8_inv", 1, -2)
    ratedist(train, "comp3_inv", 1, -2)
    
    # Result: No difference seen between click/book among competitors, hence combine data of all competitors in one feature
    # Rate_percent_diff therefore has no added value, remove 
    # Comp_rate does influences click and booking, comp_inv is negatively w.r.t. book/click
    # Add comp_i_rate and inverse to get comp_sum to get feature comp_sum
    # All NAs to zero

# NEW FEATURES: Comp_sum, Comp_inv_sum and Comp_rate_sum 
    train['comp_sum'] = train['comp1_rate'] + train['comp1_inv'] + 2
    for i in range(2,9):
        train['comp_sum'] += train['comp'+str(i)+'_rate'] + train['comp'+str(i)+'_inv']    
    train["comp_sum"].describe()    
    ratedist(train, "comp_sum", 1, -13)

    for i in range(1,9):
        ratedist(train, 'comp'+str(i)+'_rate', 1, -2)

    for i in range(1,9):
        train['comp'+str(i)+'_inv'].fillna(0, inplace=True)
        train['comp'+str(i)+'_inv'][train['comp'+str(i)+'_inv']==1] = 2
        train['comp'+str(i)+'_inv'][train['comp'+str(i)+'_inv']==-1] = 1
        train['comp'+str(i)+'_inv'][train['comp'+str(i)+'_inv']==0] = -1
        train['comp'+str(i)+'_inv'][train['comp'+str(i)+'_inv']==2] = 0
        train['comp_inv_sum'] = train['comp1_inv']
        for i in range(2,9):
            train['comp_inv_sum'] += train['comp'+str(i)+'_inv']
        train['comp_inv_sum'].describe()
            
        ratedist(train, "comp_inv_sum", 1, -9)

    for i in range(1,9):
        train['comp'+str(i)+'_rate'].fillna(0, inplace=True)
    train['comp_rate_sum'] = train['comp1_rate']
    for i in range(2,9):
        train['comp_rate_sum'] += train['comp'+str(i)+'_rate']
    train['comp_rate_sum'].describe()
    
    ratedist(train, "comp_rate_sum", 1, -7)

    # Result: Combine all comp features to comp_inv_sum and comp_rate_sum
    # Drop all individual comp features
    for i in range(1,9):
        train.drop(columns=['comp'+str(i)+'_inv'])
        # DOESNT WORK PROPERLY. REALLY NEEDED TO BE REMOVED?
        
    print('Data is explored')



def main():
    print('Finished code')

###########################################################
### start main
if __name__ == "__main__":
    main()
