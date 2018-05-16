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
    train[name+"_step"][pd.isnull(train[name])] = navalue
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
    
    
## Investigate visitor_hist_starrating    
    vist_hist_rating = train["visitor_hist_starrating"]
    print(vist_hist_rating.describe())
    vist_hist_rating.hist()
    plt.title("Distribution of visitor_hist_starrating")
    # For seaborn plot, not sure if similar
    vist_na = vist_hist_rate.dropna()
    sns.distplot(vist_na, kde=False).set_title("Distribution of visitor_hist_starrating", horizontalalignment='center')
    #Completeness of feature
    x = pd.isnull(vist_hist_rating).sum()
    y = train.shape[0]
    print(x/(1.0*y))
    
    
    # Starrating of clicked
    sns.distplot(vist_clicked["visitor_hist_starrating"].dropna(), kde = False).set_title("Distribution of historical visitor rating if clicked", horizontalalignment = 'center')  
    sns.distplot(vist_notclicked["visitor_hist_starrating"].dropna(), kde = False).set_title("Distribution of historical visitor rating if not clicked", horizontalalignment = 'center')  
    # Statistics of clicked
    vist_clicked["visitor_hist_adr_usd"].describe()
    vist_notclicked["visitor_hist_adr_usd"].describe()
    
    # Starrating of booked
    sns.distplot(vist_booked["visitor_hist_starrating"].dropna(), kde = False).set_title("Distribution of historical visitor rating if booked", horizontalalignment = 'center')
    sns.distplot(vist_notbooked["visitor_hist_starrating"].dropna(), kde = False).set_title("Distribution of historical visitor rating if not booked", horizontalalignment = 'center')
    # Statistics of booked
    vist_booked["visitor_hist_adr_usd"].describe()
    vist_notbooked["visitor_hist_adr_usd"].describe()
    
    # No clue how to properly change function to get seaborn yet
    nanrate(train, "visitor_hist_starrating")
    ratedist(train, "visitor_hist_starrating", 1, -1)
    # As starrating doesn't influence click/book, one boolean if rated historically
    # Create visitor_hist-starrating_bool
    train['visitor_hist_starrating_bool'] = pd.notnull(train["visitor_hist_starrating"])
    ratedist(train, "visitor_hist_starrating_bool", 1, -1)
    # Result: hist rating only influence booking behaviour
  
## Investigate visitor_hist_adr_usd = mean usd spending    
    vist_hist_spend = train["visitor_hist_adr_usd"]
    sns.distplot(vist_hist_spend.dropna(), kde = False).set_title("The visitor_hist_adr_usd Distribution", horizontalalignment = 'center')
    #Completeness of feature
    x = pd.isnull(vist_hist_spend.sum()
    y = train.shape[0]
    print(x/(1.0*y)) 
    
    # Spending if clicked vs no clicked
    sns.distplot(vist_clicked["visitor_hist_adr_usd"].dropna(), kde = False).set_title("Historical spending if clicked")
    sns.distplot(vist_notclicked["visitor_hist_adr_usd"].dropna(), kde = False).set_title("Historical spending if not clicked")
    # Statistics
    vist_clicked["visitor_hist_adr_usd"].describe()
    vist_notclicked["visitor_hist_adr_usd"].describe()
    
    # Spending if booked vs no booked
    sns.distplot(vist_booked["visitor_hist_adr_usd"].dropna(), kde = False).set_title("Historical spending if booked")
    sns.distplot(vist_notbooked["visitor_hist_adr_usd"].dropna(), kde = False).set_title("Historical spending if not booked")
    # Perhaps scale and put plots over each other to see difference in distribution?   
    # Statistics
    vist_booked["visitor_hist_adr_usd"].describe()
    vist_notbooked["visitor_hist_adr_usd"].describe()
    
    # Funtions should be adapted to seaborn
    nanrate(train, "visitor_hist_adr_usd" )
    ratedist(train, "visitor_hist_adr_usd", 50, -1)
    # Result: If booked before, more likely to book again. No differnence on clicking
    # As value of historical booking has no impact, change NaNs to zeros
    


    
    print('Data is explored')



def main():
    print('Finished code')

###########################################################
### start main
if __name__ == "__main__":
    main()
