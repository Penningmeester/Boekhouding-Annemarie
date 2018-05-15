"""
Data_explorer.py

Purpose:
    Explore characteristics of dataset
    Assignment 2 of Data Mining course from the Vrije Universiteit Amsterdam

Date:
    2018/05/15

@author: 	Pepijn Meewis
			Tijn Wijdoogen
			Winsen Duker
"""
#==============================================================================
# Explore characteristics of dataset
#==============================================================================

def Data_explorer(train): 
    features = list(train.columns)
    nan_count = train.isnull().sum()
    nan_rate = nan_count \ len(train)
    complete_rate = 1 - nan_rate
    
    sns.barplot(x = nan_rate, y = features)
    sns.barplot(x = complete_rate, y = features)
    
    # Check completeness of features if clicked
    click_data = train.loc[train['click_bool'] == 1]
    nan_count_clicked = click_data.isnull().sum()
    nan_rate_clicked = nan_count_clicked \ len(click_data)
    complete_rate_clicked = 1 - nan_rate_clicked
    
    # Check completeness of features if booked
    book_data = train.loc[train['booking_bool'] == 1]
    nan_count_booked = book_data.isnull().sum()
    nan_rate_booked = nan_count_booked \ len(book_data)
    complete_rate_booked = 1 - nan_rate_booked
    
    
    print('Data is explored')



def main():
    print('Finished code')

###########################################################
### start main
if __name__ == "__main__":
    main()
