"""
Rate_explorer.py

Purpose:
    Explore daily vs full stay rates
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

def Rate_explorer(train): 
    print('Rates per visitor location')
    mean_price_country = train[['visitor_location_country_id', 'price_usd']].groupby(['visitor_location_country_id'], as_index=False).mean()
    mean_stay_country = train[['visitor_location_country_id', 'srch_length_of_stay']].groupby(['visitor_location_country_id'], as_index=False).mean()
    mean_stay_price_country = train[['visitor_location_country_id', 'price_usd','srch_length_of_stay']].groupby(['visitor_location_country_id'], as_index=False).mean()
    
    # Cannot say much only on mean price. Perhaps check median as well
    
    
    
    print('Data is explored')



def main():
    print('Finished code')

###########################################################
### start main
if __name__ == "__main__":
    main()
