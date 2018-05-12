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

def User_explorer(train): 
    print('User concentration')
    users_per_country = train[['visitor_location_country_id', 'price_usd']].groupby(['visitor_location_country_id'], as_index=False).mean()
    
    # Countif for each country
    
    
    
    print('Data is explored')



def main():
    print('Finished code')

###########################################################
### start main
if __name__ == "__main__":
    main()
