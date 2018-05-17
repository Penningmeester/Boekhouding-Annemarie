def Data_exploration(train):
    print('Mean conversion rate click and booking per site ID')
    print(train[['site_id', 'click_bool']].groupby(['site_id'], as_index=False).mean())
    print('\n',train[['site_id', 'booking_bool']].groupby(['site_id'], as_index=False).mean())
    print('\nData is explored')


















    
def main():
    print('Finished code')

###########################################################
### start main
if __name__ == "__main__":
    main()