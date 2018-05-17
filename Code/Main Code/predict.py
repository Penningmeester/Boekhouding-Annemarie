import Data_importer
import feature_engineering2 as feature_engineering


def main():
    test = Data_importer.load_test_set()
    test = feature_engineering.main(test)
    
    #predict if customer clicks
    print('Loading Click model')
    model = Data_importer.load_model(False)
    print('Predicting probability of Click')
    
    click_names = feature_engineering.get_features(test, False)
    test_click =  test[click_names]
    click_predict = model.predict_proba(test_click.values)[:,1]
    click_predict = list(click_predict*-1)
    print('Predicted Clicking probability')
    
    print('Loading Book model')
    model = Data_importer.load_model(True)
    print('Predicting probability of Booking')\
    
    book_names = feature_engineering.get_features(test, True)
    test_book =  test[book_names]
    book_predict = model.predict_proba(test_book.values)[:,1]
    book_predict = list(book_predict*-1)
    print('Predicted the Booking probability')
    
    #Wrinting the predictions
    predictions = zip(test["srch_id"], test["prop_id"], 5*book_predict+click_predict)
    print("Writing predictions to file..")
    Data_importer.write_submission(predictions)
    print("wrote predictions to file")
    
if __name__=="__main__":
    main()