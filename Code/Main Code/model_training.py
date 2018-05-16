import sys
import Data_importer
import model
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier




def feature_engineering():
	print('done engineering')


def main():
    train = Data_importer.load_train_set()
    feature_engineering(train)
    train_set = train.copy()

    book_trainset = train_set[train_set['booking_bool']==1] # extract all bookings from training set
    book_rows = book_trainset.index.tolist()
    len_book = len(book_trainset.index)
    
    click_trainset = train_set[train_set['click_bool']==1] # extract all clicks from training set
    click_rows = click_trainset.index.tolist()
    len_click = len(click_trainset.index)
    
    # create two training sets of just 50% booking and random and 50% click and random
    book_trainset = book_trainset.append(train_set.iloc[random.sample(list(train_set.drop(book_rows).index), len_book)])
    click_trainset =click_trainset.append(train_set.iloc[random.sample(list(train_set.drop(click_rows).index), len_click)])
    # Train the booking model
    
    for i in range(0,2):
        if i==0:
            model_name = "Booking"
            training_feature = "booking_bool"
            train_sample = book_trainset
            isBook = True
        else:
            model_name = "Click"
            training_feature = "click_bool"
            train_sample = click_trainset
            isBook = False
        
        print("Training the "+model_name+" Classifier...")
        feature_names = get_features(train_sample, isBook)
        features = train_sample[feature_names].values
        target = train_sample[training_feature].values
        classifier = model.model()
        classifier.fit(features, target)
        
        # print the time interval
        print("Saving the classifier...")
        Data_importer.save_model(classifier, isBook)

if __name__=="__main__":
    main()