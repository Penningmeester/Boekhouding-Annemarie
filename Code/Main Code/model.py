import lightgbm as lgb
import Data_importer
from sklearn.ensemble import RandomForestClassifier
import xgboost as xg
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def model(x_train, x_test, y_train, y_test, isBook):

    # Random Forest 0.39132 score
    # classifier =  RandomForestClassifier(n_estimators=50, 
    #                                         verbose=2,
    #                                         n_jobs=1,
    #                                         min_samples_split=10,
    #                                         random_state=1)
    # classifier = classifier.fit(x_train, y_train)
 	

 	#SVC 0.35432

    # classifier = LinearSVC()
    # classifier = classifier.fit(x_train, y_train)

    # #MLP 0.35592
    # classifier = MLPClassifier(hidden_layer_sizes=(27,4,15),
    # 							early_stopping=True,
    # 							max_iter=2000)
    # classifier = classifier.fit(x_train, y_train)


    # #Kneighbors
    # classifier = KNeighborsClassifier(n_neighbors = 7)
    # classifier = classifier.fit(x_train, y_train)


    
    # #XGboost 0.41648 score
    # classifier = xg.XGBClassifier(
    #                 max_depth=7,
    #                 learning_rate=0.032,
    #                 n_estimators = 1000,
    #                 objective = 'binary:logistic',
    #                 metric = 'logloss',
    #                 silent=1,
    #                 early_stopping_rounds=20,
    #                 verbose_eval=20,)

    # classifier = classifier.fit(x_train, y_train, eval_set=[(x_train,y_train),(x_test,y_test)])

    # lightgbm approach
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)
    params = {}
    params['boosting_type']= 'gbdt'
    params['objective']= 'binary'
    params['metric']= 'binary_logloss'
    params['num_leaves']= 100
    params['learning_rate']= 0.032
    params['feature_fraction']= 0.9
    params['bagging_fraction']= .9
    params['bagging_freq']= 70
    
    classifier = lgb.train(params,lgb_train, 
                            verbose_eval=20, 
                            early_stopping_rounds=40, 
                            valid_sets=lgb_eval, 
                            num_boost_round=2000)

    # print the time interval
    print("Saving the classifier...")
    Data_importer.dump_model(classifier, isBook)
    print('classifier saved')

if __name__=="__main__":
    model(x_train, x_test, y_train, y_test, isBook)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    