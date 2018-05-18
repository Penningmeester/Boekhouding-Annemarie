from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
import Data_importer
from sklearn import model_selection

def model(x_train, x_test, y_train, y_test, isBook):
    # classifier =  RandomForestClassifier(n_estimators=50, 
                                            # verbose=2,
                                            # n_jobs=1,
                                            # min_samples_split=10,
                                            # random_state=1)
    # classifier.fit(x_train, y_train)
                                            
    # classifier =  GradientBoostingClassifier(loss='deviance', 
    #                                     learning_rate=0.1, 
    #                                     n_estimators=100, 
    #                                     subsample=1.0, 
    #                                     min_samples_split=2, 
    #                                     min_samples_leaf=1, 
    #                                     max_depth=3, 
    #                                     init=None, 
    #                                     random_state=None, 
    #                                     max_features=None, 
    #                                     verbose=0)
    # classifier.fit(x_train, y_train)


    # lightgbm approach
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)
    params = {}
    params['learning_rate'] = 0.003
    params['boosting_type'] = 'gbdt'
    params['objective'] = 'binary'
    params['metric'] = 'binary_logloss'
    params['sub_feature'] = 0.5
    params['num_leaves'] = 10
    params['min_data'] = 50
    params['max_depth'] = 10
    classifier = lgb.train(params, lgb_train, num_boost_round=1000, 
                    verbose_eval=20, early_stopping_rounds=40, valid_sets=lgb_eval)

        
    # print the time interval
    print("Saving the classifier...")
    Data_importer.dump_model(classifier, isBook)
    print('classifier saved')
