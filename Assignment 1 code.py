#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
empty.py

Purpose:
    ...

Version:
    1       First start

Date:
    2017/**/**

@author: pms590
"""
###########################################################
### Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import scipy.optimize as opt
import os
import seaborn as sns
#print(os.listdir("Input"))

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier




#==============================================================================
#     #import data
#===============================================================================
def importdata():
    train = pd.read_csv("Input/train.csv")
    test = pd.read_csv("Input/test.csv")
    
    return train,test


#==============================================================================
#     #exploratory statistics
#==============================================================================
def exploringdata():
    (train,test) = importdata()
    
    print(pd.concat([train,test]).info())
    
    print(train.dropna().describe())
    # get info on training set. Observe that total entries are 891. Most categories have no NaN. "Age", "Cabin", and 
    # "Embarked" have NaN in dataset 
    print(train.sample(8))
    # observe that Name Ticket and Fare are probably no good classifiers as is.
    print()
    print('Percentage of people who survived',np.round(train['Survived'].mean(),4)*100,'%')
    print()
    print('\nTicket class')
    print (train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
    print()
    print('\nSex')
    print (train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean())
    print()
    print('\n # of siblings / spouses aboard the Titanic')
    print (train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean())
    print()
    print('\n # of parents / children aboard the Titanic')
    print (train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean())
    print(train[['PassengerId','Embarked']].groupby(['Embarked']).count())
    print('\nNumber of missing observation is Embarked categorie:',train['Embarked'].isnull().sum())


#==============================================================================
#     #graphics
#==============================================================================
def graphics():
    
    (train,test) = importdata()
    surv = train[train['Survived']==1]
    nosurv = train[train['Survived']==0]
    surv_col = "green"
    nosurv_col = "red"
    print("Graphical presentation of data")
    plt.figure(figsize=[19,6])
#==============================================================================
#     plt.subplot(231)
#     sns.distplot(surv['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color=surv_col)
#     sns.distplot(nosurv['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color=nosurv_col,
#                 axlabel='Age')
#==============================================================================
    plt.subplot(243)
    sns.barplot('Sex', 'Survived', data=train)
    plt.subplot(242)
    sns.barplot('Pclass', 'Survived', data=train)
    plt.subplot(241)
    sns.barplot('Embarked', 'Survived', data=train)
    plt.subplot(245)
    train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
    sns.barplot('FamilySize', 'Survived', data=train)
    
#==============================================================================
#     sns.distplot(np.log(surv['Fare'].dropna().values+1), kde=False, color=surv_col)
#     sns.distplot(np.log(nosurv['Fare'].dropna().values+1), kde=False, color=nosurv_col,axlabel='Fare')
#==============================================================================
    
    dataset_title = [i.split(",")[1].split(".")[0].strip() for i in train["Name"]]
    train["Title"] = pd.Series(dataset_title)
    train["Title"] = train["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    train["Title"] = train["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
    train["Title"] = train["Title"].astype(int)
    
    plt.subplot(248)
    sns.heatmap(train[["Survived","FamilySize","Age","Fare"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                        wspace=0.35)
    plt.subplot(244) 
    sns.barplot("Title","Survived",data=train).set_xticklabels(["Master","Miss-Mrs","Mr","Rare"])
        
    train.loc[ train['Age'] <= 16, 'Age'] = 0
    train.loc[(train['Age'] > 16) & (train['Age'] <= 32), 'Age'] = 1
    train.loc[(train['Age'] > 32) & (train['Age'] <= 48), 'Age'] = 2
    train.loc[(train['Age'] > 48) & (train['Age'] <= 64), 'Age'] = 3
    train.loc[ train['Age'] > 64, 'Age'] = 4
    plt.subplot(246)
    sns.barplot('Age', 'Survived', data=train)
    
    train.loc[ train['Fare'] <= 102, 'Fare'] = 0
    train.loc[(train['Fare'] > 102) & (train['Fare'] <= 205), 'Fare'] = 1
    train.loc[(train['Fare'] > 205) & (train['Fare'] <= 307), 'Fare'] = 2
    train.loc[(train['Fare'] > 307) & (train['Fare'] <= 410), 'Fare'] = 3
    train.loc[ train['Fare'] > 410, 'Fare'] = 4
    plt.subplot(247)
    sns.barplot('Fare', 'Survived', data=train)
    
    plt.savefig('figures.png', dpi = 300)
    plt.show()
    
    print("Mean age survivors: %.1f, Mean age non-survivers: %.1f"\
      %(np.mean(surv['Age'].dropna()), np.mean(nosurv['Age'].dropna())))
    
    
#==============================================================================
#     #changing data
#==============================================================================
def featureextraction(print_):
    
    # Amount of people per embarkment location
    
    (train,test) = importdata()
    
    full_data = pd.concat([train,test])
    
  
    
    #fill in missing observations with most observed S
    full_data['Embarked'] = full_data['Embarked'].replace(np.nan, "S", regex=True)
    if print_==True:
        print("changed missing data to S")
    
    full_data.loc[full_data['Embarked']=='C','Embarked']=0
    full_data.loc[full_data['Embarked']=='Q','Embarked']=1
    full_data.loc[full_data['Embarked']=='S','Embarked']=2
    if print_==True:
        print('\nchanged C,Q,S to 0,1,2')
    
    if print_==True:
        print('\nchanging Age feature')
    mean = np.nanmean(train['Age'])
    st_dev = np.nanstd(train['Age'])
    if print_==True:
        print('mean: ',np.round(mean,3), '\nSt_dev: ',np.round(st_dev,3))
    missing_obs = full_data['Age'].isnull().sum()
    
    if print_==True:
        print('filling in missing data with mean and st_dev')
    random_ages = np.round(np.random.normal(mean,st_dev,missing_obs)).astype(int)
    if print_==True:
        print(random_ages[random_ages<=0]) 
    random_ages[random_ages<=0] = np.random.uniform(len(random_ages[random_ages<=0]))
    if print_==True:
        print(random_ages)
        
    if print_==True:
        sns.distplot(random_ages)
        plt.show()
    
    full_data.loc[np.isnan(full_data['Age']),'Age'] = random_ages
    
    
    train['AgeBand'] = pd.cut(train['Age'], 5, precision = 0)
    if print_==True:
        print(train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True))
     
    
    full_data.loc[ full_data['Age'] <= 16, 'Age'] = 0
    full_data.loc[(full_data['Age'] > 16) & (full_data['Age'] <= 32), 'Age'] = 1
    full_data.loc[(full_data['Age'] > 32) & (full_data['Age'] <= 48), 'Age'] = 2
    full_data.loc[(full_data['Age'] > 48) & (full_data['Age'] <= 64), 'Age'] = 3
    full_data.loc[ full_data['Age'] > 64, 'Age'] = 4
    if print_==True:
        print(full_data.sample(8))
    
    
    if print_==True:
        print('Cutting on Fare')
    train['Fareband'] = pd.cut(train['Fare'], 5, precision = 0)
    if print_==True:
        print(train[['Fareband', 'Survived']].groupby(['Fareband'], as_index=False).mean().sort_values(by='Fareband', ascending=True))
    
    full_data.loc[ full_data['Fare'] <= 102, 'Fare'] = 0
    full_data.loc[(full_data['Fare'] > 102) & (full_data['Fare'] <= 205), 'Fare'] = 1
    full_data.loc[(full_data['Fare'] > 205) & (full_data['Fare'] <= 307), 'Fare'] = 2
    full_data.loc[(full_data['Fare'] > 307) & (full_data['Fare'] <= 410), 'Fare'] = 3
    full_data.loc[ full_data['Fare'] > 410, 'Fare'] = 4
   
    full_data.loc[full_data.PassengerId==1044,'Fare'] = 0
    
    if print_==True:
        print(full_data.sample(8))
    
    if print_==True:
        print('Creating family feature')
    train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
    if print_==True:
        print(train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False))
    full_data['FamilySize'] = full_data['SibSp'] + full_data['Parch'] + 1
    
    if print_==True:
        print('extracting title')
    dataset_title = [i.split(",")[1].split(".")[0].strip() for i in full_data["Name"]]
    full_data["Title"] = pd.Series(dataset_title)
    
    if print_==True:
        full_data["Title"].head()
    
    if print_==True:
        print('converting to categorical data')
    full_data["Title"] = full_data["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    full_data["Title"] = full_data["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
    full_data["Title"] = full_data["Title"].astype(int)
    
    if print_==True:
        g = sns.factorplot(x="Title",y="Survived",data=full_data,kind="bar")
        g = g.set_xticklabels(["Master","Miss-Mrs","Mr","Rare"])
        g = g.set_ylabels("survival probability")
    
  
    if print_==True:
        print('converting male-female to 0-1')
    full_data = full_data.drop(labels = ["Name","Cabin", 'Ticket'], axis = 1)
    full_data.loc[ full_data['Sex'] == 'male', 'Sex'] = 0
    full_data.loc[ full_data['Sex'] == 'female', 'Sex'] = 1

    
    test = full_data.loc[full_data['PassengerId'].isin(test['PassengerId'])]
    test = test.drop(labels=["Survived"],axis = 1)
    train = full_data.loc[full_data['PassengerId'].isin(train['PassengerId'])]
    
    
    
    return train, test, full_data


#==============================================================================
#     #analysis
#==============================================================================
def analysis():
    
    (train_original, test_original, full_data) = featureextraction(False)
    for i in range(5):
        test = train_original.iloc[178*i:178*(i+1),:].copy()
        test = test.drop(labels=["Survived"],axis = 1)
        train = train_original.loc[~train_original['PassengerId'].isin(test['PassengerId'])]
        X_train = train.drop("Survived", axis=1)
        X_train = X_train.drop("PassengerId", axis=1).copy()
        Y_train = train["Survived"]
        X_test  = test.drop("PassengerId", axis=1).copy()
        X_train.shape, Y_train.shape, X_test.shape
    
        logreg = LogisticRegression()
        logreg.fit(X_train, Y_train)
        Y_pred = logreg.predict(X_test)
        acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
        print('Logistic regression:',acc_log,'%')
        
        
        svc = SVC()
        svc.fit(X_train, Y_train)
        Y_pred = svc.predict(X_test)
        acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
        print('SVC:',acc_svc,'%')
        
        for k in range(3,8,2):
            knn = KNeighborsClassifier(n_neighbors = k)
            knn.fit(X_train, Y_train)
            Y_pred = knn.predict(X_test)
            acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
            print('%s KNeighbors:'%k,acc_knn,'%')
        
        decision_tree = DecisionTreeClassifier()
        decision_tree.fit(X_train, Y_train)
        Y_pred = decision_tree.predict(X_test)
        acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
        print('Decision Tree:',acc_decision_tree,'%')
        
        random_forest = RandomForestClassifier(n_estimators=100)
        random_forest.fit(X_train, Y_train)
        Y_pred = random_forest.predict(X_test)
        random_forest.score(X_train, Y_train)
        acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
        print('Random Forest:',acc_random_forest,'%')
        
        Naive_bayes = GaussianNB()
        Naive_bayes.fit(X_train, Y_train)
        Y_pred = Naive_bayes.predict(X_test)
        Naive_bayes.score(X_train, Y_train)
        acc_Naive_bayes = round(Naive_bayes.score(X_train, Y_train) * 100, 2)
        print('Naive Bayes:',acc_Naive_bayes,'%')
        
        MLP = MLPClassifier(hidden_layer_sizes=(15,15,15))
        MLP.fit(X_train, Y_train)
        Y_pred = MLP.predict(X_test)
        MLP.score(X_train, Y_train)
        acc_MLP = round(MLP.score(X_train, Y_train) * 100, 2)
        print('MLP:',acc_MLP,'%')
        print('\n')
        
    X_train = train_original.drop("Survived", axis=1)
    X_train = X_train.drop("PassengerId", axis=1)
    Y_train = train_original["Survived"]
    X_test  = test_original.drop("PassengerId", axis=1).copy()
    X_train.shape, Y_train.shape, X_test.shape
    
    Submission_classifier =  KNeighborsClassifier(n_neighbors = 7)
    Submission_classifier.fit(X_train, Y_train)
    Y_pred = Submission_classifier.predict(X_test)
    Submission_classifier.score(X_train, Y_train)
    Submission_classifier_score = round(Submission_classifier.score(X_train, Y_train) * 100, 2)
    
    submission = pd.DataFrame({
        "PassengerId": test_original["PassengerId"],
        "Survived": Y_pred.astype(int)
    })
    submission.to_csv('submission_KNN_2.csv', index=False)
    print('Submission accuracy:',Submission_classifier_score,'%')
        
    
###########################################################
### main
def main():
    
    
    
    print('\nAll functions imported into memory')

###########################################################
### start main
if __name__ == "__main__":
    main()

