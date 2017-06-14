# -*- coding: utf-8 -*-
import sys
import os
import time
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree


reload(sys)
sys.setdefaultencoding('utf8')

##Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    model = LogisticRegression(penalty='l2') # penalty 可選擇為l1 or l2，預設為l2
    model.fit(train_x, train_y)
    return model


##Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    model = RandomForestClassifier(n_estimators=40) # n_estimators指的是模型要使用多少CART trees
    model.fit(train_x, train_y)
    return model


##Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    model = tree.DecisionTreeClassifier(criterion='gini') # here you can change the algorithm as gini or entropy
    model.fit(train_x, train_y)
    return model



def read_data(data_file):
    dataset_train = pd.read_csv('Accidents_2015.csv', low_memory=False)
    features_train = ['Accident_Severity','Light_Conditions', 'Weather_Conditions', 'Road_Surface_Conditions','Special_Conditions_at_Site','Carriageway_Hazards']
    x = dataset_train[features_train]
    y = dataset_train["Number_of_Casualties"]

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)

    return train_x, train_y, test_x, test_y


if __name__ == '__main__':

    data_file = "Accidents_2015.csv"
    thresh = 0.5

    test_classifiers = [ 'LR', 'RF', 'DT']
    classifiers = {'LR': logistic_regression_classifier,
                   'RF': random_forest_classifier,
                   'DT':decision_tree_classifier,
                   }

    print 'reading training and testing data...'
    train_x, train_y, test_x, test_y = read_data(data_file)
    num_train, num_feat = train_x.shape
    num_test, num_feat = test_x.shape

    print '******************** Data Info *********************'
    print '#training data: %d, #testing_data: %d, dimension: %d' % (num_train, num_test, num_feat)
    for classifier in test_classifiers:
        print '******************* %s ********************' % classifier
        start_time = time.time()
        model = classifiers[classifier](train_x, train_y)
        print 'training took %fs!' % (time.time() - start_time)
        predict = model.predict(test_x)
        accuracy = metrics.accuracy_score(test_y, predict)
        print 'accuracy: %.2f%%' % (100 * accuracy)
