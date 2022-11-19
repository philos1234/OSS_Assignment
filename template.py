# PLEASE WRITE THE GITHUB URL BELOW!
# https://github.com/philos1234/OSS_Assignment

import sys
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def load_dataset(dataset_path):
    
    df = pd.read_csv(os.getcwd() + "/" + dataset_path)
    return df

def dataset_stat(dataset_df):
    n_feats = data_df.shape[1] - 1 # without target column
    n_class0 = len(data_df.loc[data_df['target'] == 0])
    n_class1 = len(data_df.loc[data_df['target'] == 1])
    return n_feats, n_class0, n_class1 

def split_dataset(dataset_df, testset_size):
    dataset_df_data = dataset_df.drop('target', axis=1)
    dataset_df_target = dataset_df['target']
    return train_test_split(dataset_df_data, dataset_df_target, test_size=testset_size)

def decision_tree_train_test(x_train, x_test, y_train, y_test):
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    
    acc = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    
    return acc, precision, recall

def random_forest_train_test(x_train, x_test, y_train, y_test):
    rf_cls = RandomForestClassifier()
    rf_cls.fit(x_train, y_train)
    pred = rf_cls.predict(x_test)
    
    acc = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    
    return acc, precision, recall



def svm_train_test(x_train, x_test, y_train, y_test):
    
    pipe = make_pipeline(
    StandardScaler(),
    SVC()
    )
    pipe.fit(x_train,y_train)
    pred = pipe.predict(x_test)
    
    acc = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    
    return acc, precision, recall


def print_performances(acc, prec, recall):
    #Do not modify this function!
    print ("Accuracy: ", acc)
    print ("Precision: ", prec)
    print ("Recall: ", recall)

if __name__ == '__main__':
    #Do not modify the main script!
    data_path = sys.argv[1]
    data_df = load_dataset(data_path)

    n_feats, n_class0, n_class1 = dataset_stat(data_df)
    print ("Number of features: ", n_feats)
    print ("Number of class 0 data entries: ", n_class0)
    print ("Number of class 1 data entries: ", n_class1)

    print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
    x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

    acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
    print ("\nDecision Tree Performances")
    print_performances(acc, prec, recall)
    
    acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
    print ("\nRandom Forest Performances")
    print_performances(acc, prec, recall)

    acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
    print ("\nSVM Performances")
    print_performances(acc, prec, recall)