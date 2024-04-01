'''
 Preprocess data
    # 'test'
    preprocess_test_data(df_load,additional_name='')
        - to apply the same transformation that was learned from the training data
    
    # 'training'
    
    # fix choice of imputer preprocess:

    preprocess_data(df_load, threshold_feature_drop=27,directory_name='models',additional_name='')
        - drop features with high missing values: threshold_feature_drop=27
        - fixed imputers for different features categories:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            num_imputer = KNNImputer(n_neighbors=10)
            cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore') 
            scaler = QuantileTransformer()
        - split the data into training and testing sets
        -return:
            X_train = preprocessor.fit_transform(X_train) # model can learn the parameters of the transformations from this data
            X_test= preprocessor.transform(X_test) #to apply the same transformation that was learned from the training data

    # function to choose imputers for each feature categories
    
    preprocess_data_choose_imputer(df_load, threshold_feature_drop=25,directory_name='models',additional_name='',cat_imputer=SimpleImputer(strategy='most_frequent'), num_imputer=KNNImputer(n_neighbors=10), scaler=QuantileTransformer())
        -function to choose imputers for each feature categories
         - choose parameters values: 
            categorical and binomial (yes/no) feature imputer:
                cat_imputer=SimpleImputer(strategy='most_frequent'), 
            numerical feature imputer and numerical data scaler:
                num_imputer=KNNImputer(n_neighbors=10), 
                scaler=QuantileTransformer()

        - split the data into training (80%) and testing (20%) sets
        -return:
            X_train = preprocessor.fit_transform(X_train) # model can learn the parameters of the transformations from this data
            X_test= preprocessor.transform(X_test) #to apply the same transformation that was learned from the training data

'''
########################################## STEP 6 ##############################################################
# Preprocess Data

import numpy as np
import pandas as pd
import pickle

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, PowerTransformer, Normalizer, QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from modules.datatype_extraction import *

def preprocess_test_data(df_load,additional_name=''):
    num_features,bin_features,cat_features,ord_features,ddmmyy_features,all_cat_features = datatype_extraction_procedure(df_load)
    num_features.remove('l2_TTNT')
    df_load['event'] = 1
    with open(f'../models/imputer_{additional_name}.pkl', 'rb') as file:
        preprocessor = pickle.load(file)

    X_test= preprocessor.transform(df_load) #to apply the same transformation that was learned from the training data
    return X_test 


def preprocess_data(df_load, threshold_feature_drop=27,directory_name='models',additional_name=''):


    missing_percentage = (df_load.isnull().sum() / len(df_load)) * 100
 
    columns_to_drop = missing_percentage[missing_percentage >= threshold_feature_drop].index
    df_raw = df_load.drop(columns=columns_to_drop)
    df_raw.shape
    num_features,bin_features,cat_features,ord_features,ddmmyy_features,all_cat_features = datatype_extraction_procedure(df_raw)
    num_features.remove('l2_TTNT')

    #num_features=num_features.remove('l2_TTNT')
    #imputer definition 
    cat_imputer = SimpleImputer(strategy='most_frequent')
    num_imputer = KNNImputer(n_neighbors=10)
    cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore') 
    scaler = QuantileTransformer() 
    # Create an instance of OneHotEncoder


    lab_encoder = LabelEncoder()
    #KNNImputer(n_neighbors=10) #SimpleImputer(strategy='constant', fill_value=99999) #KNNImputer(n_neighbors=10) #SimpleImputer(strategy='constant')#, fill_value=99999) # KNNImputer(n_neighbors=10)
    # cat_imputer = SimpleImputer(strategy='most_frequent')
    
    preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('num_imputer', num_imputer),
                          ('num_scaler', scaler)
                          ]), num_features),  # Apply the numerical transformer to numerical columns
        ('cat', Pipeline([('cat_imputer', cat_imputer ),
                        # ('cat_round', Rounder()),
                          ('cat_encoder', cat_encoder)
                        ]), cat_features+ord_features),
        ('bin', Pipeline([('cat_imputer', cat_imputer)]), bin_features)
    ], remainder='passthrough').set_output(transform="pandas")

    
    #split data for train and test set
    df_raw['event'] = 1
    
    # Split the data into training and testing sets
    X_train, X_test, = train_test_split(df_raw, test_size=0.2, random_state=43)

    
    X_train = preprocessor.fit_transform(X_train) # model can learn the parameters of the transformations from this data
   
    X_test= preprocessor.transform(X_test) #to apply the same transformation that was learned from the training data
    with open(f'../{directory_name}/imputer_{additional_name}.pkl', 'wb') as file:
        pickle.dump(preprocessor, file)

    return X_train, X_test 


def preprocess_data_choose_imputer(df_load, threshold_feature_drop=25,directory_name='models',additional_name='',cat_imputer=SimpleImputer(strategy='most_frequent'), num_imputer=KNNImputer(n_neighbors=10), scaler=QuantileTransformer()):


    missing_percentage = (df_load.isnull().sum() / len(df_load)) * 100
 
    columns_to_drop = missing_percentage[missing_percentage >= threshold_feature_drop].index
    df_raw = df_load.drop(columns=columns_to_drop)
    df_raw.shape
    num_features,bin_features,cat_features,ord_features,ddmmyy_features,all_cat_features = datatype_extraction_procedure(df_raw)
    num_features.remove('l2_TTNT')

    #num_features=num_features.remove('l2_TTNT')
    #imputer definition 
    
    # Create an instance of OneHotEncoder
    cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    lab_encoder = LabelEncoder()
    #KNNImputer(n_neighbors=10) #SimpleImputer(strategy='constant', fill_value=99999) #KNNImputer(n_neighbors=10) #SimpleImputer(strategy='constant')#, fill_value=99999) # KNNImputer(n_neighbors=10)
    # cat_imputer = SimpleImputer(strategy='most_frequent')
    
    preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('num_imputer', num_imputer),
                          ('num_scaler', scaler)
                          ]), num_features),  # Apply the numerical transformer to numerical columns
        ('cat', Pipeline([('cat_imputer', cat_imputer ),
                        # ('cat_round', Rounder()),
                          ('cat_encoder', cat_encoder)
                        ]), cat_features+ord_features),
        ('bin', Pipeline([('cat_imputer', cat_imputer)]), bin_features)
    ], remainder='passthrough').set_output(transform="pandas")

    
    #split data for train and test set
    df_raw['event'] = 1
    
    # Split the data into training and testing sets
    X_train, X_test, = train_test_split(df_raw, test_size=0.2, random_state=43)

    
    X_train = preprocessor.fit_transform(X_train) # model can learn the parameters of the transformations from this data
   
    X_test= preprocessor.transform(X_test) #to apply the same transformation that was learned from the training data
    with open(f'../{directory_name}/imputer_{additional_name}.pkl', 'wb') as file:
        pickle.dump(preprocessor, file)

    return X_train, X_test 
