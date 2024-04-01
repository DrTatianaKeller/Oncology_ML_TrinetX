########################################################################################################
#This file contains function and procedure for data preparation
#Function should be called in main notebook

'''
STEPS:
1. Read dataframe and Create directory in 'data_information/' to save data information
2. Extract features datatype and create datatype lists from index_list_num_cat_ord_bin.csv

3. Data correction:
    1. Nein to Nan for ['dx_lightrat', 'l1_lightrat', 'l2_lightrat', 'dx_lightkap', 'l1_lightkap', 'l2_lightkap', 'dx_lightlam', 'l1_lightlam', 'l2_lightlam']
    2. Convert numerical features type to float (original type: object)
    3. correction for 'kofscore':  20.21 to Nan
    4. ECOG value 9.0 to Nan , bin category 0,1 - 0; 2,3,4 - 1
    5. dx_sex to dx_female_sex 1/0,  bin category
    6. Find outliers. Change for outliers 99-type input to Nan 
    7. BMI  
         1.Find BMI and using possible BMI interval estimate the swap weight and height columns
         2. BMI for dx, l1, l2 instead of height and weight features
         3. Calculate relative BMI change : l1_BMI_change, l2_BMI_change

    8. Data type columns:
        1. Calculate duration of rtx
        2. Calculate relative age when patient was diagnosed
        3. Drop data ddmmyy type columns

4. Data mapping:
    1. Replace 'Ja'/ 'Nein'/ 'Nicht getestet' to 1/ 0/ NaN (99) numerical value
    2. Mapping object type data:
        1. Sorting by number in the line:
            Example: 
            '<3,5 mg/L': 1, '3,5 - 5,5 mg/L': 2, '>5,5 mg/L': 3
        2. Sorting by Stadium I, II, III: giving indexes of mapping 1, 2, 3
        3. For Unbekant Stadium: index 0

5. Dropping Features
    
    'training':

        1. One unique value in column - 54 feautures was dropped
        2. Imbalance feature drop and categories combination with small statistic (< 6%) -> Create list of combined categories per each feature
            - Save data about categories combination to 'dictionaries/replaced_values.json'
        3. Drop all repeated/ dublicate information/ high correlated values:
            - drop_list = ['dx_weight', 'l1_weight', 'l2_weight', 'dx_height', 'l1_height','l2_height',	'l1_BMI','l2_BMI', 'l1_klinstudie','l2_klinstudie','dx_mikrogl','l1_mikrogl','l2_mikrogl','dx_albugem','l1_albugem','l2_albugem','l1_protg']
            - Drop 'l1_protg' due to hight correlation with 'l2_protg'
        4. Drop outliers for ['dx_protg', 'l1_protg', 'l2_protg'] 

        after all steps remain features list is saved in  'dictionaries/features_list_to_keep.json'`
    
    'test':    

        1. Open:
        'dictionaries/replaced_values.json'
            - to combine categories inside each feature the same way as was done for training dataset 
        'dictionaries/features_list_to_keep.json'
            - to drop all features the same way as for training dataset
            -if some column is missing, make warning to retrain dataset

        2. Remove outliers

6. Preprocess data
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


########################################################################################################
# Libraries requirement

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, PowerTransformer, Normalizer, QuantileTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
import re
from helper.functions.modelling_functions import datatype_extraction_procedure
import json
import pickle


########################################## STEP 1 ##############################################################
# Notebook display output set up
def Set_up_option():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    #pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_colwidth', -1)


# Read Dataframe
def read_df(data_name, data_path ='data/raw/' ):
    data_path = f'{data_path}{data_name}.csv'
    df = pd.read_csv(data_path)
    if 'patient_id' in df.columns:
        df = df.drop(columns='patient_id')
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns='Unnamed: 0')
    return df

#create new_directory in  parent_directory
def create_directory_internal(parent_directory, new_directory):
    new_directory_path = f'{parent_directory}/{new_directory}'
    if os.path.exists(new_directory_path):
        print(f'Directory {new_directory_path} exist')
    else:    
        os.makedirs(new_directory_path, exist_ok=True)
        print(f"Directory '{new_directory_path}' created successfully.")


# create directory in /intermediate_report/ in format yy-mm-dd _ data_name
# to store all necessary information about dataset. 
                
def create_directory_main(data_name, parent_directory = "intermediate_report"):
 # Get the current date and time
    now = datetime.now()

    # Format the date and time as mm:dd:hh:mm
    date_time_str = now.strftime("%y-%m-%d")
    dir_name = f'{date_time_str}_{data_name}'
    # Name of the parent directory
    
    # Create the parent directory if it doesn't exist


    if os.path.exists(parent_directory):
        print(f'Directory {parent_directory} exist')
    else:    
        os.makedirs(parent_directory, exist_ok=True)
        print(f"Directory '{parent_directory}' created successfully.")


    # Create a directory with the formatted name inside the parent directory
    directory_name = os.path.join(parent_directory, dir_name)
    if os.path.exists(directory_name):
        print(f'Directory {directory_name} exist')
    else:    
        os.makedirs(directory_name)
        print(f"Directory '{directory_name}' created successfully.")
    
    figures_directory_name = os.path.join(parent_directory, dir_name,'figures')
    if os.path.exists(figures_directory_name):
        print(f'Directory {figures_directory_name} exist')
    else:    
        os.makedirs(figures_directory_name)
        print(f"Directory '{figures_directory_name}' created successfully.")
            
    return directory_name

########################################## STEP 2 ##############################################################

# data type extraction function for one type of data. 
def datatype_solo_extraction(df, type_name, path_list="dictionaries/"):
    df_data_type = pd.read_csv(path_list + "index_list_num_cat_ord_bin_corrected.csv")  
    #rename columns in df_data_type
    new_column_names = ["ind2", "feature", "data_type"]
    df_data_type.rename(columns=dict(zip(df_data_type.columns, new_column_names)), inplace=True)
    df_data_type=df_data_type.drop(columns = "ind2")

    def data_type_list(dataframe, *data_type_names):
    
        data_types = []
        for data_type_name in data_type_names:
            data_types.append(data_type_name)
        
        # Filter the DataFrame to select features with specified data types
        lib_features = df_data_type[df_data_type["data_type"].isin(data_types)]["feature"].tolist()

        features = []
        for feature in dataframe.columns:
            if feature in lib_features:
                features.append(feature)
        return features
    
    print("Control output of used datatype: ")
    type_features_list = data_type_list(df, type_name)
    print(type_features_list)
    
    return type_features_list
    





########################################## STEP 3 ##############################################################
# Data Correction

#This function combine all data correction steps
def data_correction(df, ecog_type = 'bin'):
    #Nein to Nan for numeric values
    cols_to_correct = ['dx_lightrat', 'l1_lightrat', 'l2_lightrat', 'dx_lightkap', 'l1_lightkap', 'l2_lightkap', 'dx_lightlam', 'l1_lightlam', 'l2_lightlam']
    df = replace_value_function(df,cols_to_correct,original_value='Nein', new_value=np.nan)
    #Type to float for numeric values
    df=type_convert_to_float(df)
    #Body surface correction 20.21 to Nan
    df = replace_value_function(df,correction_list = ['dx_kofscore','l1_kofscore','l2_kofscore'],original_value=20.21, new_value=np.nan)
    #ECOG correction 9.0 to Nan
    df_ecog = data_correction_ecog(df,ecog_type = ecog_type)
    #Sex correction to Female 0/1
    df_sex = data_correction_sex(df_ecog)
    #99-type values to Nan
    df_nan = Converter_to_Nan(df_sex)
    #BMI/weight/height correction -> BMI difference between treatments
    df_BMI = BMI_correction(df_nan)

    df_ddmmyy_correction = DDMMYY_type_correction(df_BMI)


    df_correct = df_ddmmyy_correction
    return df_correct
    


# Function, that are used in data_correction function:

# ### 1. ECOG value 9.0 to Nan, Transform to binomial value ECOG = 0,1 -> 0; ECOG = 2,3,4 -> 1

# %%
def data_correction_ecog(df,ecog_type = 'bin'):
    list_ecog = ['dx_ecog','l1_ecog','l2_ecog' ]
    for column in list_ecog:
        df[column] = df[column].replace(9.0, np.nan)
        if ecog_type == 'bin':
            new_column = column+'bin'
            df[new_column] = df[column].apply(lambda x: 1 if x > 1 else 0)
            df = df.drop(columns =[column] )

    return df


# ### 2. Sex to Female 1/0


def data_correction_sex(df):
    # Create a mapping for 'Sex'
    sex_mapping = {'Männlich': 0, 'Weiblich': 1}

    # Rename the 'Sex' column to 'Male_sex' and apply the mapping
    df['dx_sex'] = df['dx_sex'].replace(sex_mapping)
    df = df.rename(columns={'dx_sex': 'dx_female_sex'})
    return df



# ### 3. Convert numerical features to float


def type_convert_to_float(df_02_mapping):
    # Iterate through features in the DataFrame and find outliers in numerical features
    print('Before convertion: ')

    
    num_features = datatype_solo_extraction(df_02_mapping,'num')
   
    
    print(df_02_mapping[num_features].dtypes)
    
    for feature in df_02_mapping.columns:
        if feature in num_features: 
            if df_02_mapping[feature].dtype == 'object':
            # Convert the feature to float
                try: 
                    df_02_mapping[feature] = df_02_mapping[feature].astype(float)
                except ValueError: 
                    print(f'Conversion to float failed for {feature}')
            
    print('After convertion: ')
    print(df_02_mapping[num_features].dtypes)
    return df_02_mapping


def df_missing_count(df_02_mapping,directory_name):
    # Define the output file path
    output_file_path = f'{directory_name}/raw_unique_values_and_missing_count.csv'
    #df_02_mapping_for_count = df_02_mapping.drop(columns=['Unnamed: 0'])
    num_features,_,_,_,_,all_cat_features = datatype_extraction_procedure(df_02_mapping)
    
    selected_feature_list = all_cat_features+num_features
    df_02_mapping=df_02_mapping[selected_feature_list]
    # Open the output CSV file for writing
    with open(output_file_path, 'w') as file:
        for column in df_02_mapping.columns:
            unique_counts = df_02_mapping[column].value_counts().to_dict()
            missing_count = df_02_mapping[column].isna().sum()
            file.write(f'Column: {column}\n')
            file.write(f'    Missing Count: {missing_count}\n')
            for value, count in unique_counts.items():
                if column not in num_features:
                    file.write(f'    Value: {value}, Count: {count}\n')
            file.write('\n')
    print('\n')
    print(output_file_path + ' was created')


#Function to search outliers
def find_outliers(data, feature_name):
    
    # Calculate quartiles
    # Check the data type of a specific feature
    
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)

    # Calculate the IQR
    IQR = Q3 - Q1

    # Calculate lower and upper bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

# Find outliers
    upper_outliers = data[data > upper_bound]
    lower_outliers = data[data < lower_bound]
    print(f'{feature_name} upper_outliers: {upper_outliers}')
    print(f'{feature_name} lower_outliers: {lower_outliers}')
    return upper_outliers,lower_outliers
    
# This function search 99-type data throught outliers and convert it to Nan values    
def Converter_to_Nan(df_02_mapping):    
    #look for outliers and change 99-type value to Nan
    nan_list = [9.0, 99.0, 999.0, 9999.0, 99999.0, 999999.0, 9999999.0, 99999999.0, 99999.99, 999999.99, 9999999.99, 99999999.99, 999.99, 99.99, 99.9,9.9,9.99, 9999]
    df_03_mapping= df_02_mapping.copy()
    num_features = datatype_solo_extraction(df_03_mapping,'num')
    for feature in df_03_mapping.columns:
        if feature in num_features:
            u_outliers, l_outliers = find_outliers(df_03_mapping[feature], feature)
            
            # Replace upper outliers with NaN if they match nan_list values
            df_03_mapping.loc[u_outliers[u_outliers.isin(nan_list)].index, feature] = np.nan
            
            # Replace lower outliers with NaN if they match nan_list values
            df_03_mapping.loc[l_outliers[l_outliers.isin(nan_list)].index, feature] = np.nan
    return df_03_mapping

#Replace value for body surfave 20.21 to Nan
def replace_value_function(df_03_mapping, correction_list ,original_value, new_value):
    #num_correction_list = ['dx_kofscore','l1_kofscore','l2_kofscore']
    for feature in correction_list:
        #df_03_mapping[feature] = df_03_mapping[feature].replace({20.21: np.nan})
        df_03_mapping[feature] = df_03_mapping[feature].replace({original_value: new_value})
    return df_03_mapping

#Weight correction
#By default  this function will change all weight data > ref_munber = 250 to realistic one
def weight_correction_function(df_03_mapping, weight_list=['dx_weight','l1_weight','l2_weight'], ref_number = 250):
    for list_item in weight_list:
        df_03_mapping[list_item][df_03_mapping[list_item] > ref_number] = df_03_mapping[list_item] / 10
    return df_03_mapping

#Find fake BMI
def BMI_outliers(df_03_mapping):

    # Calculate BMI for 'dx' group
    df_03_mapping['dx_BMI'] = df_03_mapping['dx_weight'] / ((df_03_mapping['dx_height'] / 100) ** 2)

    # Calculate BMI for 'l1' group
    df_03_mapping['l1_BMI'] = df_03_mapping['l1_weight'] / ((df_03_mapping['l1_height'] / 100) ** 2)

    # Calculate BMI for 'l2' group
    df_03_mapping['l2_BMI'] = df_03_mapping['l2_weight'] / ((df_03_mapping['l2_height'] / 100) ** 2)

    # Using possible BMI interval estimate the swap weight and height columns
    # Define the lower and upper bounds for BMI
    lower_bound = 14
    upper_bound = 53

    # Initialize a variable to keep track of the total number of dropped rows
    total_dropped_rows = 0
    df_correct_BMI = df_03_mapping.copy()  # Create a copy to store correct rows
    df_dropped_BMI = pd.DataFrame()  # Create a DataFrame to store dropped rows
    # Calculate the number of rows before dropping for the current column

    original_row_count = len(df_correct_BMI)

    # Iterate through the columns in the DataFrame
    for column in df_03_mapping.columns:
        if 'BMI' in column:
            # Filter rows where the current column is not in the specified range
            df_dropped_rows = df_03_mapping[(df_03_mapping[column] < lower_bound) | (df_03_mapping[column] > upper_bound)]
            # Update the main DataFrame to keep only valid rows
            df_correct_BMI = df_correct_BMI[(df_correct_BMI[column] >= lower_bound) & (df_correct_BMI[column] <= upper_bound)]
            print(f'Wrong data in column {column}:')
            print(df_dropped_rows[column])
            # Append the dropped rows to the DataFrame for dropped rows
            df_dropped_BMI = df_dropped_BMI.append(df_dropped_rows)

    # Calculate the number of dropped rows for the current column
    dropped_row_count = original_row_count - len(df_correct_BMI)

    # Add the dropped rows count to the total
    total_dropped_rows += dropped_row_count
    #df_dropped_BMI.to_csv('dropped_BMI_rows.csv', index=False)
    # Print the total number of dropped rows
    print(' ')
    print(f"Total number of rows with wrong data: ")
    print(df_dropped_BMI.shape)
    

    BMI_list = ['dx_weight',	'l1_weight',	'l2_weight',	'dx_height',	'l1_height',	'l2_height']
    print(' ')
    print('Wrong data: ')
    print(df_dropped_BMI[BMI_list])

    return df_03_mapping, df_dropped_BMI



def BMI_correction(df_03_mapping):
    BMI_list = ['dx_weight',	'l1_weight',	'l2_weight',	'dx_height',	'l1_height',	'l2_height']
   
    df_03_mapping = weight_correction_function(df_03_mapping)

    df_03_mapping, df_dropped_BMI = BMI_outliers(df_03_mapping)

# Change wrong input swap heigh and weight
    
        # Iterate through rows
    for index, row in df_dropped_BMI.iterrows():
        for prefix in ['dx_', 'l1_', 'l2_']:
            weight_col = prefix + 'weight'
            height_col = prefix + 'height'
            
            # Check if 'weight' is greater than 100 and 'height' is less than 100 in the current row
            if row[weight_col] > 100 and row[height_col] < 100:
                # Swap 'weight' and 'height' values in the same row
                df_dropped_BMI.at[index, weight_col] = row[height_col]
                df_dropped_BMI.at[index, height_col] = row[weight_col]
            elif row[weight_col] > 400:
                df_dropped_BMI.at[index, weight_col] = row[weight_col]/10
            elif row[weight_col] > 100 and row[height_col] > 100:
                df_dropped_BMI.at[index, weight_col] = row[height_col]-100
                df_dropped_BMI.at[index, height_col] = row[weight_col]
            elif row[weight_col] < 100 and row[height_col] < 100:
                
                df_dropped_BMI.at[index, height_col] = row[height_col]+100

    print(' ')
    print('Corrected data: ')
    print(df_dropped_BMI[BMI_list])

#Update Dataframe with correct input 
    df_03_mapping.loc[df_dropped_BMI.index] = df_dropped_BMI

# Fill missing values for weight and height.
    # height is constant +- 2 cm

    cols_w = ['dx_weight',	'l1_weight',	'l2_weight']
    cols_h= ['dx_height', 'l1_height','l2_height']

    #use forward and backword fill:
    print('Weight Correction: ')
    df_03_mapping[cols_w] = df_03_mapping[cols_w].ffill(axis=1)
    print(' ')
    print('Data with Nan:')
    print(df_03_mapping[cols_w].head())

    df_03_mapping[cols_w] = df_03_mapping[cols_w].bfill(axis=1)
    print(' ')
    print('Filled Data:')
    print(df_03_mapping[cols_w].head())

    print('Height Correction: ')
    df_03_mapping[cols_h] = df_03_mapping[cols_h].ffill(axis=1)
    print(' ')
    print('Data with Nan:')
    print(df_03_mapping[cols_h].head())

    df_03_mapping[cols_h] = df_03_mapping[cols_h].bfill(axis=1)
    print(' ')
    print('Filled Data:')
    print(df_03_mapping[cols_h].head())

    print(' ')
    print('Check if data is in good range for height and weight: ')
    print(' ')
    print('Height <110:')
    filtered_rows = df_03_mapping[(df_03_mapping[cols_h] < 110).any(axis=1)]
    print(filtered_rows[BMI_list])
    print(' ')
    print('Weight >140:')
    filtered_rows_w = df_03_mapping[(df_03_mapping[cols_w] > 140).any(axis=1)]
    print(filtered_rows_w[BMI_list])

    # Recalculate BMI with correct data
    df_03_mapping=df_03_mapping.drop(columns = ['dx_BMI','l1_BMI','l2_BMI'])
    df_03_mapping, df_dropped_BMI = BMI_outliers(df_03_mapping)

    # Change BMI to BMI change

    df_03_mapping['l1_BMI_change'] = round((df_03_mapping['dx_BMI']-df_03_mapping['l1_BMI'])/df_03_mapping['dx_BMI'],2)
    df_03_mapping['l2_BMI_change'] = round((df_03_mapping['l1_BMI']-df_03_mapping['l2_BMI'])/df_03_mapping['l1_BMI'],2)
    print(' ')
    print('BMI_change was added to the dataframe')

    return df_03_mapping



def DDMMYY_type_correction(df_03_mapping):
    ddmmyy_features = datatype_solo_extraction(df_03_mapping, 'date')
    print('START: DDMMYY_type_correction \n')
    # 1. calculate duration of rtx
   
    print('1. calculate duration of rtx')
    df_03_mapping['l1_rtxdauer'] = round((df_03_mapping['l1_rtxendedat'].apply(pd.to_datetime)-df_03_mapping['l1_rtxstartdat'].apply(pd.to_datetime)).dt.total_seconds() / (365.25 * 24 * 3600))
    df_03_mapping['l2_rtxdauer'] = round((df_03_mapping['l2_rtxendedat'].apply(pd.to_datetime)-df_03_mapping['l2_rtxstartdat'].apply(pd.to_datetime)).dt.total_seconds() / (365.25 * 24 * 3600))

   
    #2. calculate relative age when patient was diagnosed
    print('2. calculate relative age when patient was diagnosed')
    Day_birth = df_03_mapping['dx_birthd'].apply(pd.to_datetime)
    Data_list = ['dx_iddat', 'l1_opdat','l2_opdat'] 

        # Convert object-type columns to datetime
    df_03_mapping[Data_list] = df_03_mapping[Data_list].apply(pd.to_datetime)

        # Calculate and store the differences in years for each column
    for column in Data_list:
        new_column_name = f'{column}_age'
        df_03_mapping[new_column_name] = round((df_03_mapping[column]-Day_birth).dt.total_seconds() / (365.25 * 24 * 3600))
        print(f'{new_column_name} column was added to dataframe')
    
    # 3. Drop data ddmmyy-type columns
    print('3. Drop data ddmmyy-type columns')
    df_03_mapping=df_03_mapping.drop(columns = ddmmyy_features)
    df_03_mapping['l2_opdat_age'].count()
    print('END: DDMMYY_type_correction \n')
    
    return df_03_mapping

########################################## STEP 4 ##############################################################
# # Data mapping

# This function combines all data mapping functions:
def data_mapping(df, directory_name, data_name):
    df = data_mapping_Ja_Nein(df)
    df_02_mapping, value_mapping = data_mapping_object(df, directory_name, data_name)
    return df_02_mapping, value_mapping
    

# Subfunctions used in datamapping function:

# ### 1. Replace 'Ja'/ 'Nein'/ 'Nicht getestet' to 1/ 0/ NaN numerical value
def data_mapping_Ja_Nein(df):
    #Replace Ja/Nein/Nicht getestet to 1/0/Nan numerical values
    replacement_dict = {'Ja': int(1), 'Nein': int(0), 'Nicht getestet':np.nan}

    # Use the `replace` method to replace values in the entire DataFrame
    df_02 = df.replace(replacement_dict)
    return df_02


# ### 2. Mapping object type data
# 1. Sorting by number in the line:
# Example: 
# '<3,5 mg/L': 1, '3,5 - 5,5 mg/L': 2, '>5,5 mg/L': 3
# 2. Sorting by Stadium I, II, III: giving indexes of mapping 1, 2, 3
# 3. For Unbekant Stadium: index 0
# 
# Create the List ('List_ordinary_values') selecting all features where order is important, so in a mapping we will change the values for category's index


def sort_strings_by_number(s):
    # Use regular expression to extract the numeric part
    numeric_part = re.search(r'\d+', str(s))
    if numeric_part:
        return int(numeric_part.group())
    else:
        return float('inf')  # For values with no numeric part, place them at the end


def data_mapping_object(df, directory_name, data_name):
# Create an empty value_mapping dictionary
    value_mapping = {}
    List_ordinary_values =[] # List to add ordinary values features

     
    _,_,cat_features,ord_features,ddmmyy_features,all_cat_features = datatype_extraction_procedure(df)

    for column in df.select_dtypes(include=['object']):
        if column in all_cat_features:
            unique_values = df[column].unique()
            
            # Sort the unique values using the custom sorting function
            unique_values = sorted(unique_values, key=sort_strings_by_number)
            
            # Create a mapping with sorted values, handling 'NaN' as a string
            mapping = {}
            last_stadium = 0  # Track the last encountered "Stadium I," "Stadium II," or "Stadium III"
            count = 0
            for idx, value in enumerate(unique_values):
                if pd.notna(value) and "ISS Stadium III" in str(value):
                    last_stadium = 3
                    mapping[value] = last_stadium
                elif pd.notna(value) and "ISS Stadium II" in str(value):
                    last_stadium = 2
                    mapping[value] = last_stadium
                elif pd.notna(value) and "ISS Stadium I" in str(value):
                    last_stadium = 1
                    mapping[value] = last_stadium
                
                elif pd.notna(value) and "Stadium III" in str(value):
                    last_stadium = 3
                    mapping[value] = last_stadium
                elif pd.notna(value) and "Stadium II" in str(value):
                    last_stadium = 2
                    mapping[value] = last_stadium
                elif pd.notna(value) and "Stadium I" in str(value):
                    last_stadium = 1
                    mapping[value] = last_stadium
                elif "Unbekannt" in str(value):
                    mapping[value] = np.nan
                
                elif value in (0, 1, 99, '0', '1', '99', np.nan):
                    mapping[value]=value
                
                elif pd.notna(value) and "Unfitter" in str(value):
                    last_stadium = 3
                    mapping[value] = last_stadium
                
                elif pd.notna(value) and "Eingeschränkte" in str(value):
                    last_stadium = 2
                    mapping[value] = last_stadium

                elif pd.notna(value) and "Fitter" in str(value):
                    last_stadium = 1
                    mapping[value] = last_stadium

                else:
                    last_stadium += 1
                    mapping[value] = last_stadium
            #print(column)
                if count == 0:  # to add the feature name only once to the list
                    if pd.notna(value) and (
                        re.search(r'Stadium [I|II|III]', str(value)) or re.search(r'[<|>]', str(value))
                    ):
                        List_ordinary_values.append(column)
                        count = count + 1
        
        
            
            value_mapping[column] = mapping

        # Display the value mappings
        with open(f'{directory_name}/mapping_{data_name}_all_sort.txt', 'w') as file:   
            for column, mapping in value_mapping.items():
                file.write(f' {column}: {mapping}\n')

        with open(f'{directory_name}/mapping_{data_name}_ordinary_sort.txt', 'w') as file:   
            for column, mapping in value_mapping.items():
                if column in ord_features:
                    file.write(f' {column}: {mapping}\n')

    print('List_ordinary_values:')
    print(List_ordinary_values)
    df_02_mapping=df.copy()

    for column, mapping in value_mapping.items():
        if column in cat_features:
            df_02_mapping[column] = df_02_mapping[column].map(mapping)
        elif column in ord_features:
            df_02_mapping[column] = df_02_mapping[column].map(mapping)
            
    return df_02_mapping, value_mapping


########################################## STEP 5 ##############################################################

# DROP FEATURES AND COMBINED CATEGORIES INSIDE FEATURE
#Notes:
# Dublicate information = 'dx_mikrogl','l1_mikrogl','l2_mikrogl','dx_albugem','l1_albugem','l2_albugem' - it is not relevant information for analysis
# 'l1_protg' high correlation with ''l2_protg'' 

# Main function:
def drop_features(df,min_percent = 0.06, data_type ='training', combine_categories=True, data_name='', directory_name='', drop_list = ['dx_weight', 'l1_weight', 'l2_weight', 'dx_height', 'l1_height','l2_height',	'l1_BMI'	,'l2_BMI', 'l1_klinstudie','l2_klinstudie','dx_mikrogl','l1_mikrogl','l2_mikrogl','dx_albugem','l1_albugem','l2_albugem','l1_protg']):
    if (data_type=='training'):
        #1. One unique value in column
        df, columns_to_drop = drop_one_value_features(df)
        #2. Imbalance feature drop and categories combination with small statistic (< 6%) -> Create list of combined categories per each feature
        df = imbalance_features_combination(df, min_percent = min_percent, combine_categories=combine_categories, data_name=data_name, directory_name=directory_name)
        #3. Drop all repeated values
        df.drop(columns=drop_list, axis=1, errors='ignore', inplace=True)
        #5. Drop outliers for ['dx_protg', 'l1_protg', 'l2_protg'] 
        df = outlier_remove(df, list_features =['dx_protg', 'l1_protg', 'l2_protg'] )
        
        column_names = df.columns.tolist()

# Save to JSON file
        with open('dictionaries/features_list_to_keep.json', 'w') as file:
            json.dump(column_names, file)

#If Datatype for TEST, one need to use the same features, when the model was trained            
    
    elif (data_type=='test'):

        #Open:'dictionaries/replaced_values.json'
        #    to combine categories inside each feature the same way as was done for training dataset 

        with open('dictionaries/replaced_values.json', 'r') as file:
            replaced_values = json.load(file)
         # Iterate over the DataFrame and replace values
        for feature, value_infos in replaced_values.items():
            if feature in df.columns:
                for value_info in value_infos:
                    value_to_replace = value_info['value']
                    df[feature] = df[feature].replace(value_to_replace, 10.0)

        
        #Open 'dictionaries/features_list_to_keep.json'
        #      to drop all features the same way as for training dataset
                    
        with open('dictionaries/features_list_to_keep.json', 'r') as file:
            saved_column_names = json.load(file)
            missing_columns = [col for col in saved_column_names if col not in df.columns]

        if missing_columns:
            print("Warning: The following columns are missing in 'df':", missing_columns)
            print("Retrain the model.")

        # Filter df to keep only the columns that are present in both df2 and the JSON list
        df = df[df.columns.intersection(saved_column_names)]

        # Remove outliers
        df = outlier_remove(df, list_features =['dx_protg', 'l1_protg', 'l2_protg'] )



    return df


# Subfunction for drop_features function:

# Drop columns with only one values

def drop_one_value_features(df):
    print(f'Current Size: {df.shape}')
    columns_to_drop = []
    for col in df.columns:
        # print(col, df_raw[col].dropna().nunique())
        if df[col].dropna().nunique() == 1:
            columns_to_drop.append(col)
    df.drop(columns=columns_to_drop, axis=1, errors='ignore', inplace=True)
    #df.head()
    print('Drop one value features: ')
    print(f'{len(columns_to_drop)} columns were dropped. ')
    print(f'New Size: {df.shape}')
    
    return df, columns_to_drop


# Imbalance feature drop and categories combination with small statistic (< 6%) -> Create list of combined categories per each feature
def imbalance_features_combination(df, min_percent = 0.06, combine_categories=True, data_name='', directory_name=''):
    
    imbalanced_features = []
    drop_feature_list = []
    imbalance_to_combine=[]

    df_copy = df.copy()

    _,bin_features2,cat_features2,ord_features2,ddmmyy_features2,all_cat_features2 = datatype_extraction_procedure(df)

    with open(f'{directory_name}/drop_columns_and_cat_combination_info_{data_name}.txt', 'w') as file:   
        print('Imbalance features combination:')
        print(f'Current Size: {df.shape}')

        file.write('---------------- DROP FEATURES AND COMBINED CATEGORIES INSIDE FEATURE ------------------\n')

        combined_values = {}            

        # Iterate through each categorical feature
        for feature in all_cat_features2:
            # Calculate class distribution
            '''
            Example output:
            class_distribution
                Länger als 12 Monate    746
                Bis zu 6 Monate         673
                7-12 Monate             489
            '''
            #print(feature)
            features_unique = df[feature].unique()
            #print(features_unique)
            feature_unique_count = df[feature].nunique()
            #print(feature_unique_count)
            class_distribution = df[feature].value_counts() #the count of values per group in category 

        # Check for imbalance

        
            if class_distribution.min() < min_percent * class_distribution.sum():
            
                imbalanced_features.append(feature)

                if (feature_unique_count == 2 and feature in bin_features2):
                    file.write('----------------------------------------------------------------\n')
                    file.write(f' bin: {feature}:\n {class_distribution}\n')
                    file.write(f'%:\n {class_distribution/ class_distribution.sum() * 100}\n')

                    drop_feature_list.append(feature)
                    

                
                elif feature_unique_count > 2:
                    ratio = class_distribution / class_distribution.sum() * 100
                    low_ratio_values = ratio[ratio <= min_percent*100]

                    if len(low_ratio_values) >= 2:
                        
                        file.write('----------------------------------------------------------------\n')
                        file.write(f' {feature}: \n {class_distribution}\n')
                        file.write(f'%:\n {class_distribution/ class_distribution.sum() * 100}\n')
                       
                        # To combine categories inside feature to increase statistic
                        if combine_categories:
                            combined_values[feature] = []
                            for i, r in ratio.items():
                                if i in low_ratio_values:  # Check if the current value is in the low_ratio_values
                                    
                                    if feature in cat_features2:
                                        #print(f'{feature}: {i} has ration {r} will be replaced by 10')
                                        
                                        df_copy[feature] = df_copy[feature].replace(i, 10.0)
                                        
                                        file.write(f'CAT:         10: {i} as ratio: {r}\n')
                                        imbalance_to_combine.append(feature)

                                    elif feature in ord_features2:
                                        #print(f'{feature}: {i}  has ration {r} will be replaced by 10')
                                        df_copy[feature] = df_copy[feature].replace(i, 10.0)
                                        
                                        file.write(f'ORD:         10: {i} as ratio: {r}\n')
                                        imbalance_to_combine.append(feature)
                                        
                                    
                                    combined_values[feature].append({'value': i, 'ratio_percent': r})

                        
        if combine_categories:
            with open(f'dictionaries/replaced_values.json', 'w') as file:
                json.dump(combined_values, file, indent=4)

    
        df = df_copy
        df.drop(columns = drop_feature_list,axis=1, errors='ignore', inplace=True)    
    return df

# ## Drop outliers
def outlier_remove(df, list_features=['dx_protg', 'l1_protg', 'l2_protg']):
    print(f'Remove outliers for {list_features}')
    for column_name in list_features:
        if column_name in df.columns:
            Q1 = df[column_name].quantile(0.25)
            Q3 = df[column_name].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            df[column_name] = df[column_name].apply(lambda x: x if lower_bound <= x <= upper_bound else np.nan)
        else:
            print(f"Column '{column_name}' not found in DataFrame.")
            
    return df

########################################## STEP 6 ##############################################################
# Preprocess Data

def preprocess_test_data(df_load,additional_name=''):
    num_features,bin_features,cat_features,ord_features,ddmmyy_features,all_cat_features = datatype_extraction_procedure(df_load)
    num_features.remove('l2_TTNT')
    df_load['event'] = 1
    with open(f'models/imputer_{additional_name}.pkl', 'rb') as file:
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
    with open(f'{directory_name}/imputer_{additional_name}.pkl', 'wb') as file:
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
    with open(f'{directory_name}/imputer_{additional_name}.pkl', 'wb') as file:
        pickle.dump(preprocessor, file)

    return X_train, X_test 
  





    


















