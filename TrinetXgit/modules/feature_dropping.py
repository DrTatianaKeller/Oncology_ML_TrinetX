'''
Dropping Features
    
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
'''
########################################## STEP 5 ##############################################################

# DROP FEATURES AND COMBINED CATEGORIES INSIDE FEATURE
#Notes:
# Dublicate information = 'dx_mikrogl','l1_mikrogl','l2_mikrogl','dx_albugem','l1_albugem','l2_albugem' - it is not relevant information for analysis
# 'l1_protg' high correlation with ''l2_protg'' 
import numpy as np
import json
from modules.datatype_extraction import *

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
        with open('../dictionaries/features_list_to_keep.json', 'w') as file:
            json.dump(column_names, file)

#If Datatype for TEST, one need to use the same features, when the model was trained            
    
    elif (data_type=='test'):

        #Open:'dictionaries/replaced_values.json'
        #    to combine categories inside each feature the same way as was done for training dataset 

        with open('../dictionaries/replaced_values.json', 'r') as file:
            replaced_values = json.load(file)
         # Iterate over the DataFrame and replace values
        for feature, value_infos in replaced_values.items():
            if feature in df.columns:
                for value_info in value_infos:
                    value_to_replace = value_info['value']
                    df[feature] = df[feature].replace(value_to_replace, 10.0)

        
        #Open 'dictionaries/features_list_to_keep.json'
        #      to drop all features the same way as for training dataset
                    
        with open('../dictionaries/features_list_to_keep.json', 'r') as file:
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
            with open(f'../dictionaries/replaced_values.json', 'w') as file:
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