'''
 Data mapping:
    1. Replace 'Ja'/ 'Nein'/ 'Nicht getestet' to 1/ 0/ NaN (99) numerical value
    2. Mapping object type data:
        1. Sorting by number in the line:
            Example: 
            '<3,5 mg/L': 1, '3,5 - 5,5 mg/L': 2, '>5,5 mg/L': 3
        2. Sorting by Stadium I, II, III: giving indexes of mapping 1, 2, 3
        3. For Unbekant Stadium: index 0
'''
########################################## STEP 4 ##############################################################
# # Data mapping
import numpy as np
import pandas as pd
import re
from modules.datatype_extraction import *

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
