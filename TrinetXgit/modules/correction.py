'''
Data correction:
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
'''

########################################## STEP 3 ##############################################################
# Data Correction
import numpy as np
import pandas as pd
from modules.datatype_extraction import *

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