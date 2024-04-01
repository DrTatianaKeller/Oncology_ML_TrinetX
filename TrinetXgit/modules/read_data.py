'''
1. Read dataframe and Create directory in 'data_information/' to save data information

'''
########################################## STEP 1 ##############################################################

import pandas as pd

# Read Dataframe
def read_df(data_name, data_path ='../data/raw/' ):
    data_path = f'{data_path}{data_name}.csv'
    df = pd.read_csv(data_path)
    if 'patient_id' in df.columns:
        df = df.drop(columns='patient_id')
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns='Unnamed: 0')
    return df



