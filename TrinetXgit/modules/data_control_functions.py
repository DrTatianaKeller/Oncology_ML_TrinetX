from helper.functions.data_preprocessing_functions import *

def unique_values(df, data_type):
    data_type_list = datatype_solo_extraction(df, data_type)
    for column in df:
        if column in data_type_list:
            unique_values = df[column].unique()
            # Check if the list of unique values is not empty
            if len(unique_values) > 0:
                print(f"Column: {column} :  {df[column].dtype}")

                if data_type=='num':
                    for value in unique_values:
                        if any(char.isalpha() for char in str(value)):
                            print("Unique values:", value)
                else:
                    print("Unique values:", unique_values)