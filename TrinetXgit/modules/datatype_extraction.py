'''
2. Extract features datatype and create datatype lists from index_list_num_cat_ord_bin.csv
    - data type extraction function for one type of data: 
            datatype_solo_extraction(df, type_name, path_list="dictionaries/")
    - data type extraction function for all types of data:
            datatype_extraction_procedure(df, path_list="dictionaries/", report = 0)

'''
########################################## STEP 2 ##############################################################
import numpy as np
import pandas as pd

# data type extraction function for one type of data. 
def datatype_solo_extraction(df, type_name, path_list="../dictionaries/"):
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

# data type extraction function for all types of data. 
# return (num_features,bin_features,cat_features,ord_features,ddmmyy_features,all_cat_features)
def datatype_extraction_procedure(df, path_list="../dictionaries/", report = 0):
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
    num_features = data_type_list(df, "num")
    bin_features = data_type_list(df, "bin")
    ddmmyy_features = data_type_list(df, "date")
    cat_features = data_type_list(df, "cat")
    ord_features = data_type_list(df, "ord")
    all_cat_features = data_type_list(df, "cat","ord","bin")

    if report == 1:  
        print("Control output of used datatype: ")
        
        print("numerical: ")
        print(num_features)

        print("bin: ")
        print(bin_features)
       
        print("date: ")
        print(ddmmyy_features)
        
        print("categorized: ")
        print(cat_features)
       
        print("ordered: ")
        print(ord_features)

        print("all cat.: ")
        print(all_cat_features)

    return (num_features,bin_features,cat_features,ord_features,ddmmyy_features,all_cat_features)