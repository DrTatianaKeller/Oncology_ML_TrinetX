
# This function extract the name to store the plot of C-index scores imputer types comparison, 
# when two imputer categories are fixed. 

from modules.name_imputer_extract import *

def name_imputer_comparison_for_plot_create(imputers_type, cat_imputer, num_imputer, scaler):
    #Extract the name of each default imputer
    cat_imputer_name = name_imputer_extract(cat_imputer)
    num_imputer_name = name_imputer_extract(num_imputer)
    scaler_name = name_imputer_extract(scaler)

    # Lists of variables
    list1 = [cat_imputer_name, num_imputer_name, scaler_name]
    list3 = ['cat', 'num', 'scale']

    # Extracting indices of variables not equal to imputers_type
    indices_not_equal_var = [index for index, value in enumerate(list3) if value != imputers_type]
    list2 = [list1[indices_not_equal_var[0]], list1[indices_not_equal_var[1]]]
    list3_new = [list3[indices_not_equal_var[0]], list3[indices_not_equal_var[1]]]
    list2

    #3) Specify the name to save the result plot
    additional_name_plot=f'variable_{imputers_type}_fixed_{list3_new[0]}{list2[0]}_{list3_new[1]}{list2[1]}'
    return additional_name_plot
