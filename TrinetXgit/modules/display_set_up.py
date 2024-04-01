# Notebook display output set up
import pandas as pd

def Set_up_option():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    #pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_colwidth', -1)