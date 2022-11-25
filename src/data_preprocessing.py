# author: Chen Lin
# date: 2022-11-18

"""Filter data and split data into train and test sets.

Usage: data_preprocessing.py --data_input=<data_input> --data_output_path=<data_output_path>
 
Options:
--data_input=<data_input>                    raw data to be filtered and splitted into train and test sets.
--data_output_path=<data_output_path>        file path to store the processed data
"""

# Example:
# python src/data_preprocessing.py --data_input='data/raw/survey_results_public.csv' --data_output_path='data/processed/'

from docopt import docopt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import requests, os

multianswer_cols = [
'DevType',
'LanguageHaveWorkedWith',
'DatabaseHaveWorkedWith',
'PlatformHaveWorkedWith',
'WebframeHaveWorkedWith',
'MiscTechHaveWorkedWith',
'ToolsTechHaveWorkedWith',
'NEWCollabToolsHaveWorkedWith',
'OpSysProfessional use',
'VCInteraction',
'VersionControlSystem',
'OfficeStackAsyncHaveWorkedWith',
'Employment']

def write_na_values_for_cols(df, list_of_cols, fillna_base_text = 'unspecified'):
    temp_df = df.copy()
    for col in list_of_cols:
        fillna_text_final = col + "_" + fillna_base_text
        temp_df[col] = temp_df[col].fillna(fillna_text_final)
        
    return temp_df

# Returns float values for different string inputs
def convert2float(x):
    if  x == 'More than 50 years' :
        return float(50)
    elif x == 'Less than 1 year':
        return float(0)
    else:
        return float(x)


def preprocess_data(data_input, data_output_path):
    """
    Load the raw data.
    Filter data and split data into train and test sets.
    
    Parameters:
    data_input: raw data with its file path
    data_output_path:  file path where to store the preprocessed data
    
    Returns:
    None, just save train.csv and test.csv

    Example:
    preprocess_data('../data/raw/survey_results_public.csv', '../data/processed/')
    """

    if(not os.path.exists(data_output_path)):
        os.makedirs(os.path.dirname(data_output_path))

    # read raw data
    # df_raw = pd.read_csv(data_input)

    # filter data 
    # TODO: Add filter steps from Tanmay (plus .query('ConvertedCompYearly < 500000'))
    # TODO: Aave the filter data as '..data/processed/filtered_data.csv'

    df_filtered = pd.read_csv('data/processed/filtered_data.csv')

    train_df_filtered, test_df_filtered = train_test_split(df_filtered, test_size=0.10, random_state=522)

    # converts string year values to float
    train_df_filtered['YearsCode'] = train_df_filtered['YearsCode'].apply(lambda x: convert2float(x))
    train_df_filtered['YearsCodePro'] = train_df_filtered['YearsCodePro'].apply(lambda x: convert2float(x))
    test_df_filtered['YearsCode'] = test_df_filtered['YearsCode'].apply(lambda x: convert2float(x))
    test_df_filtered['YearsCodePro'] = test_df_filtered['YearsCodePro'].apply(lambda x: convert2float(x))

    train_df_filtered = write_na_values_for_cols(train_df_filtered, multianswer_cols)

    train_df_filtered.to_csv(data_output_path + 'train.csv')
    test_df_filtered.to_csv(data_output_path + 'test.csv')

if __name__ == "__main__":
    args = docopt(__doc__)

    data_input = args["--data_input"]
    data_output_path = args["--data_output_path"]

    preprocess_data(data_input, data_output_path)

    print("Successfully filter and split raw data into {}".format(data_output_path))