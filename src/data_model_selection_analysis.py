# author: Chen Lin
# date: 2022-11-26

"""Used the filtered train data to do model selection analysis, hyperparameter tuning and best model training

Usage: data_model_selection_analysis.py --data_train=<data_train> --data_test=<data_test> --file_out_path=<file_out_path>
 
Options:
--data_train=<data_train>                    train data used for model selection, hyperparameter tuning and best model training
--data_test=<data_test>                    test data used for best model performance evaluation
--data_output_path=<data_output_path>        file path to store the processed data
"""

# Example: (call in repo root)
# python data_model_selection_analysis.py --data_train='data/processed/.csv' --data_test='data/processed/test.csv' --file_out_path='documents/'

from docopt import docopt
import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import os

cols_to_choose = [
    'MainBranch',
    'Employment',
    'RemoteWork',
    'EdLevel',
    'YearsCode',
    'YearsCodePro',
    'DevType',
    'OrgSize',
    'Country',
    'LanguageHaveWorkedWith',
    'DatabaseHaveWorkedWith',
    'PlatformHaveWorkedWith',
    'WebframeHaveWorkedWith',
    'MiscTechHaveWorkedWith',
    'ToolsTechHaveWorkedWith',
    'NEWCollabToolsHaveWorkedWith',
    'OpSysProfessional use',
    'VersionControlSystem',
    'VCInteraction',
    'OfficeStackAsyncHaveWorkedWith',
    'Age',
    'WorkExp',
    'ICorPM',
    'ConvertedCompYearly']

# order for ordinal columns
education_order = [
    'Something else', 
    'Primary/elementary school', 
    'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)',
    'Some college/university study without earning a degree',
    'Associate degree (A.A., A.S., etc.)', 
    "Bachelor’s degree (B.A., B.S., B.Eng., etc.)", 
    "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)",
    'Professional degree (JD, MD, etc.)', 
    'Other doctoral degree (Ph.D., Ed.D., etc.)']

age_order = [
    'Prefer not to say', 
    'Under 18 years old', 
    '18-24 years old',
    '25-34 years old',
    '35-44 years old',
    '45-54 years old', 
    '55-64 years old',
    '65 years or older']

numeric_cols = ['YearsCode', 'YearsCodePro', 'WorkExp']

ordinal_edu = ['EdLevel']

ordinal_age = ['Age']

binary_cols = ['MainBranch', 'Country']

categorical_cols = ['OrgSize', 'RemoteWork']

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

passthrough_cols = ['ConvertedCompYearly']

drop_cols = ['ICorPM']

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


def get_column_names_from_preporcessor(preprocessor):
    transformed_column_names = []
    for i in range(1,6):
        temp_names = preprocessor.named_transformers_['pipeline-'+str(i)].get_feature_names_out().tolist()
        transformed_column_names += temp_names

    for i in range(1,14):
        temp_names = preprocessor.named_transformers_['countvectorizer-'+str(i)].get_feature_names_out().tolist()
        # print(temp_names)
        for name in temp_names:
            name = multianswer_cols[i-1] + "_" + name
            transformed_column_names.append(name)

    transformed_column_names.append('ConvertedCompYearly')
    # print(transformed_column_names)

    return transformed_column_names


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
    preprocess_data('data/raw/survey_results_public.csv', 'data/processed/')
    """

    if(not os.path.exists(data_output_path)):
        os.makedirs(os.path.dirname(data_output_path))

    # read raw data
    df_raw = pd.read_csv(data_input)
    # print(df_raw.columns)

    # filter data 
    north_america_data = df_raw.query("Country == 'United States of America' or Country == 'Canada'")
    north_america_data = north_america_data[cols_to_choose]
    north_america_data= north_america_data.query('ConvertedCompYearly < 500000')

    # TSave the filter data as 'data/processed/filtered_data.csv'
    north_america_data.to_csv(data_output_path + 'filtered_data.csv', index=False)
    print("Successfully saved filtered data into {}".format(data_output_path))
    
    df_filtered = north_america_data
    df_filtered = write_na_values_for_cols(df_filtered, multianswer_cols)

    train_df_filtered, test_df_filtered = train_test_split(df_filtered, test_size=0.10, random_state=522)

    # converts string year values to float
    train_df_filtered['YearsCode'] = train_df_filtered['YearsCode'].apply(lambda x: convert2float(x))
    train_df_filtered['YearsCodePro'] = train_df_filtered['YearsCodePro'].apply(lambda x: convert2float(x))

    test_df_filtered['YearsCode'] = test_df_filtered['YearsCode'].apply(lambda x: convert2float(x))
    test_df_filtered['YearsCodePro'] = test_df_filtered['YearsCodePro'].apply(lambda x: convert2float(x))

    # train_df_filtered = write_na_values_for_cols(train_df_filtered, multianswer_cols)
    # test_df_filtered = write_na_values_for_cols(test_df_filtered, multianswer_cols)

    train_df_filtered.to_csv(data_output_path + 'train.csv', index=False)
    test_df_filtered.to_csv(data_output_path + 'test.csv', index=False)

    numeric_transformer = make_pipeline(SimpleImputer(strategy='most_frequent'), StandardScaler())

    ordinal_edu_transformer = make_pipeline(OrdinalEncoder(categories=[education_order], dtype=int))

    ordinal_age_transformer = make_pipeline(SimpleImputer(strategy='most_frequent'), OrdinalEncoder(categories=[age_order], dtype=int))

    binary_transformer = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(drop='if_binary', handle_unknown='ignore', dtype=int))

    categorical_transformer = make_pipeline(OneHotEncoder(handle_unknown='ignore', sparse=False))

    preprocessor = make_column_transformer(
            (numeric_transformer, numeric_cols),
            (ordinal_edu_transformer, ordinal_edu),
            (ordinal_age_transformer, ordinal_age),
            (binary_transformer, binary_cols),
            (categorical_transformer, categorical_cols),
            ('passthrough', passthrough_cols),
            ('drop', drop_cols),
            (CountVectorizer(tokenizer=lambda text: text.split(';')), multianswer_cols[0]),
            (CountVectorizer(tokenizer=lambda text: text.split(';')), multianswer_cols[1]),
            (CountVectorizer(tokenizer=lambda text: text.split(';')), multianswer_cols[2]),
            (CountVectorizer(tokenizer=lambda text: text.split(';')), multianswer_cols[3]),
            (CountVectorizer(tokenizer=lambda text: text.split(';')), multianswer_cols[4]),
            (CountVectorizer(tokenizer=lambda text: text.split(';')), multianswer_cols[5]),
            (CountVectorizer(tokenizer=lambda text: text.split(';')), multianswer_cols[6]),
            (CountVectorizer(tokenizer=lambda text: text.split(';')), multianswer_cols[7]),
            (CountVectorizer(tokenizer=lambda text: text.split(';')), multianswer_cols[8]),
            (CountVectorizer(tokenizer=lambda text: text.split(';')), multianswer_cols[9]),
            (CountVectorizer(tokenizer=lambda text: text.split(';')), multianswer_cols[10]),
            (CountVectorizer(tokenizer=lambda text: text.split(';')), multianswer_cols[11]),
            (CountVectorizer(tokenizer=lambda text: text.split(';')), multianswer_cols[12])
    )

    # fit preprocessor with train df
    train_df_filtered_encode = preprocessor.fit_transform(train_df_filtered).todense()

    # creates list of new column names from preprocessing pipelines for train df
    transformed_column_names_train = get_column_names_from_preporcessor(preprocessor)

    # fit preprocessor with test df
    test_df_filtered_encode = preprocessor.fit_transform(test_df_filtered).todense()

    # creates list of new column names from preprocessing pipelines for test df
    transformed_column_names_test = get_column_names_from_preporcessor(preprocessor)
    

    train_enc = pd.DataFrame(
        data=train_df_filtered_encode, 
        index=train_df_filtered.index, 
        columns=transformed_column_names_train
    )

    test_enc = pd.DataFrame(
        data=test_df_filtered_encode, 
        index=test_df_filtered.index, 
        columns=transformed_column_names_test
    )

    train_enc.to_csv(data_output_path + 'train_encoded.csv', index=False)
    test_enc.to_csv(data_output_path + 'test_encoded.csv', index=False)

if __name__ == "__main__":
    args = docopt(__doc__)

    data_input = args["--data_input"]
    data_output_path = args["--data_output_path"]

    preprocess_data(data_input, data_output_path)

    print("Successfully filter and split raw data into {}".format(data_output_path))