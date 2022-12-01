# author: Chen Lin
# date: 2022-11-26

"""Used the filtered train data to do model selection analysis, hyperparameter tuning and best model training

Usage: data_model_selection_analysis.py --data_train=<data_train> --data_test=<data_test> --file_out_path=<file_out_path>
 
Options:
--data_train=<data_train>                    train data used for model selection, hyperparameter tuning and best model training
--data_test=<data_test>                      test data used for best model performance evaluation
--file_out_path=<file_out_path>              file path to store the results on model selection, hyperparameter tuning and best model training
"""

# Example: (call in repo root)
# python src/data_model_selection_analysis.py --data_train='data/processed/train.csv' --data_test='data/processed/test.csv' --file_out_path='documents/results/'

from docopt import docopt
import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import RandomizedSearchCV

import os

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

# passthrough_cols = ['ConvertedCompYearly']

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

    # transformed_column_names.append('ConvertedCompYearly')
    # print(transformed_column_names)

    return transformed_column_names

def corss_validate_result(model_name, model_type, preprocessor, cross_val_results_reg, X_train, y_train, score_types_reg):
    pipe = make_pipeline(
        preprocessor,
        model_type
    )
    cross_val_results_reg[model_name] = pd.DataFrame(cross_validate(pipe,
                                                                    X_train,
                                                                    y_train, 
                                                                    return_train_score=True, 
                                                                    scoring=list(score_types_reg.values()))).agg(['mean', 'std']).round(3).T

# Referenced from 571 lab 4
def mean_std_cross_val_scores(model, X_train, y_train, **kwargs):
    """
    Returns mean and std of cross validation

    Parameters
    ----------
    model :
        scikit-learn model
    X_train : numpy array or pandas DataFrame
        X in the training data
    y_train :
        y in the training data

    Returns
    ----------
        pandas Series with mean scores from cross_validation
    """

    scores = cross_validate(model, X_train, y_train, **kwargs)

    mean_scores = pd.DataFrame(scores).mean()
    std_scores = pd.DataFrame(scores).std()
    out_col = []

    for i in range(len(mean_scores)):
        out_col.append((f"%0.3f (+/- %0.3f)" % (mean_scores[i], std_scores[i])))

    return pd.Series(data=out_col, index=mean_scores.index)

def model_selection_analysis(data_train, data_test, file_out_path):
    """
    Load the train and test data.
    Perform model selection analysis, hyperparameter tuning and best model training
    
    Parameters:
    data_train: train data used for model selection, hyperparameter tuning and best model training
    data_test: test data used for best model performance evaluation
    file_out_path:  file path to store the results on model selection, hyperparameter tuning and best model training
    
    Returns:
    None, just save related file from model selection, hyperparameter tuning and best model training

    Example:
    model_selection_analysis('data/processed/train.csv', 'data/processed/test.csv', 'documents/results/')
    
    """

    if(not os.path.exists(file_out_path)):
        os.makedirs(os.path.dirname(file_out_path))

    # read raw data
    train_df = pd.read_csv(data_train)
    test_df = pd.read_csv(data_test)

    train_df = write_na_values_for_cols(train_df, multianswer_cols)

    # converts string year values to float
    train_df['YearsCode'] = train_df['YearsCode'].apply(lambda x: convert2float(x))
    train_df['YearsCodePro'] = train_df['YearsCodePro'].apply(lambda x: convert2float(x))

    X_train = train_df.drop(columns=["ConvertedCompYearly"])
    y_train = train_df["ConvertedCompYearly"]

    X_test = test_df.drop(columns=["ConvertedCompYearly"])
    y_test = test_df["ConvertedCompYearly"]

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
            # ('passthrough', passthrough_cols),
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

    # fit preprocessor with test df
    X_train_encode = preprocessor.fit_transform(X_train).todense()

    # creates list of new column names from preprocessing pipelines for test df
    transformed_column_names = get_column_names_from_preporcessor(preprocessor)
    
    X_train_enc = pd.DataFrame(
        data=X_train_encode, 
        index=X_train.index, 
        columns=transformed_column_names
    )

    print('Successfully make encoded X_train')

    # Feature Selection
    select_lr = SelectFromModel(Ridge(), threshold="0.8*mean")

    pipe_rf_model_based = make_pipeline(
        preprocessor, select_lr, RandomForestRegressor(random_state=522)
    )
    pipe_rf_model_based.fit(X_train, y_train)

    model_based_mask = pipe_rf_model_based.named_steps["selectfrommodel"].get_support()
    mb_selected_feats = X_train_enc.columns[model_based_mask]
    fs = mb_selected_feats.tolist()

    #TODO: Save selected features

    print('Successfully done feature selection')

    # Model Selection
    cross_val_results_reg = {}
    # cross_val_results_reg_fs = {}

    models = {
        "Baseline": DummyRegressor(),
        "KNN Regressor": KNeighborsRegressor(),
        "Ridge": Ridge(),
        "Random Forest Regressor": RandomForestRegressor(),
        "Lasso": Lasso(),
    }

    score_types_reg = {
        #"neg_mean_squared_error": "neg_mean_squared_error",
        #"neg_root_mean_squared_error": "neg_root_mean_squared_error",
        "neg_mape": "neg_mean_absolute_percentage_error", 
        "r2": "r2",
    }

    for model_item in models.items():
        model_name = model_item[0]
        model_type = model_item[1]
        corss_validate_result(model_name, model_type, preprocessor, cross_val_results_reg, X_train, y_train, score_types_reg)

    table_2 = pd.concat(
    {key: pd.DataFrame(value) for key, value in cross_val_results_reg.items()}, 
    axis=1)
    table_2.to_csv(file_out_path + "model_accuracies.csv")

    table_3 = mean_std_cross_val_scores(
    pipe_rf_model_based, X_train, y_train, return_train_score=True
    )
    table_3.to_frame().T.to_csv(file_out_path + "rf_result_with_feature_selection.csv")

    print('Successfully done model analysis')

    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {
                   'randomforestregressor__max_features': max_features,
                   'randomforestregressor__max_depth': max_depth,
                   'randomforestregressor__bootstrap': bootstrap
    }
    
    pipe_rf = make_pipeline(preprocessor, RandomForestRegressor(random_state=522))
    
    rf_random = RandomizedSearchCV(pipe_rf,
                   param_distributions=random_grid,
                   n_iter=10,
                   n_jobs=-1,
                   scoring='r2',
                   return_train_score=True)
    rf_random.fit(X_train, y_train)
    pd.DataFrame(rf_random.best_params_, index=[0]).to_csv(file_out_path + "best_params.csv")
    pd.DataFrame({'best_random_forest_train_score':rf_random.best_score_}, index=[0]).to_csv(file_out_path + "validation_score.csv") 
    print('Successfully done hyperparameter tunning')

    # TODO: Best model training and performance analysis
    pd.DataFrame({'best_random_forest_test_score':rf_random.score(X_test, y_test)}, index=[0]).to_csv(file_out_path + "test_score.csv")

    print('Successfully done model selection')
    



if __name__ == "__main__":
    args = docopt(__doc__)

    data_train = args["--data_train"]
    data_test = args["--data_test"]
    file_out_path = args["--file_out_path"]

    model_selection_analysis(data_train, data_test, file_out_path)

    print("Successfully done model selection and related files are in {}".format(file_out_path))