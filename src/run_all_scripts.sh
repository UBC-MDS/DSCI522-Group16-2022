python data_download.py --url='https://info.stackoverflowsolutions.com/rs/719-EMH-566/images/stack-overflow-developer-survey-2022.zip'
 --out_file='../data/raw'
 
# Use raw data to output 5 files:
# 1. data_filtered.csv
# 2. train.csv (90% from data_filtered.csv)
# 3. test.csv (10% from data_filtered.csv)
# 4. train_encoded.csv (encoded/preprocessed data from train.csv)
# 5. test_encoded.csv (encoded/preprocessed data from test.csv)
python data_preprocessing.py --data_input='../data/raw/survey_results_public.csv' --data_output_path='../data/processed/'

# Use data_filtered.csv do EDA
# Save generated files or images
python data_eda.py --data_input='../data/processed/train_encoded.csv' -file_out_path='../documents/' 

# Use train.csv do model selection, hyperparameter tuning, final model traning and scoring with test.csv
# Save any file generated
python data_model_selection_analysis.py --data_train='../data/processed/train.csv' --data_test='../data/processed/test.csv' --file_out_path='../documents/'