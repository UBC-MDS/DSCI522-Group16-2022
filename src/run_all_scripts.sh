python data_download.py --url='https://info.stackoverflowsolutions.com/rs/719-EMH-566/images/stack-overflow-developer-survey-2022.zip'
 --out_file='data/raw'
 
# Use raw data to output 3 files:
# 1. data_filtered.csv
# 2. train.csv (80% from data_filtered.csv)
# 3. test.csv (20% from data_filtered.csv)
python data_preprocessing.py --data_input='../data/raw/survey_results_public.csv' --data_output_path='../data/processed/'

# Use data_filtered.csv do EDA
# Save generated files or images
python data_eda.py --data_input='../data/processed/train.csv' -file_out_path='../documents/' 

# Use train.csv do model selection, hyperparameter tuning, final model traning and scoring with test.csv
# Save any file generated
python data_model_selection_analysis.py --data_train='../data/processed/train.csv' --data_test='../data/processed/test.csv' --file_out_path='../documents/'