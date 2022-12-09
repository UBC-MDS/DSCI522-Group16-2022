# Makefile
# Data Science Salary Predictor data pipe
# author: UBC MDS Cohort7 DSCI522 Group16 members
# date: 2022-11-30

all : documents/FinalReport.pdf

# download data

survey_results_public.csv survey_results_schema.csv README_2022.txt so_survey_2022.pdf : src/data_download.py
	python src/data_download.py --url='https://info.stackoverflowsolutions.com/rs/719-EMH-566/images/stack-overflow-developer-survey-2022.zip' --out_file='data/raw'

# Use raw data to output 5 files:
# 1. data_filtered.csv
# 2. train.csv (90% from data_filtered.csv)
# 3. test.csv (10% from data_filtered.csv)
# 4. train_encoded.csv (encoded/preprocessed data from train.csv)
# 5. test_encoded.csv (encoded/preprocessed data from test.csv)
filtered_data.csv train.csv test.csv train_encoded.csv test_encoded.csv : src/data_preprocessing.py survey_results_public.csv
	python src/data_preprocessing.py --data_input='data/raw/survey_results_public.csv' --data_output_path='data/processed/' 

# Use train_encoded.csv do EDA
# Save generated files or images in documents folder
yearly_compensation.png final_boxplot.png correlation_table.csv : src/data_eda.py train_encoded.csv
	python src/data_eda.py --data_input='data/processed/train_encoded.csv' --data_output_path='documents/figures/'

# Use train.csv do model selection, hyperparameter tuning, final model traning and scoring with test.csv
# Use test.csv to evaluate model performance
# Save any file generated in documents folder
model_accuracies.csv rf_result_with_feature_selection.csv best_params.csv validation_score.csv test_score.csv : src/data_model_selection_analysis.py train.csv test.csv
	python src/data_model_selection_analysis.py --data_train='data/processed/train.csv' --data_test='data/processed/test.csv' --file_out_path='documents/results/'

# render final report
documents/FinalReport.pdf : documents/FinalReport.Rmd documents/references.bib survey_results_schema.csv yearly_compensation.png final_boxplot.png correlation_table.csv model_accuracies.csv rf_result_with_feature_selection.csv best_params.csv validation_score.csv test_score.csv
	Rscript -e "rmarkdown::render('documents/FinalReport.Rmd')"

clean : 
	rm -rf data
	rm -rf documents/figures
	rm -rf documents/results
	rm -rf documents/FinalReport.pdf

