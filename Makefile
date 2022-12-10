# Data Science Salary Predictor data pipe
# author: UBC MDS Cohort7 DSCI522 Group16 members
# date: 2022-11-30

all: documents/FinalReport.pdf

# download data
data/raw/survey_results_public.csv data/raw/survey_results_schema.csv data/raw/README_2022.txt data/raw/so_survey_2022.pdf: src/data_download.py
	python src/data_download.py --url='https://info.stackoverflowsolutions.com/rs/719-EMH-566/images/stack-overflow-developer-survey-2022.zip' --out_file='data/raw'

# Use raw data to output 5 files:
# 1. data_filtered.csv
# 2. train.csv (90% from data_filtered.csv)
# 3. test.csv (10% from data_filtered.csv)
# 4. train_encoded.csv (encoded/preprocessed data from train.csv)
# 5. test_encoded.csv (encoded/preprocessed data from test.csv)
data/processed/filtered_data.csv data/processed/train.csv data/processed/test.csv data/processed/train_encoded.csv data/processed/test_encoded.csv: src/data_preprocessing.py data/raw/survey_results_public.csv
	python src/data_preprocessing.py --data_input='data/raw/survey_results_public.csv' --data_output_path='data/processed/'

# Use train_encoded.csv do EDA
# Save generated files or images in documents folder
documents/figures/yearly_compensation.png documents/figures/final_boxplot.png documents/results/correlation_table.csv: src/data_eda.py data/processed/train_encoded.csv
	python src/data_eda.py --data_input='data/processed/train_encoded.csv' --data_output_path='documents/figures/'

# Use train.csv do model selection, hyperparameter tuning, final model traning and scoring with test.csv
# Use test.csv to evaluate model performance
# Save any file generated in documents folder
documents/results/model_accuracies.csv documents/results/rf_result_with_feature_selection.csv documents/results/best_params.csv documents/results/validation_score.csv documents/results/test_score.csv: src/data_model_selection_analysis.py data/processed/train.csv data/processed/test.csv
	python src/data_model_selection_analysis.py --data_train='data/processed/train.csv' --data_test='data/processed/test.csv' --file_out_path='documents/results/'

# render final report
documents/FinalReport.pdf: documents/FinalReport.Rmd documents/references.bib data/raw/survey_results_schema.csv documents/figures/yearly_compensation.png documents/figures/final_boxplot.png documents/results/correlation_table.csv documents/results/model_accuracies.csv documents/results/rf_result_with_feature_selection.csv documents/results/best_params.csv documents/results/validation_score.csv documents/results/test_score.csv
	Rscript -e "rmarkdown::render('documents/FinalReport.Rmd')"

clean: 
	rm -rf data
	rm -rf documents/FinalReport.pdf
	rm -rf documents/figures
	rm -rf documents/results