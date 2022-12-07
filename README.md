# Data Science Salary Predictor 

## Contributors and Maintainers

  - Jonah Hamilton
  - Mike Guron
  - Chen Lin
  - Tanmay Agarwal

## Motivation

As we are all current students in the MDS program, a question we have is: where will we end up working after this program is over?? A natural follow up question to this is, how much can we expect to be compensated given our previous experience, target industry, geographic location, etc. Wouldn't it be nice if we could create some sort of model that would help us gain insight into this question? Is there anything we have learnt so far in our program that could shed some light on this conundrum? Well, you have come to the right place! Our group has found a recent and comprehensive dataset processed from the Stack overflow Annual Developers Survey which we will use to build a predictive machine learning model to help answer this burning question that is on our and the rest of our cohort's mind! Read on for a breakdown of our question and an overview of our approach. 

## Research Question

Our overarching question for this project is: Can we predict the salary for data-related jobs in North America? To Achieve this, we will attempt to build a model that will predict the converted yearly compensation (in USD) for North American data-related roles given various features such as work experience, age, programming language(s), organization size, etc. Overall, in our public GitHub repository we plan to report our final results as a table containing the prediction accuracy for our models, along with an outline of the EDA process and model building procedure. 

## Data Set Source

The data set used in this project is of survey responses from individuals with data-related careers.  It was sourced from the 2022 Stack Overflow Annual Developer Survey and can be found [here](https://insights.stackoverflow.com/survey), specifically this [file](https://info.stackoverflowsolutions.com/rs/719-EMH-566/images/stack-overflow-developer-survey-2022.zip).  The overall survey features over 70,000 responses fielded from over 180 countries, examining all aspects of the developer experience from learning to code to favorite technologies for version control and workplace experience.  A detailed summary of the overall dataset can be found [here](https://survey.stackoverflow.co/2022/). 

Each row in the data set represents the survey response of an individual working in a data-related career including information on skills, experience, technologies worked with, and demographics.  Since we are interested in developing a predictor for North American Data Science Salaries the data set was subset to only include responses from individuals in Canada and the United States as a proxy for the whole of North America as the bulk of the responses are from these two countries.  

## Data Analysis Plan
Based on the primary question we are trying to answer and predict, the dataset was selected as mentioned in the previous sections.

The target prediction will be within the US and Canadian regions thus we will filter out other countries to ensure the analysis to be as precise as possible.

Also analysis need to exclude the factors which are sensitive such as gender, in order to make the analysis as unbiased as possible.

The main features columns to be used for training purposes in the analysis from the dataset are as the following:
- MainBranch
- Employment
- RemoteWork
- EdLevel
- YearCode
- YearCodePro
- DevType
- OrgSize
- Country
- LanguageHaveWorkedWith
- DatabaseHaveWorkedWith
- PlatformHaveWorkedWith
- WebframeHaveWorkedWith
- MiscTechHaveWorkedWith
- ToolTechHaveWorkedWith
- NEWCollabToolsHaveWorkedWith
- OpSysProfessional use
- VersionControlSystem
- VCInteraction
- OfficeStackAsyncHaveWorkedWith
- Age
- (Ethnicity)
- WorkExp
- Icorpm

The feature below is the target the analysis want to predict:
- CompTotal --> ConvertedCompYearly

The EDA and model selection can be done through JupyterNotebook with required libraries mentioned in the environment yaml file, by filtering the desired features, train-test dataset split, model selection, hyper parameter optimization, final model decision.

The final model should be able to predict salary for new data scientists with an accept margin of error.

## Exploratory Data Analysis
Since we want to focus on the data points for US and Canada, data scientist/analysis related jobs, we filter the dataset.

Also there are lots of null value, thus will will drop the rows with null values in ConvertedCompYearly, this reduces the size of the dataset but will make the future work easier.

## Usage

To replicate the analysis, first to clone this GitHub repository along with installing the dependencies using the [environment yaml file](/environment.yml)

### Method 1: To reproduce the analysis follow the commands below in the project root directory:

``` bash
# Downloading the raw data
	python src/data_download.py --url='https://info.stackoverflowsolutions.com/rs/719-EMH-566/images/stack-overflow-developer-survey-2022.zip' --out_file='data/raw'

 
# Use raw data to output 3 files:
# 1. data_filtered.csv
# 2. train.csv (80% from data_filtered.csv)
# 3. test.csv (20% from data_filtered.csv)
	python src/data_preprocessing.py --data_input='data/raw/survey_results_public.csv' --data_output_path='data/processed/'


# Use data_filtered.csv do EDA
# Save generated files or images
	python src/data_eda.py --data_input='data/processed/train_encoded.csv' --data_output_path='documents/figures/'

# Use train.csv do model selection, hyperparameter tuning, final model training and scoring with test.csv
# Save any file generated
	python src/data_model_selection_analysis.py --data_train='data/processed/train.csv' --data_test='data/processed/test.csv' --file_out_path='documents/results/'

# generates the final report
	Rscript -e "rmarkdown::render('documents/FinalReport.Rmd')"
```

### Method 2: Using Makefile

Run the following command at the command line/terminal in the project root directory:

```
make
```

To reset the project with cleaning file path/directory, without any intermeidate plot images or results csv files, run the following command at the command line/terminal in the project root directory:

```
make clean
```

## Licenses

The Data Science Salary Predictor materials here are licensed under the Creative Commons Attribution 4.0 International License and the MIT License.  Please provide attribution and a link to this webpage if re-using/re-mixing any of these materials.

## References

Stack Overflow (2022) *Stack Overflow Annual Developer Survey 2022* Available at: https://insights.stackoverflow.com/survey (Accessed 18 November 2022)