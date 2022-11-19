# Data Science Salary Predictor 

## Project Proposal

Contributors:
* Jonah Hamilton
* Mike Guron
* Chen Lin
* Tanmay Agarwal

## Motivation

As we are all current students in the MDS program, a question we have is: where will we end up working after this program is over?? A natural follow up question to this is, how much can we expect to be compensated given our previous experience, target industry, geographic location, etc. Wouldn't it be nice if we could create some sort of model that would help us gain insight into this question? Is there anything we have learnt so far in our program that could shed some light on this conundrum? Well, you have come to the right place! Our group has found a recent and comprehensive dataset processed from the Stack overflow Annual Developers Survey which we will use to build a predictive machine learning model to help answer this burning question that is on our and the rest of our cohort's mind! Read on for a breakdown of our question and an overview of our approach. 

## Research Question

Our overarching question for this project is: Can we predict the salary for data-related jobs in North America? To Achieve this, we will attempt to build a model that will predict the converted yearly compensation (in USD) for North American data-related roles given various features such as work experience, age, programming language(s), organization size, etc. Overall, in our public GitHub repository we plan to report our final results as a table containing the prediction accuracy for our models, along with an outline of the EDA process and model building procedure. 

## Data Set Source

The data set used in this project is of survey responses from individuals with data-related careers.  It was sourced from the 2022 Stack Overflow Annual Developer Survey and can be found [here](https://insights.stackoverflow.com/survey), specifically this [file](https://info.stackoverflowsolutions.com/rs/719-EMH-566/images/stack-overflow-developer-survey-2022.zip).  The overall survey features over 70,000 responses fielded from over 180 countries, examining all aspects of the developer experience from learning to code to favorite technologies for version control and workplace experience.  A detailed summary of the overall dataset can be found [here](https://survey.stackoverflow.co/2022/). 

Each row in the data set represents the survey response of an individual working in a data-related career including information on skills, experience, technologies worked with, and demographics.  Since we are interested in developing a predictor for North American Data Science Salaries the data set was subset to only include responses from individuals in Canada and the United States as a proxy for the whole of North America as the bulk of the responses are from these two countries.  

## Data Analysis Plan

## Exploratory Data Analysis

## Licenses

The Data Science Salary Predictor materials here are licensed under the Creative Commons Attribution 4.0 International License and the MIT License.  Please provide attribution and a link to this webpage if re-using/re-mixing any of these materials.

## References

Stack Overflow (2022) *Stack Overflow Annual Developer Survey 2022* Available at: https://insights.stackoverflow.com/survey (Accessed 18 November 2022)