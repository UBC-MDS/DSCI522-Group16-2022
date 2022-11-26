# Authors: Jonah Hamilton and Mike Guron
# Date: 2022-11-25

"""A script that creates EDA plots and tables for the pre-processed training data from the Stack Overflow Annual Developer Survey 2022 data (from https://info.stackoverflowsolutions.com/rs/719-EMH-566/images/stack-overflow-developer-survey-2022.zip). This script saves the tables as csv files and the images as png files and stores them under...

Usage: src/data_eda.py --data_input=<data_input> --data_output_path=<data_output_path>

Options:
--data_input=<data_input>                       file path to the encoded training data
--data_output_path=<data_output_path>           File path to directory where the generated figures will be saved
"""

# Example:
# src/data_eda.py --data_input= 'data/processed/train_df_filtered.csv' --data_output_path=<data/processed>

import altair as alt
import pandas as pd
from docopt import docopt
import os
alt.data_transformers.enable('data_server')
alt.renderers.enable('mimetype')
import vl_convert as vlc

def save_chart(chart, filename, scale_factor=1):
    '''
    Save an Altair chart using vl-convert
    
    Parameters
    ----------
    chart : altair.Chart
        Altair chart to save
    filename : str
        The path to save the chart to
    scale_factor: int or float
        The factor to scale the image resolution by.
        E.g. A value of `2` means two times the default resolution.
    '''
    with alt.data_transformers.enable("default") and alt.data_transformers.disable_max_rows():
        if filename.split('.')[-1] == 'svg':
            with open(filename, "w") as f:
                f.write(vlc.vegalite_to_svg(chart.to_dict()))
        elif filename.split('.')[-1] == 'png':
            with open(filename, "wb") as f:
                f.write(vlc.vegalite_to_png(chart.to_dict(), scale=scale_factor))
        else:
            raise ValueError("Only svg and png formats are supported")
            
def main (data_input, data_output_path):
    train_encoded = pd.read_csv(data_input)
    
    # making histogram depicting the distribution of yearly compensation
    
    yearly_comp_hist = alt.Chart(train_encoded).mark_bar().encode(
        x=alt.X('ConvertedCompYearly',
                bin=alt.Bin(maxbins=30), 
                axis=alt.Axis(format='$.0f',
                              title="Yearly Compensation (USD)")),
        y=alt.Y('count()', title="Count of Salaries")
    ).properties(
        title="Distribution of Yearly Compensation",
        width=600,
        height=400
    )
    
    # saving chart as a png file
    filepath = os.path.join(data_output_path, "yearly_compensation.png")
    save_chart(yearly_comp_hist, filepath)

    country_order = ["Canada","USA" ]

    education_order = [
        'Something else', 
        'Primary school', 
        'Secondary school',
        'Some college/uni',
        'Associate degree', 
        "Bachelor’s degree", 
        "Master’s degree",
        'Professional degree', 
        'Other doctoral degree']

    main_branch_order = ["Developer by profession", "Not primarily a developer",]

    age_order = [
        'Prefer not to say', 
        'Under 18', 
        '18-24',
        '25-34',
        '35-44',
        '45-54', 
        '55-64',
        '65+']  


    column_lables = {'EdLevel': education_order,
                'MainBranch': main_branch_order,
                'Age': age_order,
                'Country' : country_order }

    plot_lables = {}

    # creates dictionary of column lables for the plots
    for column, lables in column_lables.items():
        plot_lables[column] = ""
        for i in range(len(lables)):
            plot_lables[column] += f"datum.label == {i} ? '{lables[i]}' : "
        plot_lables[column] += "'Unknown'"

    # creates individual box plots

    edu_boxplot = alt.Chart(train_encoded).mark_boxplot().encode(
                alt.X('ConvertedCompYearly', title ="Yearly Compensation (USD)", axis=alt.Axis(format='$~s')),
                alt.Y('EdLevel:O', title="Education Level", axis=alt.Axis(labelExpr=plot_lables['EdLevel'])),
                alt.Color('EdLevel', legend=None)
                )

    age_boxplot = alt.Chart(train_encoded).mark_boxplot().encode(
                alt.X('ConvertedCompYearly', title ="Yearly Compensation (USD)", axis=alt.Axis(format='$~s')),
                alt.Y('Age:O', title="Age", axis=alt.Axis(labelExpr=plot_lables['Age'])),
                alt.Color('Age', legend=None)   
                )
    mainbranch_boxplot = alt.Chart(train_encoded).mark_boxplot().encode(
                alt.X('ConvertedCompYearly', title ="Yearly Compensation (USD)", axis=alt.Axis(format='$~s')),
                alt.Y('MainBranch_I am not primarily a developer, but I write code sometimes as part of my work:O', title="MainBranch", axis=alt.Axis(labelExpr=plot_lables['MainBranch'])),
                alt.Color('MainBranch_I am not primarily a developer, but I write code sometimes as part of my work', legend=None)
                )

    country_boxplot = alt.Chart(train_encoded).mark_boxplot().encode(
                alt.X('ConvertedCompYearly', title ="Yearly Compensation (USD)", axis=alt.Axis(format='$~s')),
                alt.Y('Country_United States of America:O', title="Country", axis=alt.Axis(labelExpr=plot_lables['Country'])),
                alt.Color('Country_United States of America', legend=None)
                )

    # combines plots into final figure
    final_boxplot = ((mainbranch_boxplot | country_boxplot) & (edu_boxplot |  age_boxplot)).properties(title=alt.TitleParams(
                text='Yearly Compensation Distribuions',
                subtitle='Developers from the USA seem to make more but there is not a consisten trend for age and edcation level',
                fontSize=20,
                anchor='middle'))

    filepath = os.path.join(data_output_path, "final_boxplot.png")
    save_chart(final_boxplot, filepath )
    
    # list of features to use in correlation table
    corr_features = [
    "YearsCode",
    "YearsCodePro",
    "WorkExp",
    "ConvertedCompYearly"]
    
    # subset input data to list of features needed for correlation
    corr_df = train_encoded[corr_features]
    corr_table = corr_df.corr()
    
    # save table as csv file
    filepath = os.path.join(data_output_path, "correlation_table.csv")
    corr_table.to_csv(filepath)



if __name__ == "__main__":    
    opt = docopt(__doc__)
    main(opt["--data_input"], opt["--data_output_path"])