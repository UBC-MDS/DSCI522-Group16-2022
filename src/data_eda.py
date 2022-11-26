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
    processed_df = pd.read_csv(data_input)
    
    #Making histogram depicting the distribution of yearly compensation
    
    yearly_comp_hist = alt.Chart(processed_df).mark_bar().encode(
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
    
    filepath = "data/processed/yearly_compensation.png"
    save_chart(yearly_comp_hist, "data/processed/yearly_compensation.png")
    
if __name__ == "__main__":    
    opt = docopt(__doc__)
    main(opt["--data_input"], opt["--data_output_path"])