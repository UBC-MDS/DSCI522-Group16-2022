# author: Chen Lin
# date: 2022-11-18

"""Downloads data csv data from the web to a local filepath as a csv.

Usage: download_data.py --dataset_url=<dataset_url>
 
Options:
--dataset_url=<dataset_url>             URL from where to download the dataset (must be in standard zip format)
"""

import os
import pandas as pd
from docopt import docopt

import requests, zipfile, io

opt = docopt(__doc__)

file_path = 'data/raw'

def main(url):

    # Download zip file to out_file path
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    try:
        z.extractall(file_path)
    except:
        os.makedirs(os.path.dirname(file_path))
        z.extractall(file_path)


if __name__ == "__main__":
    main(opt["--dataset_url"])
