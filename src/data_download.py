# author: Chen Lin
# date: 2022-11-18

"""Downloads data zip data from the web to a local filepath as a zipped csv file.

Usage: data_download.py --url=<url> --out_file=<out_file> 
 
Options:
--url=<url>                  URL from where to download the dataset (must be in standard zip format)
--out_file=<out_file>        Specify the path including filename of where to locally write the file
"""

# Example:
# python src/data_download.py --url='https://info.stackoverflowsolutions.com/rs/719-EMH-566/images/stack-overflow-developer-survey-2022.zip' --out_file='data/raw'

import os
from docopt import docopt

import requests, zipfile, io

opt = docopt(__doc__)

def main(url, out_file):
    """
    Download the zipped data from the given url and save/un-zipped it locally to the path as scripted file_path
    
    Parameters:
    dataset_url: The raw url of the zipped dataset to be downloaded
    file_path:  File path of where to download/unzip/save data locally
    
    Returns:
    Stores the unzipped data file in the provided file_path location

    Example:
    main('https://info.stackoverflowsolutions.com/rs/719-EMH-566/images/stack-overflow-developer-survey-2022.zip', 'data/raw')
    """

    # Check if the dataset_url exists
    try: 
        request = requests.get(url)
        request.status_code == 200
    except Exception as exp:
        print(exp)
        print("dataset_url does not exist")

    # Download zip data file to file_path
    req = requests.get(url)
    zipped_file = zipfile.ZipFile(io.BytesIO(req.content))

    try:
        zipped_file.extractall(out_file)
    except:
        os.makedirs(os.path.dirname(out_file))
        zipped_file.extractall(out_file)


if __name__ == "__main__":
    main(opt["--url"], opt["--out_file"])
