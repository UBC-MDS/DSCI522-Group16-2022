# Docker file for the Data Science Salary Predictor
# December 8, 2022

# use jupyter/scipy-notebook as the base image 
FROM jupyter/scipy-notebook

USER root

RUN apt-get update

# Install Python 3 packages
RUN conda install -c conda-forge --quiet --yes \
    'pandas==1.4.*' \
    'altair_saver==0.1.*' \
    'matplotlib==3.6.*' \
    'jinja2==3.1.*' \
    'imbalanced-learn==0.9.*' \
    'eli5==0.13.*' \
    'python-graphviz==0.20.*' \
    'requests==2.28.*' \
    'shap==0.41.*' \
    'pip==22.3.*' \
    'selenium==4.2.*' \
    'vega_datasets==0.9.*' \
    'scikit-learn==1.1.*' \
    'graphviz==6.0.*' \
    'pandas==1.4.*' \
    'ipykernel==6.17.*' \
    'docopt==0.6.*' \
    'pandoc==2.19.2' \
    'xgboost==1.7.*' 


# Install R packages
RUN conda install -c conda-forge --quiet --yes \
    'r-base=4.2.1' \
    'r-rmarkdown' \
    'r-tidyverse=1.3*' \
    'r-kableExtra'

# Install extra packages
RUN Rscript -e "install.packages('knitr', repos = 'http://cran.us.r-project.org')"

RUN python -m pip install vl-convert-python

