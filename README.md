
# 0. Pre-requisites

To install from source:
- Have python `3.10` or above ready
- Install [poetry](https://python-poetry.org/docs/#installation)


# 1. Running the various webapps

Following webapps can be run:
- **backoffice**: Setup Wizard app that allows the customer to setup the call center app
- **app**: call center app that does the filtering of the useless utterances
- **mlflow**: MLFlow can be run to view the results of the experiments

## Run experiments and view results using MLFlow
- At the root of the project run command `poetry install --only main,ml` to install the
minimal set of dependencies.
- From folder `train` run command `mlflow ui`
- Go to http://localhost:5000 to see the MLFlow UI web app

## Running the backoffice/Setup Wizard app
- Ensure you got dependencies from `backoffice` group `poetry install --only main,ml,backoffice`
- From root folder run command `streamlit run backoffice/0_Welcome.py`
- Go to http://localhost:8001 to see the app


## Running the call-center front-office app
- Go into folder `app`
- Set properties in `app/.env` and set the 2 properties:
  - the reference to the folder where chosen model is stored 
  - and set API key for audio transcription
- Run command `python main.py` to run the server
- Go to http://localhost:8002 to see the app
- You can directly speek to the browser or choose to type text in text-box at the bottom of the page

## Running notebooks
- Ensure you got dependencies from `notebooks` group `poetry install --only main,ml,notebooks`


# 3. Folder structure
Folders in the project:

- **app**: contains a call center app that does the filtering of the useless utterances
- analysis: contains tools to track experiments and analyze results
- **backoffice**: contains the wizard app that allows the customer to setup the call center app
- datasets: cleaned up datasets for training and testing
- dataload: abstract location of datafiles for easier loading when files are moved around
- notebooks: contains the notebooks used for EDA and various analysis
- resources: raw scrapings or raw provided data
- scraping: contains the scripts used to scrape the FAQ data from the web
- tests: contains some unit tests
- **train**: contains the scripts used to train the ML models





