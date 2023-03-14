Folders in the project:

- app: contains a call center app that does the filtering of the useless utterances
- analysis: contains tools to track experiments and analyze results
- datasets: cleaned up datasets for training and testing
- dataload: abstraction of the datafiles for easier loading when files are moved around
- gen: generate validation and test datasets by randomly selection utterances from the chatterbox dataset and from the faqs
- notebooks: contains the notebooks used for EDA and various analysis
- resources: raw scrapings or raw provided data
- scraping: contains the scripts used to scrape the FAQ data from the web
- tests: contains some unit tests
- transcribe: contains the scripts used to transcribe the audio files using an existing service.


Installation from source:
- Install [poetry](https://python-poetry.org/docs/#installation)
- Run `poetry install` to install all the dependencies. Dependencies are listed in `pyproject.toml` and can be installed selectively by using one of the groups:
  - `ml`: to experiment and run ML models
  - `scrape`: to run scraping code
  - `dev`: to run dev tools and unit tests
  - `app`: to be able to run deno application
  - `backoffice`: to be able to run the backoffice wizard application

Running mlflow to list experiments:
- Ensure you got dependencies from `ml` group (run `poetry install --no-dev --extras ml`)
- From folder `train` run command `mlflow ui`

Running the backoffice/Setup Wizard app:
- Ensure you got dependencies from `backoffice` group (run `poetry install --no-dev --extras backoffice`)
- From root folder run command `streamlit run backoffice/0_Welcome.py`

Running the call-center app from source:
- Go into folder `app`
- Set properties in `app/config.yml` to reference the folder where model is stored and set API key for audio transcription
- Run command `python app.py` to run the server
