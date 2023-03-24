# 🪄 Call-Center conversation filter

This is project is a conversation Filter🪄application that helps Call-Centers to filter
live-conversations between an employee and a customer. It tells
which sentences are relevant to their domain (represented as some FAQs they upload) and which are likely not.

In general, Call-Centers already own a question-answering software that searches their
FAQs or Knowledge Base. However, those systems are slow and cannot afford to run on each
and every utterance spoken by the customer or the employee. It is here that the
Fantastic Filter comes in:
it is a fast pre-filtering on the conversation that keeps only sentences that are likely
to be relevant to the downstream question-answering system.

This backoffice application will guide you through the setup of the filter for your
Call-Center.
The frontoffice is an audio chat application that filters the conversation in real-time.

For this demo app, we only allow a pre-defined set of FAQs that are listed below. Each
of those have been scraped from the internet. It is easy to use the code with your own knowledge base instead.

# 1. 🐳 Running docker images

Docker images for front-office and back-office are available on docker hub for ARM and
x86 architectures:

- front-office: `docker pull jlinho/aajk:frontoffice` and run
  with `docker run -p 8002:8002 jlinho/aajk:frontoffice`
- back-office: `docker pull jlinho/aajk:backoffice` and run
  with `docker run -p 8001:8001 jlinho/aajk:backoffice`

To re-build multi-arch images, run:
- for backoffice: `docker buildx build --push --platform linux/amd64,linux/arm64  --tag <your_name>/<your_repo>:backoffice -f DockerfileBack .` and
- for frontoffice: `docker buildx build --push --platform linux/amd64,linux/arm64  --tag <your_name>/<your_repo>:frontoffice -f DockerfileFront .`

# 2. 🤖Running the various webapps from source

Following webapps can be run:

- **backoffice**: Setup Wizard app that allows the customer to setup the call center app
- **app**: call center app that does the filtering of the useless utterances
- **mlflow**: MLFlow can be run to view the results of the experiments

## 2.0. Pre-requisites

To install from source:

- Have python `3.10` or above ready
- For each app create a virtual environment with pipenv or conda. For instance, you can run following to create 4 virtual environments: 
  - `conda create -n frontoffice python=3.10`
  - `conda create -n backoffice python=3.10`
  - `conda create -n notebooks python=3.10`
  - `conda create -n mlflow-tracking python=3.10`
- Each app requires base dependencies. Install
  with `pip install -r requirements-base.txt`

## 2.1 Run experiments and view results using MLFlow

- Inside your virtual environment, run following commands:
    - `pip install -r requirements-ml.txt` to install dependencies for ML
    - `pip install mlflow` to get ML flow web-based UI
- From folder `train` run command `mlflow ui`
- Go to http://localhost:5000 to see the MLFlow UI web app

## 2.2 Running the backoffice/Setup Wizard app

- Inside your virtual environment, run following commands:
    - `pip install -r requirements-base.txt` to install base dependencies
    - `pip install -r requirements-ml.txt` to install dependencies for ML
    - `pip install -r requirements-backoffice.txt`
- From root folder run command `streamlit run backoffice/0_Welcome.py`
- Go to http://localhost:8001 to see the app

## 2.3 Running the call-center front-office app

- Inside your virtual environment, run following commands:
    - `pip install -r requirements-base.txt` to install base dependencies
    - `pip install -r requirements-app.txt`
- Go into folder `app`
- Create a file in folder app, in `app/.env` and put the 2 following properties in it:
    - the reference to the folder where chosen model is stored (there is
      a `pipeline.yml` file in it)
    - and API key for audio transcription from deepgram (go on their site and create
      your api key)
      File should look like this:

```
MODEL_PATH="../train/mlruns/843117580351848379/246baef1fb6f4213af0f6c1d0e188c74/artifacts"
DEEPGRAM_API_KEY=your_api_key
```

- Run command `python main.py` to run the server
- Go to http://localhost:8002 to see the app
- You can directly speak to the browser or choose to type text in text-box at the bottom
  of the page

## 2.4 Running notebooks

- Inside your virtual environment, run following commands:
    - `pip install -r requirements-ml.txt` to install dependencies for ML
    - `pip install -r requirements-notebooks.txt`

# 3 📚 Folder structure

Folders in the project:

- **app**: contains a call center app that does the filtering of the useless utterances
- analysis: contains tools to track experiments and analyze results
- **backoffice**: contains the wizard app that allows the customer to setup the call
  center app
- datasets: cleaned up datasets for training and testing
- dataload: abstract location of datafiles for easier loading when files are moved
  around
- notebooks: contains the notebooks used for EDA and various analysis
- resources: raw scrapings or raw provided data
- scraping: contains the scripts used to scrape the FAQ data from the web
- tests: contains some unit tests
- **train**: contains a script and config to run variations of the ML model and parameters. Script `train/training.py` is meant to be run with [Hydra](https://hydra.cc/) to run all conbinations of parameters, models and datasets as defined in folder `train/conf`. The command to run once you enter in folder `train` is `python training.py hydra.mode=MULTIRUN`





