FROM python:3.10-slim as python310_base
WORKDIR /webapp
RUN apt-get -y update  && apt-get install -y \
    python3-dev \
    apt-utils \
    python-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -U pip
RUN pip install --no-cache-dir -U cython
RUN pip install --no-cache-dir -U numpy
COPY requirements-base.txt ./
RUN pip install --no-cache-dir -U -r  requirements-base.txt
RUN python -m nltk.downloader punkt

FROM python310_base as frontend_app
WORKDIR /webapp
COPY requirements-app.txt ./
RUN pip install --no-cache-dir -U -r  requirements-app.txt
COPY app .
COPY dataload ./dataload
COPY pipelines ./pipelines
COPY train/mlruns/843117580351848379/246baef1fb6f4213af0f6c1d0e188c74/artifacts ./model
EXPOSE 8002
CMD ["python", "main.py"]