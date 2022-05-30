FROM continuumio/miniconda:latest

RUN mkdir /code
WORKDIR /code
ADD . /code

RUN apt-get --allow-releaseinfo-change update
RUN apt-get -y install gcc g++

# set up conda environment
WORKDIR /code
# RUN conda env create -f environment.yml
RUN conda create --name papyrus
RUN echo "conda activate papyrus" >> ~/.bashrc

ENV SQLALCHEMY_DATABASE_URL postgres://postgres@localhost:5432/papyrus

SHELL ["/bin/bash", "--login", "-c"]
RUN conda install rdkit=2021.03.5 FPSim2 openbabel psycopg2 jupyter python=3.8 -c conda-forge
RUN pip install -e git+https://github.com/reskyner/razi.git#egg=razi
RUN pip install requests python-multipart alembic modin[ray]

# ENV PORT 8001