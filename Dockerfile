FROM continuumio/miniconda3:latest

RUN apt update && apt install --yes build-essential nano htop byobu git openssh-client less wget

COPY env.yml .

RUN conda env create -f env.yml

# Make RUN commands use the new environment:
RUN echo "conda activate bertrand" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

RUN conda install -y jupyterlab ipywidgets tqdm

COPY . .

