from python:3.7-slim-buster

RUN mkdir road_embedding_gnn

WORKDIR road_embedding_gnn

RUN mkdir data
RUN mkdir data/models
RUN mkdir data/models/dgi
RUN mkdir data/models/gae
RUN mkdir data/models/graphmae
RUN mkdir data/training_data
RUN mkdir data/training_data/dgi
RUN mkdir data/training_data/gae
RUN mkdir data/training_data/graphmae
RUN mkdir data/data_train
RUN mkdir data/data_val

RUN python -m pip install --upgrade pip

COPY ./requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

#if cuda
RUN pip3 install torchvision torchaudio
RUN pip3 install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

#if cpu 
#RUN pip3 install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cpu.html

COPY ./models ./models
COPY ./configs.yml ./configs.yml
COPY ./secret.py ./secret.py
COPY ./utils.py ./utils.py
COPY ./params.py ./params.py
COPY ./dgi_train.py ./dgi_train.py
COPY ./gae_train.py ./gae_train.py
COPY ./graphmae_train.py ./graphmae_train.py

COPY ./download_graphs.py ./download_graphs.py
COPY ./transform_graphs.py ./transform_graphs.py

#VOLUME ./data ./data
COPY ./train_models.sh ./train_models.sh
CMD ["/bin/bash", "./train_models.sh"]



















