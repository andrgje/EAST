FROM tensorflow/tensorflow:1.9.0-gpu-py3 AS dev

ARG DOCKER_UID
ARG DOCKER_GID
RUN groupadd -r docker-user -g $DOCKER_GID && useradd -r -u $DOCKER_UID -g $DOCKER_GID -m -s /bin/false -g docker-user docker-user

RUN apt update && apt install -y less nano jq git libgeos-dev

COPY bash.bashrc /etc/bash.bashrc

ARG DOCKER_WORKSPACE_PATH
WORKDIR $DOCKER_WORKSPACE_PATH

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN pip install gdown

USER $DOCKER_UID:$DOCKER_GID

# -- stage for job export (e.g. include code as well) 
FROM dev AS job
COPY . .
