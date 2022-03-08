FROM python:3.6.13

ARG DOCKER_UID
ARG DOCKER_GID
RUN groupadd -r docker-user -g $DOCKER_GID && useradd -r -u $DOCKER_UID -g $DOCKER_GID -m -s /bin/false -g docker-user docker-user

RUN apt update && apt install -y less nano jq git libgeos-dev

COPY bash.bashrc /etc/bash.bashrc

ARG DOCKER_WORKSPACE_PATH
WORKDIR $DOCKER_WORKSPACE_PATH

COPY requirements.txt .
RUN pip install -r requirements.txt

USER $DOCKER_UID:$DOCKER_GID

CMD ["python", "train.py", "--data", "/path/at/ai/sweden/", "--chackpoint-dir", "/another/path/at/ai/sweden"]