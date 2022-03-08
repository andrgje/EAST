#!/bin/bash
set -e

# Path to repo/project root dir (independent of pwd)
PROJECT_ROOT=$( cd $(dirname $(readlink -f $0) ); pwd )

# Docker image name for this project
DOCKER_IMAGE_TAG="$(git rev-parse --short HEAD)-$(date +"%Y-%m-%d_%H-%M-%S")"
DOCKER_IMAGE_NAME="${DOCKER_IMAGE_NAME:-andrgje/east-job:$DOCKER_IMAGE_TAG}"

# Path to where in the docker container the project root will be mounted
export DOCKER_WORKSPACE_PATH="${DOCKER_WORKSPACE_PATH:-/workspace}"

set -x
docker build --target job --rm --build-arg DOCKER_WORKSPACE_PATH --build-arg DOCKER_UID=1000 --build-arg DOCKER_GID=1000 -t $DOCKER_IMAGE_NAME $PROJECT_ROOT
