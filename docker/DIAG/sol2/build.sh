#!/bin/bash

VERSION="v0.02"


NO_CACHE=""
# Parse command line options
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --no-cache)
      NO_CACHE="--no-cache"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

docker build $NO_CACHE -t doduo1.umcn.nl/nnunet_for_pathology/sol2:$VERSION . && \
docker push doduo1.umcn.nl/nnunet_for_pathology/sol2:$VERSION && \

docker tag doduo1.umcn.nl/nnunet_for_pathology/sol2:$VERSION doduo1.umcn.nl/nnunet_for_pathology/sol2:latest
docker push doduo1.umcn.nl/nnunet_for_pathology/sol2:latest
