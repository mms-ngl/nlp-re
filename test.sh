#!/bin/bash

# initial check

if [ "$#" != 1 ]; then
    echo "$# parameters given. Only 1 expected. Use -h to view command format"
    exit 1
fi

if [ "$1" == "-h" ]; then
  echo "Usage: $(basename "$0") [file to evaluate upon]"
  exit 1
fi

test_path=$1

# delete old docker if exists
docker ps -q --filter "name=nlp-re" | grep -q . && docker stop nlp-re
docker ps -aq --filter "name=nlp-re" | grep -q . && docker rm nlp-re

# build docker file
docker build . -f Dockerfile -t nlp-re

# bring model up
docker run -d -p 12345:12345 --name nlp-re nlp-re

# perform evaluation
/usr/bin/env python re/evaluate.py "$test_path"

# stop container
docker stop nlp-re

# dump container logs
docker logs -t nlp-re > logs/server.stdout 2> logs/server.stderr

# remove container
docker rm nlp-re