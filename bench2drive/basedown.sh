#!/bin/bash

# Check if huggingface-hub is installed
if ! python -m pip show huggingface-hub > /dev/null 2>&1; then
  echo "huggingface-hub is not installed. Installing now..."
  python -m pip install huggingface-hub
else
  echo "huggingface-hub is already installed."
fi

mkdir /home1/irteam/data-vol1/src/HiP-AD/bench2drive/Bench2Drive-base

hf download rethinklab/Bench2Drive --repo-type=dataset --local-dir /home1/irteam/data-vol1/src/HiP-AD/bench2drive/Bench2Drive-base