#!/bin/sh
CUDA_VERSION=cu111
pip-compile setup.py --find-links=https://download.pytorch.org/whl/$CUDA_VERSION/torch_stable.html --upgrade --generate-hashes --output-file=requirements-lock.txt