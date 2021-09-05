#!/usr/bin/bash

conda create -n IET_env python=3.8 -y
conda activate IET_env
python3.8 -m pip install -r requirements.txt

