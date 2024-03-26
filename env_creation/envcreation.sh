#!/bin/bash

mkdir -p $1
cd $1
git clone https://github.com/Jaiaid/hardlambda_exp_code
virtualenv -m venv
source venv/bin/activate
pip install --no-cache-dir -r requirements.txt