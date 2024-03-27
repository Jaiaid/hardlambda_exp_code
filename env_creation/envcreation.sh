#!/bin/bash

mkdir -p $1
cd $1
# if repo is there remove it
# needed for multiple attempt
rm -rf hardlambda_exp_code
git clone https://Jaiaid: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@github.com/Jaiaid/hardlambda_exp_code
# if venv already there remove it
# needed in case of multiple run
rm -rf venv
python3 -m virtualenv --python=python3 venv
COMPLETIONCODE=$?

if [[ $COMPLETIONCODE -eq 0 ]]
then
    source venv/bin/activate & python3 -m pip install --no-cache-dir -r hardlambda_exp_code/requirements.txt
else
    echo "virtualenv creation failed"
fi
