#! /bin/bash
#
date

source sequence.sh

python MODEL_1D.py EW
python MODEL_1D.py KPZ
python MODEL_1D.py BIO

date
