#! /bin/bash
#
date

source sequence.sh

python MODEL_1D.py EW  Rnd
python MODEL_1D.py KPZ Rnd
python MODEL_1D.py BIO Rnd

date
