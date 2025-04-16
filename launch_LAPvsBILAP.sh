#! /bin/bash

source /g100/home/userexternal/plazzari/sequence3.sh

mkdir -p LAPvsBILAP_data LAPvsBILAP_plots

#wn= wave number

for wn in 5 10 20 30 40 50; do
  python3 MODEL_1D_DET_ENS_FAST.py LAP ${wn} ; python3 MODEL_1D_DET_ENS_FAST.py BILAP ${wn} ; python plot_LAPvsBILAP.py ${wn}
done
