## HOW TO LAUNCH
```bash
python MODEL_1D.py EW  Det
python MODEL_1D.py KPZ Det
python MODEL_1D.py BIO Det
```

the code will produce an output image and you can check in OUTPUT REF
if your results are consitent.
Three models are corrently implemented:

Edward Wilkinson (EW)

Kardar Parisi Zhang (KPZ)

Fisher-KPP (BIO) <--to be adimensionalized 

In the case of production use the commands as reported in launch_serial.sh

```bash
python MODEL_1D.py EW  Rnd
python MODEL_1D.py KPZ Rnd
python MODEL_1D.py BIO Rnd
```
this will use random seed. 
