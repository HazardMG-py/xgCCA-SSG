#!/bin/bash

# Run xgCCA-SSG on OGB-BioKG
python src/main.py --dataname biokg --epochs 100 --gpu 0

# Run xgCCA-SSG on Cora
python src/main.py --dataname cora --epochs 100 --gpu 0