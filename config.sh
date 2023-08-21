#!/bin/bash

#Step0. Configure Envirenment
echo "<<<<<<<<<  Step0. Configure Envirenment  >>>>>>>>>>>"
source ~/.bashrc
conda create --yes -n ModelTrain python=3.7
source activate ModelTrain
pip install -r requirements.txt

