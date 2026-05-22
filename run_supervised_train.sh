#!/bin/bash
# Tutorial:  https://linuxconfig.org/how-to-use-a-bash-script-to-run-your-python-scripts

# cd /home/Boudreault/Documents/OCT_classification ||

# Activate the conda env
source /home/Boudreault/anaconda3/etc/profile.d/conda.sh
conda activate pytorch-env

# echo $(pwd)

HOME2="/home/Boudreault/Dokumente"
for arch in resnet18 # mobilenetv3 pvtv2b0
  do
  # Run train from 5 to 20% of the lab data starting from random weights
  python $HOME2/OCT_classification/train_supervised.py --config $HOME2/OCT_classification/config.yaml --ratio_sup 0.05 --dataset_name oct --weight_init random --arch $arch
  python $HOME2/OCT_classification/train_supervised.py --config $HOME2/OCT_classification/config.yaml --ratio_sup 0.10 --dataset_name oct --weight_init random --arch $arch
  python $HOME2/OCT_classification/train_supervised.py --config $HOME2/OCT_classification/config.yaml --ratio_sup 0.15 --dataset_name oct --weight_init random --arch $arch
  python $HOME2/OCT_classification/train_supervised.py --config $HOME2/OCT_classification/config.yaml --ratio_sup 0.20 --dataset_name oct --weight_init random --arch $arch
  # Run train from 5 to 20% of the lab data starting from imagenet weights
  python $HOME2/OCT_classification/train_supervised.py --config $HOME2/OCT_classification/config.yaml --ratio_sup 0.05 --dataset_name oct --weight_init default --arch $arch
  python $HOME2/OCT_classification/train_supervised.py --config $HOME2/OCT_classification/config.yaml --ratio_sup 0.10 --dataset_name oct --weight_init default --arch $arch
  python $HOME2/OCT_classification/train_supervised.py --config $HOME2/OCT_classification/config.yaml --ratio_sup 0.15 --dataset_name oct --weight_init default --arch $arch
  python $HOME2/OCT_classification/train_supervised.py --config $HOME2/OCT_classification/config.yaml --ratio_sup 0.20 --dataset_name oct --weight_init default --arch $arch
  # Run train on 100% of the clinical data starting random weights
  # python $HOME2/OCT_classification/train_supervised.py --config $HOME2/OCT_classification/config.yaml --ratio_sup 1 --dataset_name oct_clinical --weight_init random --arch $arch
  # Run train on 100% of the clinical data starting imagenet weights
  # python $HOME2/OCT_classification/train_supervised.py --config $HOME2/OCT_classification/config.yaml --ratio_sup 1 --dataset_name oct_clinical --weight_init default --arch $arch
done
#
