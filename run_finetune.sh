#!/bin/bash
# Tutorial:  https://linuxconfig.org/how-to-use-a-bash-script-to-run-your-python-scripts

# cd /home/Boudreault/Documents/OCT_classification ||

# Activate the conda env
source /home/Boudreault/anaconda3/etc/profile.d/conda.sh
conda activate pytorch-env

# echo $(pwd)

HOME2="/home/Boudreault/Dokumente"
for approach in simclr byol
do
  for arch in resnet18 # mobilenetv3 pvtv2b0
  do
    # Run finetune from 5 to 20% of the lab data
  python $HOME2/OCT_classification/finetune_model.py --config $HOME2/OCT_classification/config.yaml --ratio_sup 0.05 --dataset_name oct --approach $approach --arch $arch
  python $HOME2/OCT_classification/finetune_model.py --config $HOME2/OCT_classification/config.yaml --ratio_sup 0.10 --dataset_name oct --approach $approach --arch $arch
  python $HOME2/OCT_classification/finetune_model.py --config $HOME2/OCT_classification/config.yaml --ratio_sup 0.15 --dataset_name oct --approach $approach --arch $arch
  python $HOME2/OCT_classification/finetune_model.py --config $HOME2/OCT_classification/config.yaml --ratio_sup 0.20 --dataset_name oct --approach $approach --arch $arch
  # Run finetune on 100% of the clinical data
  # python $HOME2/OCT_classification/finetune_model.py --config $HOME2/OCT_classification/config.yaml --ratio_sup 1 --dataset_name oct_clinical --approach $approach --arch $arch
  done
done