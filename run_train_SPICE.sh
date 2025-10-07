#!/bin/bash
# Tutorial:  https://linuxconfig.org/how-to-use-a-bash-script-to-run-your-python-scripts

# cd /home/Boudreault/Documents/Mandible_tracking ||

# Activate the conda env
source /home/Boudreault/anaconda3/etc/profile.d/conda.sh
conda activate pytorch-env

# echo $(pwd)

# Train the feature model
# PYTHONPATH=/home/Boudreault/Dokumente/OCT_classification python SPICE/tools/train_moco.py config.yaml
# PYTHONPATH=/home/Boudreault/Dokumente/OCT_classification python SPICE/tools/pre_compute_embedding.py config.yaml

# Train the clustering head
# PYTHONPATH=/home/Boudreault/Dokumente/OCT_classification python SPICE/tools/train_self_v2.py config.yaml

# Joint training of feature model and clustering head
PYTHONPATH=/home/Boudreault/Dokumente/OCT_classification python SPICE/tools/local_consistency.py config.yaml
# PYTHONPATH=/home/Boudreault/Dokumente/OCT_classification python SPICE/tools/train_semi.py config.yaml
