#!/bin/bash
#SBATCH -p gpu_e  # Use gpu_e nodes (u007)
#SBATCH --ntasks 1 # Probably not necessary to make any changes here 
#SBATCH --cpus-per-task 16 # number of requested cpu cores 
#SBATCH --gres=gpu:1 # number of requested gpus
#SBATCH --mem=128G # requested ram
#SBATCH --time 150:00:00 # maximum runtime. Keep high values
#SBATCH -o outputlog_gpu_%j.out # sbatch outputlog. %j is job ID
#SBATCH -e errorlog_gpu_%j.out # sbatch errorlog

# --- Load toolchain modules ---
# ./etc/profile.d/module.sh # probably always necessary to make module available?
module load conda # load a module

# activate conda env
conda init bash
source ~/.bashrc
conda activate pytorch-env

#--- Prepare Workspace in a Job-Specific Directory ---
MYWORKDIR="$SCRATCH"
mkdir -p "$MYWORKDIR"
echo "Working directory: $MYWORKDIR"
echo "Permanent storage (WORK): $WORK"

#--- Clean previous artifacts in SCRATCH to free space ---
rm -rf "$SCRATCH/experiments" "$SCRATCH/cache" 2>/dev/null || true

#--- Copy Data to Temporary Workspace ---
echo "--- Copying data to SCRATCH... ---"
rsync -a --info=progress2 "$WORK/OCT_lab_data/512mscans_noNoise_sample20" "$MYWORKDIR/OCT_lab_data"
cd "$MYWORKDIR"
pwd
ls

#--- Set Cache Directories --- 
export WANDB_API_KEY="8ab9b1718d14ea7c2ca1"
export HF_HOME="$MYWORKDIR/.cache/huggingface"
export PIP_CACHE_DIR="$MYWORKDIR/.cache/pip"
export TORCH_HOME="$MYWORKDIR/.cache/torch"
mkdir -p $HF_HOME $PIP_CACHE_DIR $TORCH_HOME


# --- Training params ---


#--- Run Training ---
python /fibus/fs0/14/cab8351/OCT_classification/BYOL/train_byol.py --config config.yaml

#
