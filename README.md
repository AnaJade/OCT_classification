# Self-supervised OCT M-Scan classification

## Packages and env setup
This code was tested using the following configuration: 
* PyTorch 2.2.2
* Cuda 11.8
* Nvidia driver 535

To install PyTorch, create a conda env and run `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`.
Also install the following packages: 
* matplotlib: `conda install matplotlib`
* opencv: `pip install opencv-python`
* pandas: `conda install pandas`
* sklearn: `conda install scikit-learn`
* timm: `pip install timm`
  * hf_xet: `pip install huggingface_hub[hf_xet]`
* tqdm: `conda install tqdm`
* wandb: `pip install wandb`
* addict: `pip install addict` (for SPICE)
* lmdb: `pip install lmdb` (for SPICE)
* tensorboard: `pip install tensorboard` (for SPICE)
  * Modify protobuf version as needed: `pip install protobuf==3.20.3 --upgrade` ([Source](https://stackoverflow.com/a/75844470))
* BYOL model: `pip install byol-pytorch`
* nvidia-ml-py (gpu debugging): `pip install nvidia-ml-py`

To log into wandb, run `wandb login [api key]`

## Config file parameters
The configuration file is split into a data section, a section for each available model, and a wandb section.

**Data**
- `dataset_root`: Path to the root of the dataset containing the train-ready files
- `ds_split`: [x, y, z], where x, y, and z are the ratio of data to be used in the train, valid and test sets respectively 
- `labels`: list of labels present in the OCT scan data
- `ascan_per_group`: Number of A-scans within each M-scan subgroup
- `use_mini_dataset`: Whether to create a reduced dataset to use less storage space (only a subset of s8 scans)

**Training**
- `random_seed`: Random seed

**SPICE**

**NOTE: Also say when the same config is reused for multiple files** 
- `MoCo`: Representation learning model to be pre-trained
  - `dataset_name`: Name of the dataset used to pretrain the model. Can be either [stl10, cifar10, cifar100]
  - `dataset_path_windows`: Dataset files when running on windows will be found at [dataset_path]/[dataset_name] (if not already downloaded, files will be saved there)
  - `dataset_path_linux`: Dataset files when running on linux will be found at [dataset_path]/[dataset_name] (if not already downloaded, files will be saved there)
  - `use_all`: If true, the both the train and test sets will be used. If false, only the train set will be used. If the stl10 dataset is selected, then the unlabeled images will be included in both cases. 
  - `base_model`: Base model architecture, used to extract features from the images
  - `save_folder`: Checkpoint files will be saved at [save_folder]/[dataset_name]
  - `save_freq`: Frequency (in terms of epochs) at which checkpoints will be saved
  - `num_workers`: Used to limit memory usage
  - `max_epochs`: Number of epochs for which the model will be trained
  - `start_epoch`: If training from a checkpoint, epoch at which training should be resumed
  - `batch_size`: Number of images to use per batch
  - `lr`: Learning rate, impacts how much variation the model weights will have from one step to the other. This value 
  determines the speed at which the model will learn.
  - `lr_scheduler`: Epochs at which the learning rate will drop by 10x
  - `momentum`: momentum of SGD optimizer
  - `weight_decay`: weight decay used by the optimizer
  - `print_freq`: Frequency (in terms of batches) at which an update will be displayed 
  - `resume`: Whether to resume training from a checkpoint
  - `world_size`: Number of available GPUs ([Source](https://stackoverflow.com/a/58703819))
  - `rank`: Node rank for distributed training ([Source](https://stackoverflow.com/a/58703819))
  - `dist_url`: URL used to set up distributed training
    - On Windows: `'env://?use_libuv=False'`
    - On Linux: `'env://'`
  - `dist_backend`: Distributed backend ([PyTorch documentation](https://docs.pytorch.org/docs/stable/distributed.html))
    - On Windows: `'gloo'`
    - On Linux: `'nccl'`
  - `gpu_id`: GPU id to use
  - `multiprocessing_distributed`: Whether to use multiprocessing. If used, there will be N processes per node, which has N GPUs
  - `moco_queue_size`: queue size, number of negative keys
  - `moco_momentum`: moco momentum of updating the key encoder
  - `moco_softmax_temp`: softmax temperature
  - `moco2_mlp`: use a mlp head
  - `moco2_aug_plus`: use moco v2 data augmentation
  - `moco2_cos`: use cosine lr scheduler
- `embedding`: Precomputation of embedding features
  - `model_name`: Name of the model being trained, 'embedding' in this case
  - `weight`: Model checkpoint weights to be used saved in [MoCo save_folder]/[MoCo dataset_name]/[weight]
  - `batch_size`: Batch size 
  - `data_test`: Test data settings
    - `shuffle`: Whether test set images will be shuffled
    - `imgs_per_batch`: Number of images per batch
    - `aspect_ratio_grouping`: Not used?
    - `train`: Whether the train or test set will be used in the cifar10 or cifar100 datasets
    - `show`: Whether images will be shown while being passed via the dataloader
  - `model_sim`: Embedding model settings
    - `num_classes`: Size of the embedding??
    - `in_channels`: xxx
    - `batchnorm_track`: xxx
    - `test`: xxx
    - `features_only`: xxx
    - `model_type`: xxx
- `SPICE_self`: Training of the classification head
  - `all`: 0 # 1
  - `model_name`: spice_self
  - `resume`: False # Start again from ckp
  - `num_head`: 10 # Number of different versions of heads to be trained (best will be kept)
  - `num_train`: 5 # Not used??
  - `batch_size`: 32 # 5000
  - `target_sub_batch_size`: 16 # 100 # Must be smaller than the batch size
  - `train_sub_batch_size`: 16 # 128 # Must be smaller than the batch size
  - `batch_size_test`: 16 # 100
  - `num_trans_aug`: 1 # Not used??
  - `num_repeat`: 8
  - `fea_dim`: 512
  - `att_size`: 7
  - `center_ratio`: 0.5
  - `sim_center_ratio`: 0.9
  - `epochs`: 100
  - `start_epoch`: 0
  - `print_freq`: 2 # Can't be larger than batch_size//sub_batch_size
  - `test_freq`: 1
  - `eval_ent`: False
  - `eval_ent_weight`: 0
- `local_consistecy`: Finding the reliable pseudo-labels
  - `model_name`: eval
  - `batch_size`: 32 # 100
  - `batch_size_test`: 32
- `semi`: Semi-supervised joint training of the entire model using the reliable pseudo-labels
  - `model_name`: 'spice_semi'
  - `resume`: False
  - `overwrite`: True
  - `epoch`: 1
  - `num_train_iter`: 2**20
  - `num_eval_iter`: 1000
  - `num_labels`: 6510
  - `batch_size`: 32 # 64
  - `uratio`: 7   # Ratio of unlabeled to labeled data in each mini-batch
  - `eval_batch_size`: 1024
  - `hard_label`: True
  - `T`: 0.5l
  - `p_cutoff`: 0.95
  - `ema_m`: 0.999 # ema momentum for eval model
  - `ulb_loss_ratio`: 1
  - `lr`: 0.03
  - `momentum`: 0.9
  - `weight_decay`: 5e-4
  - `amp`: False  # Use of mixed precision training or not
  - `net`: 'WideResNet_stl10' # ['WideResNet', 'WideResNet_stl10', 'WideResNet_tiny', 'resnet18', 'resnet18_cifar', 'resnet34']
  - `net_from_name`: False
  - `depth`: 28
  - `widen_factor`: 2
  - `leaky_slope`: 0.1
  - `dropout`: 0
  - `all`: 0 # 1
  - `unlabeled`: 0 # 1
  - `train_sampler`: RandomSampler

**SimCLR**
- `dataset_name`: Name of the dataset used to pretrain the model. Can be either [oct, stl10, cifar10, cifar100]
- `dataset_path_windows`: Dataset files when running on windows will be found at [dataset_path]/[dataset_name] (if not already downloaded, files will be saved there)
- `dataset_path_linux`: Dataset files when running on linux will be found at [dataset_path]/[dataset_name] (if not already downloaded, files will be saved there)
- `dataset_sample`: Ratio of the entire dataset that will be loaded. Images will be sample grouped by label and trajectory. Used to decrease training time.
- `arch`: Architecture of the feature extractor
- `num_workers`: Used to limit memory usage
- `max_epochs`: Number of epochs for which the model will be trained
- `batch_size`: Number of images to use per batch
- `lr`: Learning rate, impacts how much variation the model weights will have from one step to the other. This value 
  determines the speed at which the model will learn.
- `weight_decay`: weight decay used by the optimizer
- `disable_cuda`: Disable cuda for debugging purposes
- `fp16_precision`: Whether to use fp16 in the forward pass during training
- `log_every_n_steps`: Print update every x batch
- `temperature`: Loss function temperature
- `gpu_index`: GPU id to use
- `patience`: Number of epochs for which the loss can not reach a new minimum and the model will keep training
**WandB**
- `wandb_log`: Choose if you want this experiment logged on weights and biases.
- `project_name`: Corresponds to the project name on wandb.

## Data preparation
View the [Dataset file struction](#dataset-file-structure) section to view how the files should be saved and how they will be organized.
- Reconstruct the raw OCT scans using the Matlab script (`Reconstruct_all_data_part_11052025.m`) to generate the `*.mat` files
- Run the `Convert_to_jpg.m` script to convert all `.mat` files to individual `.jpg` images.
  - This is done because `.mat` take a very long time to load in python
- Run the `prep_oct_data.py` script to generate the train-valid-test map dfs and get the mean and std of the resulting train set images.
  

## Dataset file structure
The root folder can be the same for the following three categories of files, 
but can also be separate if insufficient storage space is available to store everything on the same disk.
### Raw scan data
Unprocessed OCT scans are saved in a folder containing the following sub-folders and files:
```
└── Raw scan data root
    └── [scan name 0]
        ├── [scan name 0]_[traj name 0]_*_raw.dat
        ├── [scan name 0]_[traj name 0]_*_time.dat
        ├── [scan name 0]_[traj name 0]_*_framegrabber.mp4
        ├── [scan name 0]_[traj name 0]_*_framegrabber.txt
        ├── [traj name 0].txt
        └── ...
    └── [scan name 1]

        └── ...
    └── ...
```

### After Matlab reconstruction
Reconstructed OCT scans are saved in a folder containing the following sub-folders and files:
```
└── Reconstructed data root
    └── [scan name 0]
        ├── [scan name 0]_[traj name 0]_*_raw.mat
        ├── [scan name 0]_[traj name 0]_*_time.csv
        └── ...
    └── [scan name 1]

        └── ...
    └── ...
```

### Saving as individual images (also done in Matlab)
The reconstructed OCT scans are split into [`ascan_per_group`] chunks and saved as individual images. 
Any desired pre-processing (in this case [moving average](https://de.mathworks.com/help/matlab/ref/movmean.html) and 
[speckle](https://de.mathworks.com/help/medical-imaging/ref/specklefilt.html) 
([SRAD](https://pubmed.ncbi.nlm.nih.gov/18249696/)) filter) can be done at this step
The resulting files will be saved in the following format:
```
└── Dataset root/[ascan_per_group]mscans
    └── [scan name 0]
        ├── [scan name 0]_[traj name 0]_[idx_start]_[idx_end].jpg
        └── ...
    └── [scan name 1]
        └── ...
    └── ...
```

### Train-ready files
Mapping dataframes used by the dataset class are created and final reconstructed OCT scans are saved in a folder 
containing the following sub-folders and files:
```
└── Dataset root/[ascan_per_group]mscans
    └── [scan name 0]
        ├── [scan name 0]_[traj name 0]_[idx_start]_[idx_end].jpg
        └── ...
    └── [scan name 1]
        └── ...
    ├── ...
    ├── test_mapping_[ascan_per_group]scans.csv
    ├── train_mapping_[ascan_per_group]scans.csv
    └── valid_mapping_[ascan_per_group]scans.csv
```
The train, valid and test mapping dataframes are created based on the desired A-scan number per grouping and saved to `.csv` files.
This is done in a way to ensure all images recorded over the same area (identified by `scan name`) will be treated as a unit, 
meaning that they will all end up in the same subset. This was done to mimic how scans from different patients would be
treated in a clinical trial. The A-scan count over each area is used to iteratively assign areas to subsets until the minimum error is reached.
The following table will be created using the resulting area splits for the train, valid and test sets:

| img_relative_path                                                                             | label | idx_start  | idx_end  | 
|-----------------------------------------------------------------------------------------------|-------|------------|----------|
| [ascan_per_group]mscans/[scan name 0]/[scan name 0]_[traj name 0]_[idx_start0]_[idx_end0].jpg | ...   | idx_start0 | idx_end0 | 
| [ascan_per_group]mscans/[scan name 0]/[scan name 0]_[traj name 0]_[idx_start1]_[idx_end1].jpg | ...   | idx_start1 | idx_end1 |
| ...                                                                                           | ...   | ...        | ...      |
| [ascan_per_group]mscans/[scan name 1]/[scan name 0]_[traj name 0]_[idx_start0]_[idx_end0].jpg | ...   | idx_start0 | idx_end0 |
| ...                                                                                           | ...   | ...        | ...      |



