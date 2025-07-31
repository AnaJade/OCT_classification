# OCT_classification
Self-supervised OCT M-Scan classification

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
* tqdm: `conda install tqdm`
* wandb: `pip install wandb`
* addict: `pip install addict` (for SPICE)
* lmdb: `pip install lmdb` (for SPICE)
* tensorboard: `pip install tensorboard` (for SPICE)
  * Modify protobuf version as needed: `pip install protobuf==3.20.3 --upgrade` (Source)[https://stackoverflow.com/a/75844470]

To log into wandb, run `wandb login [api key]`

## Config file parameters
The configuration file contains 3 main section: data, training and wandb.

**Data**
- `dataset_root`: Path to the root of the dataset containing the train-ready files
- `ds_split`: [x, y, z], where x, y, and z are the ratio of data to be used in the train, valid and test sets respectively 
- `labels`: list of labels present in the OCT scan data
- `ascan_per_group`: Number of A-scans within each M-scan subgroup
- `use_mini_dataset`: Whether to create a reduced dataset to use less storage space (only a subset of s8 scans)

**Training**
- `sub_model`: Name of the sub-model that will be used in the model architecture
- `weights_file_addon`: String to be added at the end of the weight file name (can be used to create new file names during sweeps)
- `use_pretrained`: Whether to load the subnet pre-trained weights 
- `cam_imputs`: Images from these cameras will be used for training
- `num_fc_hidden_units`: Number of units in the fully connected hidden layer
- `train_bs`: Training batch size
- `valid_bs`: Validation batch size
- `test_bs`: Test batch size
- `max_epochs`: Maximum number of epochs that the model will be trained. This number can be large, since the training 
should stop on its own once performance on the valid set stops improving.
- `patience`: Number of epochs for which the model will continue training, even though no improvement is seen on the 
valid set performance.
- `lr`: Learning rate, impacts how much variation the model weights will have from one step to the other. This value 
determines the speed at which the model will learn. A value too small or too big can significantly increase the 
training time.
- `lr_scheduler`: A learning rate scheduler can be used to reduce the learning rate as training progresses. This means
that as the weights get closer to their local optimum, the step size for each update will be smaller to avoid 
overshooting.
- `step_size`: Number of epochs between every learning rate update. The default value given in the tutorial is 3.
- `gamma`: Value by which the learning rate will change. The default value given in the tutorial is 0.1.

**SPICE**

**NOTE: Also say when the same config is reused for multiple files** 
- `MoCo`: Representation learning model to be pre-trained
  - `dataset_name`: Name of the dataset used to pretrain the model. Can be either [stl10, cifar10, cifar100]
  - `dataset_path`: Dataset files will be found at [dataset_path]/[dataset_name] (if not already downloaded, files will be saved there)
  - `use_all`: If true, the both the train and test sets will be used. If false, only the train set will be used. If the stl10 dataset is selected, then the unlabeled images will be included in both cases. 
  - `base_model`: Base model architecture, used to extract features from the images
  - `save_folder`: Checkpoint files will be saved at [save_folder]/[dataset_name]
  - `save_freq`: Frequency (in terms of epochs) at which checkpoints will be saved
  - `num_workers`: Used to limit memory usage
  - `max_epochs`: Number of epochs for which the model will be trained
  - `start_epoch`: If training from a checkpoint, epoch at which training should be resumed
  - `batch_size`: Number of images to use per batch
  - `lr`: Learning rate, impacts how much variation the model weights will have from one step to the other. This value 
  determines the speed at which the model will learn. A value too small or too big can significantly increase the 
  training time.
  - `lr_scheduler`: Epochs at which the learning rate will drop bz 10x
  - `momentum`: momentum of SGD optimizer
  - `weight_decay`: weight decay
  - `print_freq`: Frequency (in terms of batches) at which an update will be displayed 
  - `last_ckp_path`: If restarting training, path to the last checkpoint
  - `world_size`: Number of available GPUs ([Source](https://stackoverflow.com/a/58703819))
  - `rank`: Node rank for distributed training ([Source](https://stackoverflow.com/a/58703819))
  - `dist_url`: URL used to set up distributed training
  - `dist_backend`: Distributed backend ([PyTorch documentation](https://docs.pytorch.org/docs/stable/distributed.html))
  - `gpu_id`: GPU id to use
  - `multiprocessing_distributed`: Whether to use multiprocessing. If used, there will be N processes per node, which has N GPUs
  - `moco_queue_size`: queue size, number of negative keys
  - `moco_momentum`: moco momentum of updating the key encoder
  - `moco_softmax_temp`: softmax temperature
  - `moco2_mlp`: use a mlp head
  - `moco2_aug_plus`: use moco v2 data augmentation
  - `moco2_cos`: use cosine lr scheduler
- `embedding`: Precomputation of embedding features
  - `model_name`: xxx
  - `weight`: Model checkpoint weights to be used saved in [MoCo save_folder]/[MoCo dataset_name]/[weight]
  - `batch_size`: Batch size 
  - `data_test`: Test data settings
    - `shuffle`: xxx
    - `imgs_per_batch`: xxx
    - `aspect_ratio_grouping`: xxx
    - `train`: xxx
    - `show`: xxx
  - `model_sim`: Embedding model settings
    - `num_classes`: xxx
    - `in_challens`: xxx
    - `batchnorm_track`: xxx
    - `test`: xxx
    - `features_only`: xxx
    - `model_type`: xxx

**WandB**
- `wandb_log`: Choose if you want this experiment logged on weights and biases.
- `project_name`: Corresponds to the project name on wandb.



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

### Train-ready files
Final reconstructed OCT scans are saved in a folder containing the following sub-folders and files:
```
└── Dataset root
    └── [scan name 0]
        ├── [scan name 0]_[traj name 0]_*_raw.npy
        └── ...
    └── [scan name 1]
        └── ...
    ├── ...
    ├── test_mapping_*scans.csv
    ├── train_mapping_*scans.csv
    └── valid_mapping_*scans.csv
```
The train, valid and test mapping dataframes are created based on the desired A-scan number per grouping and saved to `.csv` files.
If grouping by 5000 A-scans, the following table would be created:

| relative_path                                       | label | idx_start | idx_end | 
|-----------------------------------------------------|-------|-----------|---------|
| [scan name 0]/[scan name 0]_[traj name 0]_*_raw.npy | ...   | 0         | 5000    | 
| [scan name 0]/[scan name 0]_[traj name 0]_*_raw.npy | ...   | 5000      | 1000    |
| ...                                                 | ...   | ...       | ...     |
| [scan name 1]/[scan name 1]_[traj name 0]_*_raw.npy | ...   | 0         | 5000    |
| ...                                                 | ...   | ...       | ...     |

## Data preparation
- Reconstruct the raw OCT scans using the Matlab script to generate the `*.mat` files
- Run the `mat2npy.py` script to convert all `.mat` files to `.npy`
  - This is done because `.npy` can be loaded significantly faster than `.mat` files
- Run the `split_train_test.py` script to generate the train-valid-test map dfs
  


## TODO
- Decide which pre-processing techniques use what parameters, and put them in the config file
- Compare run time vs memory tradeoff for loading the `.mat` files in the dataset `__init__` vs `__getitem__`
