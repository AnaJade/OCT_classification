import copy
import argparse
import pathlib
import warnings
import platform
import socket
from sys import platform
from random import randint
import re
import h5py

import cv2
import matplotlib
from tqdm import tqdm
from typing import Union

if (platform == "linux" or platform == "linux2") and ('hpc' in socket.gethostname() or 'u00' in socket.gethostname()):
    matplotlib.use('Agg')
else:
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from scipy import signal
from scipy.io import loadmat
from PIL import Image
import random
from sklearn.metrics import mean_squared_error

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
# Disable warning for using transforms.v2
torchvision.disable_beta_transforms_warning()
from torchvision.io import read_image
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import v2, InterpolationMode

import utils


class OCTDataset(Dataset): # Used in train_moco
    def __init__(self, root: pathlib.Path, split: str, map_df_paths: dict, labels_dict: dict, ch_in=3,
                 sample_within_image=-1, use_iipp=False, num_same_area=-1, transforms=None, pre_shuffle=True,
                 pre_sample=1, seq_split=False, ratio_sup=1, overwrite_split=None):
        """
        Dataset object used to pass images to a siamese network
        :param root: dataset root path
        :param split: dataset split ('train', 'valid', 'test')
        :param map_df_paths: Path to the mapping dataframes of start and stop idx of M-Scans (train, valid, test)
        :param labels_dict: int to string label conversion
        :param ch_in: Number of input channels (1: grayscale, 3: rgb)
        :param sample_within_image: a-scan sequence length to be sampled from within the m-scan, set to -1 for no sampling
        :param use_iipp: Whether to prepare the mapping df for intra-image positive pairs (adding meta-data)
        :param num_same_area: number of images that will also be sampled within the same area (only for SimCLR, pairs have to be pre-assigned for BYOL)
        :param transforms: Set of transforms that will be applied to each image before being used as input by the model
        :param pre_shuffle: Whether to shuffle the images once at the beginning
        :param pre_sample: Ratio of images to be kept
        :param seq_split: Whether to split the dataset per trajectory (first 60% train, 20% valid, last 20% test)
        :param ratio_sup: Ratio of images to be used for supervised training
        :param overwrite_split: Dict with new split based on pat {'train': [], 'valid': [], 'test': []}
        """
        # Check if re-splitting is needed (in case split in ['train_supervised', 'valid_supervised', 'test_supervised'])
        supervised = 'supervised' in split
        if ('supervised' in split) and ('_' in split):
            split = split.split('_')[0]
        self.root = root
        self.transforms = transforms
        self.split = split
        self.seq_split = seq_split
        self.map_df = map_df_paths[split]
        self.map_df_sampling = None
        self.label_dict = labels_dict
        self.ch_in = ch_in
        self.sample_within_image = sample_within_image
        self.use_iipp = use_iipp
        self.num_same_area = num_same_area
        self.map_df = pd.read_csv(self.map_df)
        if self.seq_split:
            self.map_df = pd.concat([pd.read_csv(p) for p in map_df_paths.values()], axis=0, ignore_index=True)
        self.pre_shuffle = pre_shuffle
        self.pre_sample = pre_sample
        self.map_df_sampling = None # set for iipp when num_same_area >= 2
        self.show = False # For debug purposes
        self.new_labels = 'old_label' in self.map_df.columns
        self.ratio_sup = ratio_sup
        self.overwrite_split = overwrite_split

        # Redo-split if necessary
        if self.overwrite_split is not None:
            map_dfs = pd.concat([pd.read_csv(pathlib.Path(p)) for p in map_df_paths.values()], axis=0)
            map_dfs.loc[:, 'img_relative_path'] = [pathlib.Path(p) for p in map_dfs['img_relative_path']]
            map_dfs.loc[:, 'area'] = [p.parts[-2] for p in map_dfs['img_relative_path']]
            # map_dfs.loc[:, 'pat'] = [int(re.sub(r'[^\d]+', '', p.parts[-2])) for p in map_dfs['img_relative_path']]
            new_areas = self.overwrite_split[split]
            self.map_df = map_dfs[map_dfs['area'].isin(new_areas)].copy()
            self.map_df.loc[:, 'subset'] = split

        # Update relative path to image paths
        ascan_per_group = self.map_df['idx_end'].iloc[0]
        # img_subdir = pathlib.Path(f"{ascan_per_group}mscans")
        self.map_df.loc[:, 'img_relative_path'] = [pathlib.Path(p) for p in self.map_df.loc[:,'img_relative_path']]

        # Restructure mapping df
        self.map_df = self.map_df.rename(columns={'label': 'label_str'})
        for i, lbl in self.label_dict.items():
            self.map_df.loc[self.map_df['label_str'] == lbl, 'label'] = i
        if not self.new_labels:
            missing_labels = [l for l in labels_dict.values() if l not in self.map_df['label_str'].unique()]
            extra_labels = self.map_df[self.map_df['label'].isna()]['label_str'].unique().tolist()
            if len(missing_labels) > 0:
                print(f"{missing_labels} not found in mapping dataset")
            if len(extra_labels) > 0:
                print(f"{extra_labels} label found in mapping dataset but not in label dict")
                print(f"Removing images...")
                self.map_df = self.map_df.dropna(axis=0)
        else:
            self.map_df = self.map_df.dropna(subset=['label'], axis=0).copy()
        self.map_df['area'] = [re.sub('|'.join(f'{f}_' for f in self.label_dict.values()), '', p.parts[1]) for p in self.map_df['img_relative_path']]
        self.map_df['trajectory'] = ['_'.join(p.stem.split('_')[:-2]) for p in self.map_df['img_relative_path']]
        self.map_df['area_id'] = self.map_df.groupby(['label', 'area']).ngroup()

        # Remove images from the second half of the trajectories
        if supervised and (self.ratio_sup == 2):
            self.map_df['id_traj'] = self.map_df.groupby('trajectory').cumcount()
            idx_split = self.map_df.groupby('trajectory').agg(max_idx=('id_traj', 'max'))
            idx_split.loc[:, 'max_keep'] = idx_split.loc[:, 'max_idx'] * 0.5
            self.map_df = pd.merge(self.map_df, idx_split, on='trajectory')
            self.map_df = self.map_df[self.map_df['id_traj'] < self.map_df['max_keep']].copy()
            self.map_df = self.map_df.drop(columns=['id_traj', 'max_idx', 'max_keep'])
            # Update split info
            self.map_df.loc[:, 'subset'] = f'{split}_supervised'

        # Redo splitting
        if self.seq_split:
            # New re-splitting
            self.map_df['id_traj'] = self.map_df.groupby('trajectory').cumcount()
            # Get idx thresh per traj
            idx_split = self.map_df.groupby('trajectory').agg(max_idx=('id_traj', 'max'))
            idx_split.loc[:, 'max_train'] = idx_split.loc[:, 'max_idx'] * 0.6
            idx_split.loc[:, 'max_valid'] = idx_split.loc[:, 'max_idx'] * 0.8
            idx_split = idx_split.drop(columns=['max_idx']).reset_index()
            self.map_df = pd.merge(self.map_df, idx_split, on='trajectory')
            # Redefine subsets
            self.map_df.loc[:, 'subset'] = ''
            self.map_df.loc[self.map_df['id_traj'] < self.map_df['max_train'], 'subset'] = 'train'
            self.map_df.loc[self.map_df['id_traj'] > self.map_df['max_valid'], 'subset'] = 'test'
            self.map_df.loc[self.map_df['subset'] == '', 'subset'] = 'valid'
            # Redefine supervised subset
            self.map_df.loc[self.map_df['id_traj'] % 10 == 9, 'subset'] = [f"{s}_supervised" for s in self.map_df.loc[self.map_df['id_traj'] % 10 == 9, 'subset']]
            # Remove extra cols
            self.map_df = self.map_df.drop(columns=['id_traj', 'max_train', 'max_valid'])

        # Reset ratio of images used for supervised training
        if  (self.ratio_sup > 0) and (self.ratio_sup < 1):
            self.map_df.loc[:, 'subset_id'] = self.map_df.groupby(['area_id', 'trajectory']).cumcount()
            self.map_df.loc[:, 'subset'] = ''
            mod = int(1/self.ratio_sup)
            sup = int(mod-1)
            # Reserve 10% of data for supervised training
            self.map_df.loc[self.map_df['subset_id'] % mod == sup, 'subset'] = f'{split}_supervised'
            self.map_df.loc[self.map_df['subset'] != f'{split}_supervised', 'subset'] = split
            self.map_df = self.map_df.drop(columns=['subset_id'])
            print(f"{round((len(self.map_df[self.map_df['subset'].str.contains('supervised')])/len(self.map_df[~self.map_df['subset'].str.contains('supervised')]))*100, 2)}% ({len(self.map_df[self.map_df['subset'].str.contains('supervised')])}/{len(self.map_df[~self.map_df['subset'].str.contains('supervised')])}) of images in the {split} set remain for supervised learning.")
        elif supervised and self.ratio_sup == 1:
            self.map_df.loc[:, 'subset'] = f'{split}_supervised'

        if 'subset' in self.map_df.columns:
            if supervised: # subset is not None:
                self.map_df = self.map_df[self.map_df['subset'] == f'{split}_supervised'].reset_index(drop=True).copy()
            else:
                self.map_df = self.map_df[self.map_df['subset'].str.contains(split)].reset_index(drop=True).copy()

        if (self.sample_within_image > 1) and (self.sample_within_image < ascan_per_group):
            self.map_df.loc[:, 'img_idx_start'] = self.map_df.loc[:, 'idx_start']
            self.map_df.loc[:, 'img_idx_end'] = self.map_df.loc[:, 'idx_end']
            new_img_count = round(ascan_per_group / self.sample_within_image) # Get new count of images, faire comme si les images étaient générées avec le new ascan count
            self.map_df = pd.concat([self.map_df]*new_img_count, axis=0).sort_index().reset_index(drop=True)
            # self.map_df = self.map_df.loc[self.map_df['idx_start'] <= self.map_df['idx_end'] - ascan_per_group, :] # Comment after debug
            self.map_df.loc[:, 'img_idx_start'] = [randint(0, ascan_per_group-self.sample_within_image) for _ in self.map_df.index.tolist()]
            self.map_df.loc[:, 'img_idx_end'] = self.map_df.loc[:, 'img_idx_start'] + self.sample_within_image
        elif self.sample_within_image == -1:
            pass
        else:
            print(f'Cannot sample {self.sample_within_image} A-scans from within {ascan_per_group} A-scan images')

        if self.pre_sample < 1:
            self.map_df.loc[:, 'area'] = [s.parts[1] for s in self.map_df.loc[:, 'img_relative_path']]
            self.map_df.loc[:, 'traj'] = ['_'.join(s.stem.split('_')[:-2]) for s in
                                          self.map_df.loc[:, 'img_relative_path']]
            self.map_df = self.map_df.groupby(['label_str', 'area', 'traj']).sample(frac=self.pre_sample)
            self.map_df = self.map_df.drop(columns=['area', 'traj'])
            self.map_df = self.map_df.reset_index(drop=True)
        """
        elif self.pre_sample == 2:
            # Remove images from sin traj
            print(f"Removing images from sin trajectory...")
            self.map_df = self.map_df[~self.map_df['trajectory'].str.contains('sin')].copy()
        """

        if self.use_iipp and self.num_same_area < 1:
            # Make the number of imgs per area even
            self.map_df['pair_id'] = self.map_df.groupby('area_id').cumcount()
            rm_rows_idx = self.map_df.reset_index().groupby('area_id').last()
            rm_rows_idx = rm_rows_idx[rm_rows_idx['pair_id'] % 2 == 0]['index'].tolist()
            self.map_df = self.map_df.loc[~self.map_df.index.isin(rm_rows_idx)].copy()
            self.create_iipp_map_df()
        elif self.use_iipp and self.num_same_area >=2:
            # Sample 1/num_same_area of map_df
            map_df = self.map_df.groupby('area_id').sample(frac=1/self.num_same_area).sort_index()
            self.map_df_sampling = self.map_df[~self.map_df.index.isin(map_df.index)].reset_index(drop=True).copy()
            self.map_df = map_df.reset_index(drop=True).copy()
            # Add weights to sampling df
            self.map_df_sampling['weights'] = 1.0

        if self.pre_shuffle:
            self.map_df = self.map_df.sample(frac=1).reset_index(drop=True)

        # Print number of images per area per label
        print(f"Image distribution for the {split} set")
        if self.map_df_sampling is None:
            print(self.map_df.sort_values('area').groupby('label_str').agg({'img_relative_path': 'count', 'area': lambda x: ', '.join(x.unique())}).rename(columns={'img_relative_path': 'n_imgs', 'area': 'areas'}))
        else:
            # Merge both back and get stats
            map_df_all = pd.concat([self.map_df, self.map_df_sampling], axis=0)
            print(map_df_all.sort_values('area').groupby('label_str').agg(
                {'img_relative_path': 'count', 'area': lambda x: ', '.join(x.unique())}).rename(
                columns={'img_relative_path': 'n_imgs', 'area': 'areas'}))

    def __len__(self):
        if self.use_iipp and self.num_same_area < 1:
            return len(self.map_df['pair_id'].unique())
        else:
            return len(self.map_df)

    def __getitem__(self, idx):
        if self.use_iipp:
            # For BYOL
            if self.num_same_area < 1:
                scan_paths = [self.root.joinpath(img_rel_path) for img_rel_path in
                              self.map_df.loc[self.map_df['pair_id'] == idx, 'img_relative_path'].tolist()]
                label = int(self.map_df.loc[self.map_df['pair_id'] == idx, 'label'].unique())
                extra_idx = []
            # For SimCLR, self.num_same_area = min 2
            else:
                scan_path = [self.root.joinpath(self.map_df['img_relative_path'].iloc[idx])]
                area = self.map_df['area_id'].iloc[idx]
                extra_idx = self.map_df_sampling[self.map_df_sampling['area_id'] == area].sample(n=self.num_same_area-1, weights='weights').index.tolist()
                scan_path_extra = [self.root.joinpath(self.map_df_sampling.iloc[i]['img_relative_path']) for i in extra_idx]
                scan_paths = scan_path + scan_path_extra
                # Update weights
                self.map_df_sampling.loc[self.map_df_sampling.index.isin(extra_idx), 'weights'] = self.map_df_sampling.loc[self.map_df_sampling.index.isin(extra_idx), 'weights'] - 0.25
                self.map_df_sampling.loc[self.map_df_sampling['weights'] < 0, 'weights'] = 0 # Set weight to 0 if it becomes negative
                label = int(self.map_df['label'].iloc[idx])
            # Get associated start and end idx within image
            if self.sample_within_image > 1:
                img_idx_start = [self.map_df['img_idx_start'].iloc[idx]]
                img_idx_end = [self.map_df['img_idx_end'].iloc[idx]]
                img_idx_start = img_idx_start + self.map_df.loc[self.map_df.index.isin(extra_idx), 'img_idx_start'].tolist()
                img_idx_end = img_idx_end + self.map_df.loc[self.map_df.index.isin(extra_idx), 'img_idx_end'].tolist()
            else:
                img_idx_start = []
                img_idx_end = []

            # Read M-Scan
            data = [cv2.imread(scan_path) for scan_path in scan_paths]  # shape: (512, 512, 3)
            # Crop image to designed idx start and end
            if self.sample_within_image > 1:
                for i, (idx0, idx1) in enumerate(zip(img_idx_start, img_idx_end)):
                    data[i] = data[i][:, idx0:idx1, :]
            data = [Image.fromarray(d) for d in data]

            if self.show:
                data_og = data.copy()

            # Apply transforms
            if self.transforms is not None:
                data = [self.transforms(d) for d in data]

            # Concat along channel dimension
            if self.num_same_area < 1:
                # For BYOL
                data = torch.cat(data, 0)
            else:
                # For SimCLR
                data = [[data[i][j] for i in range(len(data))] for j in range(len(data[0]))] # Reshape to get a sub-list for each transform
                data = [torch.cat(data[i], dim=0) for i in range(len(data))] # Concat all images per transform

            if self.show:
                data_tr = [data[i,:,:] for i in range(data.shape[0])]
                fig, axs = plt.subplots(2,2)
                for i, ax in enumerate(axs.flatten()):
                    if i < 2:
                        ax.imshow(data_og[i])
                        ax.title.set_text(f'OG img {i}')
                    else:
                        ax.imshow(data_tr[i-2])
                        ax.title.set_text(f'tr img {i-2}')
                fig.show()
                print()

        else:
            scan_path = self.root.joinpath(self.map_df['img_relative_path'].iloc[idx])
            label = int(self.map_df['label'].iloc[idx])
            data = cv2.imread(scan_path)  # shape: (512, 512, 3)
            # Crop image to designed idx start and end
            if self.sample_within_image > 1:
                img_idx_start = self.map_df['img_idx_start'].iloc[idx]
                img_idx_end = self.map_df['img_idx_end'].iloc[idx]
                data = data[:, img_idx_start:img_idx_end, :]
            try:
                data = Image.fromarray(data)
            except AttributeError:
                print(f"Problem with idx{idx}: {scan_path}")

            # Apply transforms
            if self.transforms is not None:
                data = self.transforms(data)

        # Convert labels to one-hot
        # if len(self.label_dict) > 2:
        #     label = torch.nn.functional.one_hot(torch.tensor(label), len(self.label_dict)).to(torch.float16)
        # else:
        #     label = torch.Tensor([label]).to(torch.float16)
        label = torch.nn.functional.one_hot(torch.tensor(label), len(self.label_dict)).to(torch.float16)
        # Add metadata if iipp is used
        if self.use_iipp:
            metadata = self.map_df['area_id'].iloc[idx]
            return [data, metadata], label
        else:
            return data, label

    def create_iipp_map_df(self):
        # Set random pair ids
        map_df1 = self.map_df.groupby('area_id').sample(frac=0.5).copy()
        map_df1['pair_id'] = np.arange(len(map_df1))
        map_df2 = self.map_df.loc[~self.map_df.index.isin(map_df1.index)].sort_values('area_id').copy()
        map_df2['pair_id'] = np.arange(len(map_df2))
        self.map_df = pd.concat([map_df1, map_df2], axis=0).sort_index()

    def reset_sampling_weights(self):
        self.map_df_sampling.loc[:, 'weights'] = 1


def get_oct_data_loaders(root_path:pathlib.Path, args:argparse.Namespace, batch_size:int, train_aug: list, test_aug:list,
                         mean:list, std:list, supervised=False, ratio_sup=1, shuffle=False, seq_split=False, overwrite_split=None):
    img_transforms = [v2.ToTensor(),
                      v2.Resize((args.img_reshape, args.img_reshape), interpolation=InterpolationMode.BILINEAR),
                      NormTransform()
    ]
    if args.img_channel == 1:
        img_transforms.append(transforms.Grayscale())
    split_names = ['train', 'valid', 'test']
    if 'oct' in args.dataset_name and supervised:
        split_names = [f'{s}_supervised' for s in split_names]
    # Check if RandomEqualize is in train_aug
    req_idx = [i for i in range(len(train_aug)) if isinstance(train_aug[i], torchvision.transforms.transforms.RandomEqualize)]
    if len(req_idx) > 0:
        req_idx = req_idx[0]
        train_augs = [train_aug[req_idx]] + img_transforms + train_aug[:req_idx] + train_aug[req_idx+1:]
    else:
        train_augs = img_transforms + train_aug
    # Check if RandomEqualize is in test_aug
    req_idx = [i for i in range(len(test_aug)) if
               isinstance(test_aug[i], torchvision.transforms.transforms.RandomEqualize)]
    if len(req_idx) > 0:
        req_idx = req_idx[0]
        test_augs = [test_aug[req_idx]] + img_transforms + test_aug[:req_idx] + test_aug[req_idx + 1:]
    else:
        test_augs = img_transforms + test_aug

    train_dataset = OCTDataset(root_path, split_names[0],
                               args.map_df_paths, args.labels_dict,
                               ch_in=args.img_channel,
                               sample_within_image=args.sample_within_image,
                               use_iipp=False, # args.use_iipp,
                               num_same_area=-1,
                               transforms=transforms.Compose(train_augs),
                               pre_sample=args.dataset_sample,
                               seq_split=seq_split,
                               ratio_sup=ratio_sup,
                               overwrite_split=overwrite_split)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=0, drop_last=False, shuffle=shuffle)

    valid_dataset = OCTDataset(root_path, split_names[1],
                               args.map_df_paths, args.labels_dict,
                               ch_in=args.img_channel,
                               sample_within_image=args.sample_within_image,
                               use_iipp=False,  # args.use_iipp,
                               num_same_area=-1,
                               transforms=transforms.Compose(test_augs),
                               pre_sample=args.dataset_sample,
                               seq_split=seq_split,
                               ratio_sup=ratio_sup,
                               overwrite_split=overwrite_split)

    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                              num_workers=0, drop_last=False, shuffle=False)

    test_dataset = OCTDataset(root_path, split_names[2],
                              args.map_df_paths, args.labels_dict,
                              ch_in=args.img_channel,
                              sample_within_image=args.sample_within_image,
                              use_iipp=False,
                              num_same_area=-1,
                              transforms=transforms.Compose(test_augs),
                              pre_sample=args.dataset_sample,
                              seq_split=seq_split,
                              ratio_sup=ratio_sup,
                              overwrite_split=overwrite_split)

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             num_workers=0, drop_last=False, shuffle=False)
    return train_loader, valid_loader, test_loader


def get_supervised_oct_data_loaders(root_path:pathlib.Path, args:argparse.Namespace, batch_size:int, train_aug: list, test_aug:list,
                                    mean:list, std:list, supervised=True, ratio_sup=1, shuffle=False, seq_split=False, overwrite_split=None):
    return get_oct_data_loaders(root_path, args, batch_size, train_aug, test_aug, mean, std, True, ratio_sup, shuffle, seq_split, overwrite_split)


def get_cross_valid_splits(args:argparse.Namespace, k: int) -> list:
    """
    Take the
    :param root_path: dataset root path
    :param args: dataset arguments
    :param k: Number of splits
    :return: List of splits {'train': [], 'valid': [], 'test': []}
    """
    subsets = ['train', 'valid', 'test']
    splits = []
    # Load mapping dfs
    map_dfs = pd.concat([pd.read_csv(pathlib.Path(p)) for p in args.map_df_paths.values()], axis=0)
    map_dfs.loc[:, 'img_relative_path'] = [pathlib.Path(p) for p in map_dfs['img_relative_path']]
    map_dfs.loc[:, 'area'] = [p.parts[-2] for p in map_dfs['img_relative_path']]
    # map_dfs.loc[:, 'pat'] = [int(re.sub(r'[^\d]+', '', p.parts[-2])) for p in map_dfs['img_relative_path']]

    # Get dict for current split
    base_split = {s: map_dfs[map_dfs['subset'] == s]['area'].unique().tolist() for s in subsets}
    splits.append(base_split)

    # Split info
    all_pats = sum(base_split.values(), [])
    all_pats = sorted(all_pats)
    pat_per_subset = {s: len(p) for s, p in base_split.items()}

    # Generate new k splits
    i = 0
    overall_i = 0
    while i < k-1:
        random.shuffle(all_pats)
        cv_split = {'train': all_pats[:pat_per_subset['train']],
                    'valid': all_pats[pat_per_subset['train']:pat_per_subset['train']+pat_per_subset['valid']],
                    'test': all_pats[-pat_per_subset['test']:]}
        # Check if all labels are in each split
        map_dfs_cv = map_dfs.copy()
        map_dfs_cv.loc[:, 'subset'] = ''
        for s, p in cv_split.items():
            map_dfs_cv.loc[map_dfs_cv['area'].isin(p), 'subset'] = s
        nb_lbls_per_split = map_dfs_cv.groupby('subset').agg({'label': 'nunique', 'area': lambda x: ', '.join(x.unique())}).rename(columns={'label': 'lbl_count', 'area': 'patients'})
        # DEBUG: Print lbls per split
        # print(f"iter {i}")
        # print(nb_lbls_per_split)
        # Add split to c-v splits
        if (len(nb_lbls_per_split['lbl_count'].unique()) == 1) and (nb_lbls_per_split['lbl_count'].unique().tolist()[0] == 2):
            # Check for label balance
            img_count_per_subset = map_dfs_cv.sort_values('area').groupby(['subset', 'label']).agg(
            {'img_relative_path': 'count', 'area': lambda x: ', '.join(x.unique())}).rename(
            columns={'img_relative_path': 'n_imgs', 'area': 'areas'}).reset_index()
            img_count_per_subset.loc[:, 'ratio_diff'] = np.nan
            for s in subsets:
                total_imgs_in_subset = img_count_per_subset.loc[img_count_per_subset['subset'] == s, 'n_imgs'].sum()
                ratio_healthy = img_count_per_subset.loc[(img_count_per_subset['subset'] == s) & (
                            img_count_per_subset['label'] == 'Healthy'), 'n_imgs'].iloc[0] / total_imgs_in_subset
                ratio_lesion = img_count_per_subset.loc[(img_count_per_subset['subset'] == s) & (
                        img_count_per_subset['label'] == 'Lesion'), 'n_imgs'].iloc[0] / total_imgs_in_subset
                img_count_per_subset.loc[img_count_per_subset['subset'] == s, 'ratio_diff'] = ratio_lesion - ratio_healthy
            # Overall ratio -> Healthy = 0.428, Lesion = 0.572
            # Base diff: Train_diff = 0.092, Valid_diff = 0.0898, Test_diff = 0.275
            if (img_count_per_subset['ratio_diff'].max() < 0.3) and (img_count_per_subset['ratio_diff'].min() > 0):
                splits.append(cv_split)
                assert len([p for ps in splits[i + 1].values() for p in ps]) == len(
                    list(set([p for ps in splits[i + 1].values() for p in ps])))
                i = i+1
                print(f"Found cross-validation split {i} after {overall_i} iterations.")
        overall_i = overall_i + 1

    return splits


def get_stl10_data_loaders(root_path, batch_size=128, shuffle=False, download=False):
    train_dataset = datasets.STL10(root_path, split='train', download=download,
                                   transform=transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=0, drop_last=False, shuffle=shuffle)

    test_dataset = datasets.STL10(root_path, split='test', download=download,
                                  transform=transforms.ToTensor())

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             num_workers=0, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader


def open_mat_file(file: pathlib.Path):
    with open(file, 'rb') as f:
        header = f.read(128).decode('utf-8')

    if "MATLAB 7.3" in header:
        with h5py.File(file, 'r') as mat_file:
            data = mat_file["mscan_real"][:]
    else:
        mat_contents = loadmat(file)
        data = mat_contents["mscan_real"]
    return data


def find_keywords(df: pd.DataFrame, col: str, keywords: list) -> pd.DataFrame:
    """
    Check whether each row in column col contains one of the given keywords.
    :param df: dataframe to be evaluated
    :param col: column in which to search for the keywords
    :param keywords: list of keywords to be found
    :return: Returns the df with extra columns: total keywords found + one per keyword
    """
    # Init new cols
    df = df.copy()
    df.loc[:, 'keyword_count'] = 0
    df.loc[:, [w for w in keywords]] = 0

    # Check if keyword is present
    for word in keywords:
        df.loc[df[col].str.contains(word, case=False), word] = 1

    # Count total number of keywords found
    df['keyword_count'] = np.sum(df[[w for w in keywords]], axis=1)

    return df


def build_image_root(ascan_per_group: int, pre_processing: dict) -> str:
    """
    Build image root folder name based on the Matlab script
    :param ascan_per_group: number of ascans per image
    :param pre_processing: dict with info on the selected pre-processing options
    :return:
    """
    p = f"{ascan_per_group}mscans"
    if pre_processing['no_noise']:
        p = f"{p}_noNoise"
    if pre_processing['use_movmean']:
        p = f"{p}_movmean"
    if pre_processing['use_speckle']:
        p = f"{p}_speckle"
    if pre_processing['ascan_sampling'] > 1:
        p = f"{p}_sample{pre_processing['ascan_sampling']}"
    return p


def movmean(a:np.ndarray, w:int) -> np.ndarray:
    # a_pre = np.expand_dims(np.mean(a[:, :2], axis=1), axis=-1)
    # a_post = np.expand_dims(np.mean(a[:, -2:], axis=1), axis=-1)
    # a_exp = np.concat([a_pre, a, a_post], axis=1)
    # a = signal.convolve2d(a_exp, np.ones((1, w)), 'valid') / w
    a = signal.fftconvolve(a, np.ones((1, w)), mode='same')/w
    return a


class RandomWrapAround:
    """
    Used to move the place where the tissue structure is seen on the image (reproduce probe vertical mouvement)
    """
    def __init__(self, dim=-1, p=1.0):
        """
        dim: dimension along which to wrap (default: last dimension, e.g. width / A-scan)
        p: probability of applying the transform
        """
        self.dim = dim
        self.p = p

    def __call__(self, x):
        """
        x: torch.Tensor (e.g. [C, H, W] or [H, W])
        """
        if random.random() > self.p:
            return x

        size = x.size()[self.dim]
        shift = random.randint(0, size - 1)

        if shift == 0:
            return x

        # split + concatenate (circular shift)
        return torch.cat(
            [x.narrow(self.dim, shift, size - shift),
             x.narrow(self.dim, 0, shift)],
            dim=self.dim
        )


class NormTransform(torch.nn.Module):
    """
    Convert tensor type from uint8 to float32, and divide by 255
    """
    def forward(self, img):
        img = img.to(torch.float32)
        img = img - img.mean()
        img = img / img.std()
        return img


if __name__ == '__main__':
    config_file = pathlib.Path("config_windows.yaml")
    configs = utils.load_configs(config_file)
    dataset_root = pathlib.Path(configs['data']['dataset_root'])
    labels = configs['data']['labels']
    labels_dict = {i: lbl for i, lbl in enumerate(labels)}
    ascan_per_group = configs['data']['ascan_per_group']
    use_mini_dataset = configs['data']['use_mini_dataset']
    map_df_paths ={split: dataset_root.joinpath(f"{split}{'Mini' if use_mini_dataset else ''}_mapping_{ascan_per_group}scans.csv") for split in ['train', 'valid', 'test']}


    # Create dataset object
    print("Initializing dataset object...")
    split = 'train'
    # Create dataset objects
    transforms = v2.Compose([NormTransform()])
    dataset_train = OCTDataset(dataset_root, split, map_df_paths, labels_dict, transforms)
    dataset_valid = OCTDataset(dataset_root, split, map_df_paths, labels_dict, transforms)
    dataset_test = OCTDataset(dataset_root, split, map_df_paths, labels_dict, transforms)

    print("Creating dataloader...")
    dataloader_train = DataLoader(dataset_train, batch_size=4, shuffle=True, num_workers=0)
    dataloader_valid = DataLoader(dataset_valid, batch_size=4, shuffle=False, num_workers=0)
    dataloader_test = DataLoader(dataset_test, batch_size=4, shuffle=False, num_workers=0)

    print("Fetching data...")
    for imgs, lbl in dataloader_train:
        # print(type(imgs))
        # print(type(lbl))
        img = (imgs[0, :, :, :].numpy().transpose((1, 2, 0)) * 255)[:, :, 0].astype('uint8')
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.title(labels_dict[lbl[0].item()])
        plt.tight_layout()
        plt.show(block=True)
        # Keep only first image in the batch
        # imgs = [img[0, :, :, :].squeeze(0) for img in imgs]
        # lbl = lbl[0, :].squeeze(0).numpy()

        # Plot images
        """
        plt.figure(figsize=(3*len(cam_inputs), 3))
        for i, (img, cam) in enumerate(zip(imgs, cam_inputs)):
            img = (img.numpy().transpose((1, 2, 0)) * 255).astype('uint8')
            plt.subplot(1, len(cam_inputs), i+1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(cam)
        plt_title = f'Position: {np.round(lbl[:3], 3)}\nRx: {euler[0]}   Ry: {euler[1]}    Rz: {euler[2]}'
        plt.suptitle(plt_title)
        plt.tight_layout()
        plt.show(block=True)
        """
