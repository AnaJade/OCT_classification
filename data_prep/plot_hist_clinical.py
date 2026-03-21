import argparse
import pathlib
import sys
from sys import platform
import re

import cv2
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import torch
from addict import Dict
from tqdm import tqdm
from torchvision import models

from scipy.stats import chisquare, ks_2samp
from scipy import stats

# Import utils
parent_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
import utils
from utils_data import get_oct_data_loaders, get_supervised_oct_data_loaders, build_image_root


# Img size and moco_dim (nb of classes) values based on the dataset
img_size_dict = {'stl10': 96,
                 'cifar10': 32,
                 'cifar100': 32}
num_cluster_dict = {'stl10': 10,
                    'cifar10': 10,
                    'cifar100': 100}

mean, std = {}, {}
mean['cifar10'] = [x / 255 for x in [125.3, 123.0, 113.9]]
mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]
mean['stl10'] = [0.485, 0.456, 0.406]
mean['npy'] = [0.485, 0.456, 0.406]
mean['npy224'] = [0.485, 0.456, 0.406]

std['cifar10'] = [x / 255 for x in [63.0, 62.1, 66.7]]
std['cifar100'] = [x / 255 for x in [68.2,  65.4,  70.4]]
std['stl10'] = [0.229, 0.224, 0.225]
std['npy'] = [0.229, 0.224, 0.225]
std['npy224'] = [0.229, 0.224, 0.225]


# Set up the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--config_path',
                    help='Path to the config file',
                    type=str)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.config_path is None:
        args.config_path = pathlib.Path('../config.yaml')
    config_file = pathlib.Path(args.config_path)

    if not config_file.exists():
        print(f'Config file not found at {args.config_path}')
        raise SystemExit(1)

    configs = utils.load_configs(config_file)
    if platform == "linux" or platform == "linux2":
        dataset_root = pathlib.Path(configs['data']['dataset_root_linux'])
        dataset_path = pathlib.Path(r"/data/Boudreault/OCT_clinical_data")
    elif platform == "win32":
        dataset_root = pathlib.Path(configs['data']['dataset_root_windows'])
        dataset_path = pathlib.Path(r"X:\Boudreault\OCT_clinical_data")
    labels = configs['data']['labels']
    trajectories = configs['data']['trajectories']
    ascan_per_group = configs['data']['ascan_per_group']
    pre_processing = Dict(configs['data']['pre_processing'])
    use_mini_dataset = configs['data']['use_mini_dataset']
    overwrite_labels = configs['data']['overwrite_labels']

    ### Convert config file values to the args variable equivalent (match the format of the existing code)
    print("Assigning config values to corresponding args variables...")
    # Dataset
    args.dataset_name = 'oct_clinical'
    mean[args.dataset_name] = 3 * [configs['data']['img_mean'] / 255]
    std[args.dataset_name] = 3 * [configs['data']['img_std'] / 255]
    img_size_dict[args.dataset_name] = (512, ascan_per_group)
    # Update variables for clinical data
    args.data = dataset_path
    labels = ['Healthy', 'Lesion']
    num_cluster_dict[args.dataset_name] = len(labels)
    # Update pre processing
    pre_processing['no_noise'] = False  # M-Scans have already been cropped to remove noise
    pre_processing['ascan_sampling'] = 1
    args.scan_no_noise = False
    args.scan_sampling = 1
    # Get other args
    image_root = build_image_root(ascan_per_group, pre_processing)
    print(f"dataset image root: {args.data.joinpath(image_root)}")
    args.labels_dict = {i: lbl for i, lbl in enumerate(labels)}
    args.map_df_paths = {
        split: args.data.joinpath(image_root).joinpath(
            f"{split}{'Mini' if use_mini_dataset else ''}_mapping_{ascan_per_group}scans.csv")
        for split in ['train', 'valid', 'test']}
    args.img_channel = 1
    if args.dataset_name != 'oct':
        args.img_channel = 3
    args.sample_within_image = -1
    args.img_reshape = None
    if args.img_reshape is not None:
        args.img_size = args.img_reshape
    else:
        args.img_size = 512  # BYOL requires square images, so all images will be reshaped to 512x512
    args.use_iipp = False
    args.num_same_area = -1
    args.use_simclr_augmentations = False
    args.ascan_per_group = ascan_per_group

    # Define other args as needed
    args.dataset_sample = 1
    save_plots = True

    # Load and merge mapping dfs
    map_dfs = {s: pd.read_csv(p) for s, p in args.map_df_paths.items()}
    for s, map_df in map_dfs.items():
        map_df.loc[:, 'subset'] = s
    map_df = pd.concat(map_dfs.values(), axis=0, ignore_index=True).rename(columns={'label': 'label_str'})
    map_df.loc[:, 'label'] = [0 if l == 'Healthy' else 1 for l in map_df['label_str']]
    map_df.loc[:, 'img_relative_path'] = [pathlib.Path(p) for p in map_df.loc[:, 'img_relative_path']]
    map_df.loc[:, 'pat'] = [int(re.sub(r'[^\d]+', '',p.stem.split('_')[0])) for p in map_df.loc[:, 'img_relative_path']]
    map_df.loc[:, 'trajectory'] = ['_'.join(p.stem.split('_')[:-3]) for p in map_df.loc[:, 'img_relative_path']]

    # Filter out unused recordings
    traj_to_remove = ['pat01_vc_re_run1',
                      'pat04_vc_le_run1',
                      'pat04_vc_le_run3',
                      'pat04_vc_re_run2',
                      'pat06_vc_re_run1',
                      'pat07_vc_re_run1',
                      'pat15_vc_le_run1']
    map_df = map_df[~map_df['trajectory'].str.contains('|'.join(f'{f}_' for f in traj_to_remove))].reset_index(drop=True).copy()

    # Prep hist df (group per traj)
    hist_df = map_df.groupby('trajectory').agg(img_count=('img_relative_path', 'count')).reset_index()
    hist_df = pd.merge(hist_df, map_df.drop_duplicates(subset=['trajectory'], keep='first')[['pat','trajectory', 'label', 'label_str', 'subset']], on=['trajectory'], how='left')
    # Find traj count per patient per label
    traj_count_per_pat = hist_df.groupby(['pat', 'label_str', 'label']).agg(traj_count=('trajectory', 'nunique'), img_count=('img_count', 'sum')).reset_index()

    # Loop over traj
    hist_dict = {}
    for traj in tqdm(hist_df['trajectory'], desc='Getting histograms'):
        # Get complete image path
        img_paths = map_df.loc[map_df['trajectory'] == traj, 'img_relative_path'].tolist()
        img_paths = [args.data.joinpath(rp) for rp in img_paths]
        # Load all images
        imgs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in img_paths]
        # Get mean A-scan values per img
        mean_ascans = [np.mean(s, axis=0) for s in imgs]
        del(imgs)
        # Get hist
        hist_traj = [np.histogram(a, bins=255, range=(0, 255), density=False) for a in mean_ascans]
        # Add hist for all images in traj
        hist_traj = np.vstack([h[0] for h in hist_traj])
        hist_traj = np.sum(hist_traj, axis=0)
        # Update hist dict
        hist_dict[traj] = hist_traj

    # Merge dict to hist_df
    hist_dict= {t: {'hist': h} for t, h in hist_dict.items()}
    # hist_dict = pd.DataFrame.from_dict(hist_dict, orient='index', columns=[f'h{i}' for i in range(255)]).reset_index().rename(columns={'index': 'trajectory'})
    hist_dict = pd.DataFrame.from_dict(hist_dict, orient='index').reset_index().rename(columns={'index': 'trajectory'})
    hist_df = pd.merge(hist_df, hist_dict, on='trajectory')

    # Define plot variables
    bin_uthresh = np.arange(255).tolist()
    x = np.arange(len(bin_uthresh))
    hist_folder = args.data.joinpath('hist_per_label')
    if not hist_folder.exists():
        hist_folder.mkdir()

    # Update hist to get the mean hist per patient per label
    hist_per_label_pat = hist_df.groupby(['label', 'pat']).agg({'img_count': 'sum', 'hist': 'sum'}).reset_index()
    hist_per_label_pat = pd.merge(hist_df[['pat', 'label', 'label_str', 'subset']].drop_duplicates(subset=['pat', 'label']), hist_per_label_pat, on=['pat', 'label'], how='right')

    for lbl in hist_df['label_str'].unique().tolist():
        hist_per_pat = hist_per_label_pat.loc[hist_per_label_pat['label_str'] == lbl, :].copy()
        ks_per_pat = []
        # Find pairs per patient
        if save_plots:
            fig, ax = plt.subplots(figsize=(11, 10))
            for pid in tqdm(hist_per_pat['pat'].unique(), desc=f'Plotting per patient ({lbl})'):
                pat_hist = hist_per_pat.loc[hist_per_pat['pat'] == pid, 'hist'].iloc[0].copy()
                # Plot and save histograms per area
                plt_hist = pat_hist / pat_hist.sum()
                plt_hist[plt_hist == 0] = np.nan
                ax.scatter(x, plt_hist,
                           marker='o',
                           label=f"pat{pid}")
                ax.set_xticks(x[::50])
                ax.set_xlabel('Pixel value')
                ax.set_ylabel('Pixel frequency')
                ax.set_title(f"Normalized histogram per patient on {lbl} tissue")
                ax.legend(loc='upper left')
            hist_path = hist_folder.joinpath(f"{lbl}.jpg")
            plt.savefig(hist_path)
            # plt.show()
            plt.close()
        combis = list(itertools.permutations(hist_per_pat['pat'].unique(), 2))
        combis = [tuple(sorted(c)) for c in combis]
        combis = list(set(combis))
        for combi in combis:
            h1 = hist_per_pat.loc[hist_per_pat['pat'] == combi[0], 'hist'].iloc[0]
            h2 = hist_per_pat.loc[hist_per_pat['pat'] == combi[1], 'hist'].iloc[0]
            # Use Kolmogorov-Smirnov test
            # H0: two samples were drawn from the same distribution
            # if p<0.05, reject H0
            ks_test = ks_2samp(h1, h2)
            ks_per_pat.append({'label': lbl,
                                'pat1': combi[0],
                                'pat2': combi[1],
                                'ks_test': ks_test.statistic,
                                'pvalue': ks_test.pvalue})
        ks_per_pat = pd.DataFrame.from_dict(ks_per_pat, orient='columns')
        # ks_per_pat = pd.merge(hist_per_pat.drop_duplicates(subset=['pat', 'label', 'label_str'])[['pat', 'label', 'label_str']], ks_per_pat, on='label', how='left')
        ks_per_pat.to_csv(hist_folder.joinpath(f'ks_per_pat{lbl}.csv'), index=False)
        ks_prob = ks_per_pat[ks_per_pat['pvalue'] < 0.05].sort_values('pvalue')
        print(f"{len(ks_prob)} patient pairs with {lbl} tissue that have a KS pvalue< 0.05.")
        print(ks_prob[ks_prob['pvalue'] < 0.05]) # [['label', 'pat1', 'pat2', 'pvalue']])

    # Define new positive pairs
    print()
