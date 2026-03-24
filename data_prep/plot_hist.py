import argparse
import pathlib
import sys
from sys import platform

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
mean['oct'] = [x /255 for x in [42.573, 42.573, 42.573]]

std['cifar10'] = [x / 255 for x in [63.0, 62.1, 66.7]]
std['cifar100'] = [x / 255 for x in [68.2,  65.4,  70.4]]
std['stl10'] = [0.229, 0.224, 0.225]
std['npy'] = [0.229, 0.224, 0.225]
std['npy224'] = [0.229, 0.224, 0.225]
std['oct'] = [x /255 for x in [26.688, 26.688, 26.688]]

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

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
        dataset_path = pathlib.Path(configs['SimCLR']['dataset_path_linux'])
    elif platform == "win32":
        dataset_root = pathlib.Path(configs['data']['dataset_root_windows'])
        dataset_path = pathlib.Path(configs['SimCLR']['dataset_path_windows'])
    # chkpt_file = pathlib.Path('runs/Sep18_19-15-48_ilmare/checkpoint_0200.pth.tar')
    # chkpt_file = pathlib.Path('runs/Sep20_09-16-58_ilmare/checkpoint_best_top1.pth')
    labels = configs['data']['labels']
    trajectories = configs['data']['trajectories']
    ascan_per_group = configs['data']['ascan_per_group']
    pre_processing = Dict(configs['data']['pre_processing'])
    use_mini_dataset = configs['data']['use_mini_dataset']
    overwrite_labels_path = pathlib.Path(configs['data']['overwrite_labels'])
    mean['oct'] = 3 * [configs['data']['img_mean'] / 255]
    std['oct'] = 3 * [configs['data']['img_std'] / 255]
    img_size_dict['oct'] = (512, ascan_per_group)
    num_cluster_dict['oct'] = len(labels)

    ### Convert config file values to the args variable equivalent (match the format of the existing code)
    print("Assigning config values to corresponding args variables...")
    # Dataset
    args.dataset_name = 'oct'
    args.data = pathlib.Path(dataset_path).joinpath(
        'OCT_lab_data' if args.dataset_name == 'oct' else args.dataset_name)
    image_root = build_image_root(ascan_per_group, pre_processing)
    print(f"dataset image root: {args.data.joinpath(image_root)}")
    args.labels_dict = {i: lbl for i, lbl in enumerate(labels)}
    new_lbl_str = f'{overwrite_labels_path.stem}_' if overwrite_labels_path is not None else ''
    traj_str = f"{''.join([t.capitalize() for t in trajectories])}_" if len(trajectories) < 3 else ''
    args.map_df_paths = {
        split: args.data.joinpath(image_root).joinpath(
            f"{split}{'Mini' if use_mini_dataset else ''}_mapping_{new_lbl_str}{traj_str}{ascan_per_group}scans.csv")
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
    save_plots = False

    # Load and merge mapping dfs
    usup_loaders = get_oct_data_loaders(args.data, args, 1,
                                        mean=mean[args.dataset_name],
                                        std=std[args.dataset_name],
                                        shuffle=False)
    sup_loaders = get_supervised_oct_data_loaders(args.data, args, 1,
                                                  mean=mean[args.dataset_name],
                                                  std=std[args.dataset_name],
                                                  shuffle=False)
    usup_dfs = [l.dataset.map_df for l in usup_loaders]
    sup_dfs = [l.dataset.map_df for l in sup_loaders]
    map_df = pd.concat([*usup_dfs, *sup_dfs], axis=0)

    # Filter to keep only s8 trajectories
    map_df = map_df[map_df['trajectory'].str.contains('_s8')].copy()

    # Re-number area_id for merged df
    map_df['area_id'] = map_df.groupby(['label', 'area']).ngroup()
    map_df.loc[map_df['trajectory'].str.contains('line'), 'traj_type'] = 'line'
    map_df.loc[map_df['trajectory'].str.contains('sine'), 'traj_type'] = 'sine'
    map_df.loc[map_df['trajectory'].str.contains('jitter'), 'traj_type'] = 'jitter'

    # Prep hist df (group per traj)
    hist_df = map_df.groupby('trajectory').agg(img_count=('img_relative_path', 'count'), label=('label', 'mean'), area_id=('area_id', 'mean')).reset_index()
    hist_df = pd.merge(hist_df, map_df.drop_duplicates(subset=['trajectory'], keep='first')[['trajectory', 'traj_type', 'label', 'label_str', 'area_id', 'area', 'subset']], on=['trajectory', 'label', 'area_id'], how='left')

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

    # Convert counts to density
    # https://matplotlib.org/stable/gallery/statistics/histogram_normalization.html
    # density = counts / (sum(counts) * np.diff(bins))
    # bins = np.arange(256)
    # hist_df.loc[:, 'density'] = [h / sum(h) * np.diff(bins) for h in hist_df['hist'].tolist()]

    # Define plot variables
    bin_uthresh = np.arange(255).tolist()
    w, x = 0.3, np.arange(len(bin_uthresh))
    hist_folder = args.data.joinpath('hist_per_traj')
    if not hist_folder.exists():
        hist_folder.mkdir()

    # Compare per traj for each area (Use KS-test due to sparse histograms)
    ks_per_traj = []
    for aid in tqdm(hist_df['area_id'].unique(), desc='Plotting per trajectory'):
        aid_df = hist_df[hist_df['area_id'] == aid].copy()
        # Plot and save histograms per area
        if save_plots:
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.bar(x-w/3, aid_df['hist'].iloc[0]/aid_df['hist'].iloc[0].sum(), width=w, label=aid_df['traj_type'].iloc[0])
            ax.bar(x , aid_df['hist'].iloc[1]/aid_df['hist'].iloc[1].sum(), width=w, label=aid_df['traj_type'].iloc[1])
            ax.bar(x+w/3, aid_df['hist'].iloc[2]/aid_df['hist'].iloc[2].sum(), width=w, label=aid_df['traj_type'].iloc[2])
            ax.set_xticks(x[::50])
            ax.set_xlabel('Pixel value')
            ax.set_ylabel('Pixel frequency')
            ax.set_title(f"Normalized histogram per trajectory on {aid_df['label_str'].iloc[0]} {aid_df['area'].iloc[0]}")
            ax.legend(loc='upper left')
            hist_path = hist_folder.joinpath(f"{aid_df['label_str'].iloc[0]}_{aid_df['area'].iloc[0]}.jpg")
            plt.savefig(hist_path)
            # plt.show()
            plt.close()
        for combi in [(0, 1), (0, 2), (1, 2)]:
            h1 = aid_df['hist'].iloc[combi[0]]
            h2 = aid_df['hist'].iloc[combi[1]]
            # Use Kolmogorov-Smirnov test
            # H0: two samples were drawn from the same distribution
            # if p<0.05, reject H0
            ks_test = ks_2samp(h1, h2)
            ks_per_traj.append({'area_id': aid,
                                'traj1': aid_df['trajectory'].iloc[combi[0]],
                                'traj2': aid_df['trajectory'].iloc[combi[1]],
                                'ks_test': ks_test.statistic,
                                'pvalue': ks_test.pvalue})
    ks_per_traj = pd.DataFrame.from_dict(ks_per_traj, orient='columns')
    ks_per_traj.to_csv(hist_folder.joinpath('ks_per_traj.csv'), index=False)
    ks_prob = ks_per_traj[ks_per_traj['pvalue'] < 0.05].sort_values('pvalue')
    print(f"{len(ks_prob)} trajectory pairs within the same area that have a KS pvalue< 0.05.")
    print(ks_prob[ks_prob['pvalue'] < 0.05][['traj1', 'traj2', 'pvalue']])

    # Update hist to get the mean hist per area
    hist_per_area_df = hist_df.groupby('area_id').agg({'img_count': 'sum', 'label': 'mean', 'hist': 'sum'}).reset_index()
    hist_per_area_df = pd.merge(hist_per_area_df, hist_df[['area_id', 'area', 'label', 'label_str', 'subset']].drop_duplicates(subset=['area_id', 'label']), on=['area_id', 'label'], how='left')

    # Compare per area on each organ (Use KS-test due to sparse histograms)
    hist_folder = args.data.joinpath('hist_per_area')
    if not hist_folder.exists():
        hist_folder.mkdir()
    subset_marker = {'train': 'o', 'valid': 's', 'test': 'x'}
    ks_per_area = []
    # Find pairs per organ
    for oid in tqdm(hist_per_area_df['label'].unique(), desc='Plotting per area'):
        oid_df = hist_per_area_df[hist_per_area_df['label'] == oid].copy()
        # Plot and save histograms per area
        if save_plots:
            fig, ax = plt.subplots(figsize=(11, 10))
            for i in range(len(oid_df)):
                plt_hist = oid_df['hist'].iloc[i] / oid_df['hist'].iloc[i].sum()
                plt_hist[plt_hist == 0] = np.nan
                ax.scatter(x, plt_hist,
                           marker=subset_marker[oid_df['subset'].iloc[i]],
                           label=f"{oid_df['area'].iloc[i]} ({oid_df['subset'].iloc[i]})")
            ax.set_xticks(x[::50])
            ax.set_xlabel('Pixel value')
            ax.set_ylabel('Pixel frequency')
            ax.set_title(f"Normalized histogram per area on {oid_df['label_str'].iloc[0]}")
            ax.legend(loc='upper left')
            hist_path = hist_folder.joinpath(f"{oid_df['label_str'].iloc[0]}.jpg")
            plt.savefig(hist_path)
            # plt.show()
            plt.close()
        combis = list(itertools.permutations(list(range(len(oid_df))), 2))
        combis = [tuple(sorted(c)) for c in combis]
        combis = list(set(combis))
        for combi in combis:
            h1 = oid_df['hist'].iloc[combi[0]]
            h2 = oid_df['hist'].iloc[combi[1]]
            # Use Kolmogorov-Smirnov test
            # H0: two samples were drawn from the same distribution
            # if p<0.05, reject H0
            ks_test = ks_2samp(h1, h2)
            ks_per_area.append({'label': oid,
                                'area1': oid_df['area'].iloc[combi[0]],
                                'area2': oid_df['area'].iloc[combi[1]],
                                'ks_test': ks_test.statistic,
                                'pvalue': ks_test.pvalue})
    ks_per_area = pd.DataFrame.from_dict(ks_per_area, orient='columns')
    ks_per_area = pd.merge(hist_per_area_df.drop_duplicates(subset=['label', 'label_str'])[['label', 'label_str']], ks_per_area, on='label', how='left')
    ks_per_area.to_csv(hist_folder.joinpath('ks_per_area.csv'), index=False)
    ks_prob = ks_per_area[ks_per_area['pvalue'] < 0.05].sort_values('pvalue')
    print(f"{len(ks_prob)} area pairs within the same organ that have a KS pvalue< 0.05.")
    print(ks_prob[ks_prob['pvalue'] < 0.05][['label_str', 'area1', 'area2', 'pvalue']])

    # Define new positive pairs
    print()
