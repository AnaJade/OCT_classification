import argparse
import pathlib
from sys import platform

import re
import numpy as np
from addict import Dict
from tqdm import tqdm
import pandas as pd
import cv2

import utils
from utils_data import open_mat_file, find_keywords, build_image_root
from prep_oct_data import save_as_img, get_img_dataset_info, split_train_valid_test, create_mapping_dfs, get_img_mean_std


def merge_all_labels(lbl_root_path: pathlib.Path) -> pd.DataFrame:
    # Find all available label files and corresponding .mat files
    label_paths = list(lbl_root_path.rglob("*_labels.csv"))
    label_paths = pd.DataFrame(label_paths, columns=['label_path'])
    label_paths.loc[:, 'pat'] = [int(re.sub(r'[^\d]+', '', p.parts[-2])) for p in label_paths['label_path']]

    # Load all labels
    merged_labels = []
    for i in range(len(label_paths)):
        # Load csv
        labels = pd.read_csv(label_paths['label_path'].iloc[i])
        if len(labels) > 0:
            labels.insert(0, 'patient', label_paths['pat'].iloc[i])
            labels.loc[:, 'lbl_path'] = label_paths['label_path'].iloc[i].name
        merged_labels.append(labels)

    merged_labels = pd.concat(merged_labels, axis=0)

    return merged_labels


def get_clinical_img_dataset_info(dataset_root:pathlib.Path, jpg_root_path:pathlib.Path, labels: pd.DataFrame) -> pd.DataFrame:
    """
    Get relative path, label, idx_start and idx_end for every available jpg image
    :param dataset_root: dataset root path
    :param jpg_root_path: jpg subdir within dataset root
    :param labels: list of possible labels
    :return: df with the info
    """
    print("Getting info on available .jpg or .png files...")
    img_paths = {f.stem: list(f.rglob("*.[jpg][png]*")) for f in list(jpg_root_path.iterdir())}
    img_paths = pd.DataFrame([(a, f) for a, files in img_paths.items() for f in files], columns=['folder', 'path'])
    img_paths['img_relative_path'] = [p.relative_to(dataset_root) for p in img_paths['path']]
    img_paths['trajectory'] = ['_'.join(p.stem.split('_')[:-2]) for p in img_paths['path']]
    labels['trajectory'] = ['_'.join(p.split('_')[:-1]) for p in labels['lbl_path']]
    if labels.groupby('trajectory').agg(lbl_count=('Label', 'nunique'))['lbl_count'].max() == 1:
        labels = labels.drop_duplicates(subset='trajectory', keep='first')
        img_paths = pd.merge(img_paths, labels.rename(columns={'Label': 'label'})[['trajectory', 'label']], on='trajectory', how='left')
    img_paths['idx_start'] = [int(p.stem.split('_')[-2]) for p in img_paths['path']]
    img_paths['idx_end'] = [int(p.stem.split('_')[-1]) for p in img_paths['path']]

    return img_paths.sort_values(['trajectory', 'idx_start'], ascending=[True, True])


def split_clinical_train_valid_test(ds_split: list, jpg_files_info:pd.DataFrame, labels: list) -> pd.DataFrame:
    """
    Split available files into train-valid-test. Trajectories recorded on one area are handled as a unit,
    meaning that all trajectories recorded over one area will be put into the same subset. The A-scan count over each
    area is used to iteratively assign areas to subsets until the minimum error is reached.

    :param ds_split: ratio of data to be assigned to each subset
    :param jpg_files_info: df with the relative path, label, idx_start and idx_end of each available jpg image. Output of get_jpg_dataset_info
    :param labels: list of possible labels
    :return: df assigning each area to a subset, including history of assignments over the iterations
    """
    file_stats = jpg_files_info[['trajectory', 'folder', 'idx_end']].groupby('trajectory').last().rename(
        columns={'idx_end': 'col_count'})

    # Count columns per folder
    folder_stats = file_stats.reset_index(drop=True).groupby('folder').sum()

    # Add label to folder stats
    mscan_count_per_pat = jpg_files_info.groupby(['folder', 'trajectory', 'label']).agg(
        mscan_count=('idx_end', 'max')).reset_index()
    mscan_count_per_pat_lbl = mscan_count_per_pat.groupby(['folder', 'label']).agg({'mscan_count': 'sum'}).reset_index()
    lbl_stats = mscan_count_per_pat_lbl.groupby(['folder'])['label'].apply(lambda x: ', '.join(x)).rename('label')
    folder_stats = pd.merge(folder_stats, lbl_stats, left_index=True, right_index=True, how='left')
    labels = folder_stats['label'].unique().tolist()
    folder_stats_split = {l: folder_stats[folder_stats['label'] == l].copy() for l in labels}

    # Double check validity of
    if sum(ds_split) != 1:
        print(f'Desired [train, valid, test] split of {ds_split} does not sum to 1.')
        ds_split = [0.6, 0.15, 0.25]
        print(f'Split has been reset to {ds_split}.')

    # Split into train-valid-test based on ratio
    # Get desired total col count for all subsets
    target_col_count_per_split = {split: ds_split[i] * folder_stats['col_count'].sum() for i, split in
                                  enumerate(['train', 'valid', 'test'])}
    # Take one random folder per label, and assign it to train, test, valid in sequence
    sub_sets = ['train', 'valid', 'test']
    for l in labels:
        folder_stats_split[l].loc[:, 'split0'] = [sub_sets[i % 3] for i in range(len(folder_stats_split[l]))]
    df_split = pd.concat(folder_stats_split.values(), axis=0)

    # Adjust folder assignment to minimize gap between target column count and current split
    # Priority of reaching at least the min col count: train, test, valid
    print('Splitting folders into train-valid-test...')
    i = 1
    best_err = df_split['col_count'].sum()
    curr_err = df_split['col_count'].sum()
    while True:
        # Get col count diff to target (target - current)
        col_err = {
            split: target_col_count_per_split[split] - df_split[df_split[f'split{i - 1}'] == split]['col_count'].sum()
            for split in sub_sets}
        # Find the folder with the least amount of columns per split
        min_col_count_per_split = {split: df_split[df_split[f'split{i - 1}'] == split]['col_count'].min() for split in
                                   sub_sets}
        # Find the max error each split can have based on the current iteration (smallest change possible)
        max_err_i = {split: min([min_col_count_per_split[sm] for sm in sub_sets if sm != split]) for split in sub_sets}

        # Check whether more improvements can be made
        err_state = {'current_err': col_err, 'min_change_possible': max_err_i}
        err_state = pd.DataFrame.from_dict(err_state, orient='columns')
        # err_state['abs_err'] = abs(err_state['current_err'] - err_state['min_change_possible'])
        err_state['abs_err'] = abs(abs(err_state['current_err']) - err_state['min_change_possible'])
        err_state.loc[abs(err_state['current_err']) >= err_state['min_change_possible'], 'change_possible'] = 1

        # Update current total error
        curr_err = err_state['abs_err'].sum()

        # Exit condition
        if curr_err < best_err:
            best_err = curr_err
        # If more than one split can be improved, continue
        elif err_state['change_possible'].sum() > 1:
            pass
        # If no improvements are found, exit loop
        else:
            break

        print(f"Iter {i - 1} total error: {curr_err}")

        # Update split
        df_split[f'split{i}'] = df_split[f'split{i - 1}']
        # Find which splits will lose/gain a trajectory
        source_split = err_state['current_err'].idxmin()
        dest_split = err_state['current_err'].idxmax()
        # Check if any labels are off limits
        df_split_source = df_split[df_split[f'split{i}'] == source_split]
        df_split_source_label_count = df_split_source[[f'split{i}', 'label']].groupby('label').count()
        protected_labels = df_split_source_label_count[df_split_source_label_count[f'split{i}'] == 1].index.tolist()
        df_split_source = df_split_source[~df_split_source['label'].isin(protected_labels)]
        if len(df_split_source) == 0:
            print(f"No duplicate labels in the source ({source_split}) subset so no changes can be made.")
            print("Exiting loop...")
            break
        # Check which folder in other splits has the closest necessary value
        source_err = err_state.loc[source_split, 'abs_err']
        dest_err = err_state.loc[dest_split, 'abs_err']
        df_split_source.loc[:, 'diff_to_source_err'] = abs(df_split_source.loc[:, 'col_count'] - source_err)
        df_split_source.loc[:, 'diff_to_dest_err'] = abs(df_split_source.loc[:, 'col_count'] - dest_err)
        df_split_source.loc[:, 'total_diff_err'] = df_split_source.loc[:, 'diff_to_source_err'] + df_split_source.loc[:,
                                                                                                  'diff_to_dest_err']
        traj_to_change = df_split_source['total_diff_err'].idxmin()
        # Update traj split in f"split{i}" column
        df_split.loc[traj_to_change, f'split{i}'] = dest_split

        # Prepare for next iteration
        i = i + 1
    # Add resulting percentage split
    disp_err_state = err_state.copy().drop(columns=['abs_err', 'min_change_possible', 'change_possible'])
    # df_split_final = df_split[[c for i, c in enumerate(df_split.columns) if i in [0, 1, len(df_split.columns)-1]]]  # Keep only final split column
    disp_err_state = pd.merge(disp_err_state, df_split.groupby(f'split{i - 1}').agg(col_count=('col_count', 'sum')),
                              left_index=True, right_index=True)
    disp_err_state['split_ratio'] = disp_err_state['col_count'] / disp_err_state['col_count'].sum()
    print("Done splitting into subsets")
    print("Minimum A-Scan count error (target - current): ")
    print(disp_err_state)
    print("\n")
    # Rename last split column to split
    df_split = df_split.rename(columns={f'split{i - 1}': 'split'})
    df_split.to_excel('df_split_dbg.xlsx')

    return df_split



if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        help='Path to the config file',
                        type=str)

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
    elif platform == "win32":
        dataset_root = pathlib.Path(configs['data']['dataset_root_windows'])
    ds_split = configs['data']['ds_split']
    labels = configs['data']['labels']
    ascan_per_group = configs['data']['ascan_per_group']
    pre_processing = Dict(configs['data']['pre_processing'])
    use_mini_dataset = configs['data']['use_mini_dataset']

    # Update labels if clinical data is used
    if 'clinical' in dataset_root.__str__():
        lbl_root_path = dataset_root.parent.joinpath('Labels')
        merged_labels = merge_all_labels(lbl_root_path)
        labels = merged_labels['Label'].unique().tolist()
        # Update settings
        pre_processing['no_noise'] = False # M-Scans have already been cropped to remove noise
        pre_processing['ascan_sampling'] = 1
        # Set target path
        target_path = pathlib.Path(r"X:\Boudreault\OCT_clinical_data")
        # target_path = pathlib.Path(r"/data/Boudreault/OCT_clinical_data")
        img_root_path = target_path.joinpath(build_image_root(ascan_per_group, pre_processing))
        # Get jpg file info
        jpg_files_info = get_clinical_img_dataset_info(target_path, img_root_path, merged_labels)

    else:
        # target_path = pathlib.Path(r"C:\Users\anaja\OneDrive\Documents\Ecole\TUHH\Semester 6\Masterarbeit\OCT_lab_data")
        target_path = pathlib.Path(r"/data/Boudreault/OCT_lab_data")
        img_root_path = target_path.joinpath(build_image_root(ascan_per_group, pre_processing))
        # Save images as individual .jpg chunks
        save_as_img(dataset_root, img_root_path, ascan_per_group, labels, use_mini_dataset)
        # Get jpg file info
        jpg_files_info = get_img_dataset_info(target_path, img_root_path, labels)

    # Update idx_end to idx_start + ascan_per_group for new Matlab-generated images
    #   Due to noise removal and sampling, idx_end - idx_start no longer equals ascan_per_group
    if ('noNoise' in img_root_path.stem) or ('sample' in img_root_path.stem):
        jpg_files_info.loc[:, 'idx_end'] = jpg_files_info.loc[:, 'idx_start'] + ascan_per_group-1

    # Split into train-valid-test
    df_split = split_clinical_train_valid_test(ds_split, jpg_files_info, labels)

    # Save mapping dfs
    create_mapping_dfs(img_root_path, df_split, jpg_files_info, ascan_per_group, use_mini_dataset)

    # Get mean and std of train set
    print("Calculating the mean and std of training images...")
    train_map_df = img_root_path.joinpath(
        f"train{'Mini' if use_mini_dataset else ''}_mapping_{ascan_per_group}scans.csv")
    train_map_df = pd.read_csv(train_map_df)
    mean, std = get_img_mean_std(train_map_df, target_path)
    print(f"Train dataset mean: {mean}")
    print(f"Train dataset std: {std}")

