"""
Take a list of annotation files and combine them.
Split all images based on the robot pose, and then split each pose into train-test sets
"""
import argparse
import pathlib

import numpy as np
from tqdm import tqdm
from collections.abc import Iterable

import pandas as pd
import cv2

import utils
from siamese_net import utils_data


def flatten(xs):
    """
    Recursively flatten a nested list. Taken from https://stackoverflow.com/q/2158395
    :param xs: nested list
    :return: flattened list
    """
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x


def split_imgs_by_pose(df: pd.DataFrame, bin_size: int) -> list:
    """
    Split the given images based on the mandible pose
    :param df: dataframe with the annotations to be split
    :param bin_size: bin size in mm
    :return: 3D list [Z, Y, X] with the df separated per pose
    """
    # Find the min and max values per axis
    min_pos = list(np.floor(df.loc[:, ['x', 'y', 'z']].min(axis=0)))
    max_pos = list(np.ceil(df.loc[:, ['x', 'y', 'z']].max(axis=0)))

    # Create the bin limits
    bins_x = list(np.ceil(np.linspace(start=min_pos[0], stop=max_pos[0],
                                      num=np.ceil((max_pos[0]-min_pos[0])/bin_size).astype(np.int32))))
    bins_y = list(np.ceil(np.linspace(start=min_pos[1], stop=max_pos[1],
                                      num=np.ceil((max_pos[1] - min_pos[1]) / bin_size).astype(np.int32))))
    bins_z = list(np.ceil(np.linspace(start=min_pos[2], stop=max_pos[2],
                                      num=np.ceil((max_pos[2] - min_pos[2]) / bin_size).astype(np.int32))))
    print(f'Bin count: {len(bins_x)-1} in X, {len(bins_y)-1} in Y, {len(bins_z)-1} in Z')
    # Split images based on the bins
    img_bins = []
    empty_bin_count = 0
    for zi in range(len(bins_z)-1):
        xy_bins = []
        df_xy = df[df['z'].between(bins_z[zi], bins_z[zi + 1])]
        for yi in range(len(bins_y)-1):
            x_bins = []
            df_x = df_xy[df_xy['y'].between(bins_y[yi], bins_y[yi + 1])]
            for xi in range(len(bins_x)-1):
                df_bin = df_x[df_x['x'].between(bins_x[xi], bins_x[xi + 1])]
                if len(df_bin) == 0:
                    empty_bin_count += 1
                    print(f'No images in X = [{bins_x[xi]}, {bins_x[xi + 1]}], '
                          f'Y = [{bins_y[yi]}, {bins_y[yi + 1]}], '
                          f'Z = [{bins_z[zi]}, {bins_z[zi+1]}]')
                x_bins.append(df_bin)
            xy_bins.append(x_bins)
        img_bins.append(xy_bins)
    print(f'{empty_bin_count}/{(len(bins_x)-1)*(len(bins_y)-1)*len(bins_z)-1} bins had no images')

    return img_bins


def split_df(df: pd.DataFrame, split_ratio: float, random=True, random_seed=100) -> list[pd.DataFrame]:
    """
    Split a pandas df in two based on the split ratio
    :param df: df to be split
    :param split_ratio: data proportion of the first df. Second df will have 1-split_ratio data
    :param random: whether to split randomly or not
    :param random_seed: random seed
    :return: list with the two dfs
    """
    if random:
        df_train = df.sample(frac=split_ratio, random_state=random_seed)
        df_test = df.drop(df_train.index)
    else:
        # Find splitting index
        split_id = np.floor(len(df)*split_ratio).astype(int)
        df_train = df.iloc[:split_id, :]
        df_test = df.iloc[split_id:, :]
    return [df_train, df_test]


def df_split_train_test(data: list | pd.DataFrame, train_ratio: float, random=True,
                        random_seed=100) -> list[pd.DataFrame]:
    """
    Split the given data into train and test sets based on the desired train ratio
    :param data: either a nested list of dataframes (ex.: output of split_imgs_by_pose), or directly a df
    :param train_ratio: proportion of images to use in the train set. 1-train_ratio will be used for the test set
    :param random: whether to split randomly
    :param random_seed: random seed
    :return:
    """
    # Set random seed
    np.random.seed(random_seed)
    if isinstance(data, list):
        # Flatten list and remove empty dfs
        data = [d for dxy in data for dx in dxy for d in dx if len(d) > 0]
        # Split remaining dfs
        data_split = [split_df(d, train_ratio, random, random_seed) for d in data]
        # Merge resulting dfs
        df_train = pd.concat([d[0] for d in data_split], axis=0)
        df_test = pd.concat([d[1] for d in data_split], axis=0)
    else:
        [df_train, df_test] = split_df(data, train_ratio, random, random_seed)
    # Remove any duplicates
    df_train = df_train[~df_train.index.duplicated(keep='first')]
    df_test = df_test[~df_test.index.duplicated(keep='first')]
    return [df_train, df_test]


if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path',
                        help='Path to the config file',
                        type=str)

    args = parser.parse_args()
    config_file = pathlib.Path(args.config_path)
    # config_file = pathlib.Path("data_prep/data_config.yaml")

    if not config_file.exists():
        print(f'Config file not found at {args.config_path}')
        raise SystemExit(1)
    configs = utils.load_configs(config_file)

    data_folder_path = pathlib.Path(configs['annotations']['data_folder_path'])
    traj_name = configs['annotations']['traj_name']
    traj_folder_path = data_folder_path.joinpath(traj_name)

    dataset_root = pathlib.Path(configs['images']['img_root'])
    anno_files = configs['merge_trajectories']['traj_to_merge']
    new_file_name_base = configs['merge_trajectories']['merged_file_name']
    reduce_rot = configs['merge_trajectories']['filter_rot']
    filter_oof = configs['merge_trajectories']['filter_oof']
    test_ratio = configs['merge_trajectories']['test_ratio']
    valid_ratio = configs['merge_trajectories']['valid_ratio']
    
    # Merge all annotation files together
    annotations = utils_data.merge_annotations(dataset_root, anno_files)

    # Filter out images where the mandible isn't fully visible
    if filter_oof:
        print('Filtering images to remove frames where the mandible is out of frame...')
        # annotations, removed_imgs = utils_data.filter_out_oof_mandible(dataset_root, annotations, 80)
        annotations, removed_imgs = utils_data.filter_out_oof_mandible_by_pixel_match(dataset_root, annotations,
                                                                                      [180, 121, 81], 10)
        print(f'Removed {len(removed_imgs)} images')

    # Filter data
    if reduce_rot:
        annotations = utils_data.filter_imgs_per_rotation_euler(annotations, None)

    # Split into bins
    print("Splitting images into bins...")
    print(f'Bin overview for {new_file_name_base}')
    split_imgs = split_imgs_by_pose(annotations, 10)

    # Split into train and test set
    print("Splitting images into train, valid and test sets...")
    [full_train_df, test_df] = df_split_train_test(split_imgs, (1-test_ratio))
    [train_df, valid_df] = df_split_train_test(full_train_df, (1-valid_ratio))

    # Save resulting df as csv
    print("Saving results...")
    train_file = dataset_root.joinpath(f'{new_file_name_base}_train.csv')
    valid_file = dataset_root.joinpath(f'{new_file_name_base}_valid.csv')
    test_file = dataset_root.joinpath(f'{new_file_name_base}_test.csv')
    trajectories_file = dataset_root.joinpath(f'{new_file_name_base}.txt')
    train_df.to_csv(train_file, index=True)
    valid_df.to_csv(valid_file, index=True)
    test_df.to_csv(test_file, index=True)
    with open(trajectories_file, 'w', encoding='utf-8') as f:
        f.writelines([f'{a}\n' for a in anno_files])
        f.close()

    print(f'# train set images: {len(train_df)}\n'
          f'# valid set images: {len(valid_df)}\n'
          f'# test set images:  {len(test_df)}')


