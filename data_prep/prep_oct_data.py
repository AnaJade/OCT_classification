import argparse
import pathlib
from sys import platform

import numpy as np
from tqdm import tqdm
import pandas as pd
import cv2

import utils
from utils_data import open_mat_file, find_keywords

def save_as_jpg(mat_file_root: pathlib.Path, target_jpg_root:pathlib.Path, ascan_per_group: int, labels: list, mini_dataset: bool):
    """
    Load reconstructed scans (as .mat files) and save them into individual sections
    :param mat_file_root: path to .mat files
    :param target_jpg_root: path to jpg files
    :param ascan_per_group: number of A-scans within each group
    :param labels: list of labels in the dataset
    :param mini_dataset: if true, only s8 scans will be converted to jpg
    :return:
    """
    # Find all *_raw.mat files
    raw_mat_files = list(mat_file_root.rglob("*_raw.mat"))
    raw_mat_files = [f for f in raw_mat_files if
                     'reject' not in f.__str__()]  # Remove files if they are in the rejected folder

    # Filter file names if a mini dataset is desired
    if mini_dataset:
        print("Selecting files for the mini dataset...")
        # Keep only s8 files
        raw_mat_files = [fm for fm in raw_mat_files if '_s8_' in fm.__str__()]
        raw_files = raw_mat_files
        print(f"Keeping {len(raw_files)} areas over {len(labels)} labels.")
    else:
        raw_files = raw_mat_files

    # Convert file format
    print("Saving mat files to individual images...")
    # Create new folder if needed
    if not target_jpg_root.is_dir():
        target_jpg_root.mkdir(parents=True, exist_ok=True)

    for i, f in enumerate(raw_files):  # enumerate(raw_mat_files):
        print(f"File {i + 1}/{len(raw_files)}: Working on {f.name}")
        # Create target subfolder if needed
        target_img_subdir = target_jpg_root.joinpath(f.parts[-2])
        if not target_img_subdir.exists():
            target_img_subdir.mkdir(parents=True, exist_ok=True)

        # Check if file has already been converted
        img_file_name = target_img_subdir.joinpath(
            f"{'_'.join(f.stem.split('_')[:-2])}_0_{ascan_per_group}.jpg")
        if img_file_name.exists():
            print(f".jpg files for this trajectory have already been found.")
            print(f"Skipping to next trajectory...")
            continue

        # Load mat file
        mat_data = open_mat_file(f).T
        mat_data = cv2.normalize(mat_data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # Find how many images will be generated
        split_idx = [0] + [(i + 1) * ascan_per_group for i in
                           range(mat_data.shape[1] // ascan_per_group)] + [int(mat_data.shape[1])]

        # Save into individual images
        for j in tqdm(range(len(split_idx) - 1)):
            img_file_name = target_img_subdir.joinpath(
                f"{'_'.join(f.stem.split('_')[:-2])}_{split_idx[j]}_{split_idx[j + 1]}.jpg")
            img = mat_data[:, split_idx[j]:split_idx[j + 1]]
            try:
                cv2.imwrite(img_file_name, img)
            except cv2.error as e:
                print(f'Problem saving {img_file_name}:')
                print(e)
    print("Done saving .mat files as individual images.")


def get_jpg_dataset_info(dataset_root:pathlib.Path, jpg_root_path:pathlib.Path, labels: list) -> pd.DataFrame:
    """
    Get relative path, label, idx_start and idx_end for every available jpg image
    :param dataset_root: dataset root path
    :param jpg_root_path: jpg subdir within dataset root
    :param labels: list of possible labels
    :return: df with the info
    """
    print("Getting info on available .jpg files...")
    img_paths = {f.stem: list(f.rglob("*.jpg")) for f in list(jpg_root_path.iterdir())}
    img_paths = pd.DataFrame([(a, f) for a, files in img_paths.items() for f in files], columns=['folder', 'path'])
    img_paths['img_relative_path'] = [p.relative_to(dataset_root) for p in img_paths['path']]
    img_paths['trajectory'] = ['_'.join(p.stem.split('_')[:-2]) for p in img_paths['path']]
    img_labels = pd.from_dummies(
        find_keywords(img_paths[['img_relative_path', 'trajectory']], 'trajectory', labels).drop(
            columns=['trajectory']).set_index('img_relative_path')[[l for l in labels]]).reset_index().rename(
        columns={'': 'label'})
    img_paths = pd.merge(img_paths, img_labels, on='img_relative_path')
    img_paths['idx_start'] = [int(p.stem.split('_')[-2]) for p in img_paths['path']]
    img_paths['idx_end'] = [int(p.stem.split('_')[-1]) for p in img_paths['path']]

    return img_paths.sort_values(['trajectory', 'idx_start'], ascending=[True, True])


def split_train_valid_test(ds_split: list, jpg_files_info:pd.DataFrame, labels: list) -> pd.DataFrame:
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

    # Extract label
    folder_stats = folder_stats.reset_index().rename(columns={'index': 'folder'})
    folder_stats = find_keywords(folder_stats, 'folder', labels)
    folder_stats = folder_stats.set_index('folder')
    # Check for problem folders
    folder_problem = folder_stats[folder_stats['keyword_count'] != 1]
    if len(folder_problem) > 0:
        print(f"The following folders have more than one label: {list(folder_problem.index)}")
        print("Rename the folders to have only one label")
    folder_stats = pd.merge(folder_stats['col_count'],
                            pd.from_dummies(folder_stats[[l for l in labels]]), left_index=True, right_index=True)
    folder_stats = folder_stats.rename(columns={'': 'label'})
    folder_stats = folder_stats.sort_values(['label', 'col_count'], ascending=[True, False])
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


def create_mapping_dfs(jpg_root_path: pathlib.Path, df_split:pd.DataFrame, jpg_files_info:pd.DataFrame, ascan_per_group: int, mini_dataset:bool):
    """
    Create the mapping dataframes used by the dataset class
    :param jpg_root_path: Directory holding the jpg images
    :param df_split: df assigning each area to a subset (output of split_train_valid_test)
    :param jpg_files_info: df with the relative path, label, idx_start and idx_end of each available jpg image. Output of get_jpg_dataset_info
    :param ascan_per_group: number of A-scans within each grouped image
    :param mini_dataset: if true, only s8 scans will be converted to jpg
    :return:
    """
    # Create map df
    print("Creating the mapping dataframes...")
    sub_sets = df_split['split'].unique().tolist()
    for split in sub_sets:
        # traj_in_split = df_split[df_split['split'] == split]
        traj_in_split = df_split[df_split['split'] == split].reset_index()
        df_map_split = jpg_files_info[jpg_files_info['folder'].isin(traj_in_split['folder'])][
            ['img_relative_path', 'label', 'idx_start', 'idx_end']]

        # Save to csv
        map_df_path = f"{split}{'Mini' if mini_dataset else ''}_mapping_{ascan_per_group}scans.csv"
        print(f"Saving {split} mapping as {map_df_path}...")
        df_map_split.to_csv(jpg_root_path.joinpath(map_df_path), index=False)


def get_img_mean_std(map_df:pd.DataFrame, dataset_root: pathlib.Path) -> list[np.ndarray]:
    """
    Get the mean and std values over all images within the map_df
    :param map_df: mapping dataframe, with the relative path, label, idx_start and idx_end
    :param dataset_root: dataset root path
    :return: mean and std of the images within the map_df
    """
    mean = []
    std = []
    for i in tqdm(range(len(map_df))):
        # data.shape = (512, 5000, 3)
        data = cv2.imread(dataset_root.joinpath(map_df['img_relative_path'].iloc[i]))
        mean.append(np.mean(data, (0, 1)))
        std.append(np.std(data, (0, 1)))

    mean = np.mean(np.stack(mean, axis=0), 0)
    std = np.mean(np.stack(std, axis=0), 0)
    return [mean, std]


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
    use_mini_dataset = configs['data']['use_mini_dataset']

    # Dataset target path (if different then dataset_root)
    # target_path = pathlib.Path(r"C:\Users\anaja\OneDrive\Documents\Ecole\TUHH\Semester 6\Masterarbeit\OCT_lab_data")
    target_path = pathlib.Path(r"/data/Boudreault/OCT_lab_data")
    img_root_path = target_path.joinpath(f"{ascan_per_group}mscans")

    # Save images as individual .jpg chunks
    save_as_jpg(dataset_root, img_root_path, ascan_per_group, labels, use_mini_dataset)

    # Get jpg file info
    jpg_files_info = get_jpg_dataset_info(target_path, img_root_path, labels)

    # Split into train-valid-test
    df_split = split_train_valid_test(ds_split, jpg_files_info, labels)

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

