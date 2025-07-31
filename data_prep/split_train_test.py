import argparse
import pathlib
import time
import numpy as np
import pandas as pd
import random
from sys import platform
from collections import Counter

import utils
import utils_data

if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        help='Path to the config file',
                        type=str)

    args = parser.parse_args()
    if args.config_path is None:
        if platform == "linux" or platform == "linux2":
            args.config_path = pathlib.Path('../config.yaml')
        elif platform == "win32":
            args.config_path = pathlib.Path('../config_windows.yaml')
    config_file = pathlib.Path(args.config_path)
    if not config_file.exists():
        print(f'Config file not found at {args.config_path}')
        raise SystemExit(1)
    configs = utils.load_configs(config_file)

    dataset_root = pathlib.Path(configs['data']['dataset_root'])
    ds_split = configs['data']['ds_split']
    labels = configs['data']['labels']
    ascan_per_group = configs['data']['ascan_per_group']
    use_mini_dataset = configs['data']['use_mini_dataset']

    # Find all image files
    img_root_path = dataset_root.joinpath(f"{ascan_per_group}mscans")
    img_paths = {f.stem: list(f.rglob("*.jpg")) for f in list(img_root_path.iterdir())}
    img_paths = pd.DataFrame([(a, f) for a, files in img_paths.items() for f in files], columns=['folder', 'path'])
    img_paths['img_relative_path'] = [p.relative_to(dataset_root) for p in img_paths['path']]
    img_paths['trajectory'] = ['_'.join(p.stem.split('_')[:-2]) for p in img_paths['path']]
    img_labels = pd.from_dummies(
        utils_data.find_keywords(img_paths[['img_relative_path', 'trajectory']], 'trajectory', labels).drop(
            columns=['trajectory']).set_index('img_relative_path')[[l for l in labels]]).reset_index().rename(columns={'': 'label'})
    img_paths = pd.merge(img_paths, img_labels, on='img_relative_path')
    img_paths['idx_start'] = [int(p.stem.split('_')[-2]) for p in img_paths['path']]
    img_paths['idx_end'] = [int(p.stem.split('_')[-1]) for p in img_paths['path']]
    img_paths = img_paths.sort_values(['trajectory', 'idx_start'], ascending=[True, True])
    file_stats = img_paths[['trajectory', 'folder', 'idx_end']].groupby('trajectory').last().rename(
        columns={'idx_end': 'col_count'})

    # Get file info
    """
    print("Loading .npy files...")
    raw_npy_files = list(dataset_root.rglob("*_raw.npy"))
    file_stats = {}
    for i, fn in enumerate(raw_npy_files):
        file_stats[fn] = {}
        # Load npy file
        data = np.load(fn)
        # Save relevant info
        file_stats[fn]['folder'] = fn.parent.name
        file_stats[fn]['col_count'] = data.shape[0]
    file_stats = pd.DataFrame.from_dict(file_stats, orient='index')
    """

    # Count columns per folder
    folder_stats = file_stats.reset_index(drop=True).groupby('folder').sum()

    # DEBUG: Add other random entries
    """
    nb_folders = 30     # Random number
    folder_stats_dbg = [f"{random.sample(labels, 1)[0]}_{i}" for i in range(30)]
    # folder_stats_dbg = [dataset_root.joinpath(f) for f in folder_stats_dbg]
    folder_stats_dbg = {f: random.randint(3100, 45000) * 50 for f in folder_stats_dbg}
    folder_stats_dbg = pd.DataFrame.from_dict(folder_stats_dbg, orient='index').rename(columns={0:'col_count'})
    folder_stats = folder_stats_dbg.copy()
    """

    # Extract label
    folder_stats = folder_stats.reset_index().rename(columns={'index': 'folder'})
    folder_stats = utils_data.find_keywords(folder_stats, 'folder', labels)
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
    target_col_count_per_split = {split: ds_split[i]*folder_stats['col_count'].sum() for i, split in enumerate(['train', 'valid', 'test'])}
    # Take one random folder per label, and assign it to train, test, valid in sequence
    sub_sets = ['train', 'valid', 'test']
    for l in labels:
        folder_stats_split[l].loc[:, 'split0'] = [sub_sets[i%3] for i in range(len(folder_stats_split[l]))]
    df_split = pd.concat(folder_stats_split.values(), axis=0)

    # Adjust folder assignment to minimize gap between target column count and current split
    # Priority of reaching at least the min col count: train, test, valid
    print('Splitting folders into train-valid-test...')
    i = 1
    best_err = df_split['col_count'].sum()
    curr_err = df_split['col_count'].sum()
    while True:
        # Get col count diff to target (target - current)
        col_err = {split: target_col_count_per_split[split] - df_split[df_split[f'split{i - 1}'] == split]['col_count'].sum() for split in sub_sets}
        # Find the folder with the least amount of columns per split
        min_col_count_per_split = {split: df_split[df_split[f'split{i - 1}'] == split]['col_count'].min() for split in sub_sets}
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
        df_split[f'split{i}'] = df_split[f'split{i-1}']
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
        df_split_source.loc[:, 'total_diff_err'] = df_split_source.loc[:, 'diff_to_source_err'] + df_split_source.loc[:, 'diff_to_dest_err']
        traj_to_change = df_split_source['total_diff_err'].idxmin()
        # Update traj split in f"split{i}" column
        df_split.loc[traj_to_change, f'split{i}'] = dest_split

        # Prepare for next iteration
        i = i+1
    # Add resulting percentage split
    disp_err_state = err_state.copy().drop(columns=['abs_err', 'min_change_possible', 'change_possible'])
    # df_split_final = df_split[[c for i, c in enumerate(df_split.columns) if i in [0, 1, len(df_split.columns)-1]]]  # Keep only final split column
    disp_err_state = pd.merge(disp_err_state, df_split.groupby(f'split{i-1}').agg(col_count=('col_count', 'sum')), left_index=True, right_index=True)
    disp_err_state['split_ratio'] = disp_err_state['col_count']/disp_err_state['col_count'].sum()
    print("Done splitting into subsets")
    print("Minimum A-Scan count error (target - current): ")
    print(disp_err_state)
    print("\n")
    # Rename last split column to split
    df_split = df_split.rename(columns={f'split{i-1}': 'split'})
    df_split.to_excel('df_split_dbg.xlsx')

    # Create map df
    print("Creating the mapping dataframes...")
    for split in sub_sets:
        # traj_in_split = df_split[df_split['split'] == split]
        traj_in_split = df_split[df_split['split'] == split].reset_index()
        df_map_split = img_paths[img_paths['folder'].isin(traj_in_split['folder'])][['img_relative_path', 'label', 'idx_start', 'idx_end']]

        # Save to csv
        map_df_path = f"{split}{'Mini' if use_mini_dataset else ''}_mapping_{ascan_per_group}scans.csv"
        print(f"Saving {split} mapping as {map_df_path}...")
        df_map_split.to_csv(dataset_root.joinpath(f"{ascan_per_group}mscans").joinpath(map_df_path), index=False)

    """
    files_split = pd.merge(file_stats.reset_index(), df_split[['label', 'split']].reset_index(), on='folder', how='left').set_index('index').rename(columns={'folder': 'relative_path'})
    files_split['relative_path'] = [p.relative_to(dataset_root) for p in files_split.index]
    files_split = files_split.set_index('relative_path')
    for split in sub_sets:
        # traj_in_split = df_split[df_split['split'] == split]
        traj_in_split = files_split[files_split['split'] == split]
        df_map_split = []
        for traj in traj_in_split.index.tolist():
            traj_cuts = [0] + [(i+1)*ascan_per_group for i in range(traj_in_split.loc[traj, 'col_count']//ascan_per_group)] + [int(traj_in_split.loc[traj, 'col_count'])]
            # Remove last idx if it is repeated
            if traj_cuts[-2] == traj_cuts[-1]:
                traj_cuts = traj_cuts[:-1]
            # TODO: Adjust this part based on clinical data labeling
            df_map_traj = {'relative_path': (len(traj_cuts)-1)*[traj],
                           'label': (len(traj_cuts)-1)*[traj_in_split.loc[traj, 'label']],
                           'idx_start':traj_cuts[0:-1],
                           'idx_end': traj_cuts[1:]}
            df_map_traj = pd.DataFrame.from_dict(df_map_traj, orient='columns')
            df_map_split.append(df_map_traj)
        # Merge all trajectories
        df_map_split = pd.concat(df_map_split, axis=0)
        # Save to csv
        map_df_path = f"{split}{'Mini' if use_mini_dataset else ''}_mapping_{ascan_per_group}scans.csv"
        print(f"Saving {split} mapping as {map_df_path}...")
        df_map_split.to_csv(dataset_root.joinpath(f"{ascan_per_group}mscans").joinpath(map_df_path), index=False)
    """
    print("Done!")







