import argparse
import pathlib
import time
import numpy as np
import pandas as pd
import random
from sys import platform
import utils
from utils_data import open_mat_file

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
    labels = configs['data']['labels']
    use_mini_dataset = configs['data']['use_mini_dataset']

    # Dataset target path (if different then dataset_root)
    target_path = pathlib.Path(r"C:\Users\anaja\OneDrive\Documents\Ecole\TUHH\Semester 6\Masterarbeit\OCT_lab_data")

    # Find all *_raw.mat files
    raw_mat_files = list(dataset_root.rglob("*_raw.mat"))
    raw_mat_files = [f for f in raw_mat_files if 'reject' not in f.__str__()]   # Remove files if they are in the rejected folder
    raw_file_pairs = [(fm, fm.with_suffix(".npy")) for fm in raw_mat_files]

    # Update raw_npy_files if needed
    if target_path != dataset_root:
        raw_file_pairs = [(fm, target_path.joinpath(fn.parts[-2]).joinpath(fn.name)) for (fm, fn) in raw_file_pairs]

    # Filter file names if a mini dataset is desired
    if use_mini_dataset:
        print("Selecting files for the mini dataset...")
        # Keep only s8 files
        raw_file_pairs = [(fm, fn) for (fm, fn) in raw_file_pairs if '_s8_' in fm.__str__()]
        # Sample 4 areas from each label
        raw_file_pairs_per_label = {l: [(fm, fn) for (fm, fn) in raw_file_pairs if l in fm.__str__()] for l in labels}
        areas_to_keep = {lbl:list(set([fm.parts[-2].split('_')[-1] for (fm, fn) in files])) for (lbl, files) in raw_file_pairs_per_label.items()}
        areas_to_keep = {lbl: random.sample(areas, 4) for (lbl, areas) in areas_to_keep.items()}
        raw_file_pairs = [(fm, fn) for (lbl, areas) in areas_to_keep.items() for area in areas for (fm, fn) in raw_file_pairs if area+"_" in fm.__str__() and lbl in fm.__str__()]
        print(f"Keeping {len(raw_file_pairs)} areas over {len(labels)} labels.")

    # Convert file format
    print("Converting mat files to npy...")
    for i, (fm, fn) in enumerate(raw_file_pairs):
        print(f"File {i+1}/{len(raw_file_pairs)}: Working on {fm.name}")
        # Create target folder if needed
        if not fn.parent.is_dir():
            fn.parent.mkdir(parents=True, exist_ok=True)
        # Skip files that already exit
        if fn.exists():
            continue
        # Load mat file
        mat_data = open_mat_file(fm)
        # Convert mat to npy
        print("\tSaving to .npy...")
        np.save(fn, mat_data)
    print("Done!")







