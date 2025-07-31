import argparse
import pathlib
import time
import numpy as np
import pandas as pd
import random
import cv2
from PIL import Image
from tqdm import tqdm
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
    ascan_per_group = configs['data']['ascan_per_group']
    labels = configs['data']['labels']
    use_mini_dataset = configs['data']['use_mini_dataset']

    # Dataset target path (if different then dataset_root)
    target_path = pathlib.Path(r"C:\Users\anaja\OneDrive\Documents\Ecole\TUHH\Semester 6\Masterarbeit\OCT_lab_data")
    img_root_path = target_path.joinpath(f"{ascan_per_group}mscans")

    # Find all *_raw.mat files
    raw_mat_files = list(dataset_root.rglob("*_raw.mat"))
    raw_mat_files = [f for f in raw_mat_files if 'reject' not in f.__str__()]   # Remove files if they are in the rejected folder

    # Filter file names if a mini dataset is desired
    if use_mini_dataset:
        print("Selecting files for the mini dataset...")
        # Keep only s8 files
        raw_mat_files = [fm for fm in raw_mat_files if '_s8_' in fm.__str__()]
        # Sample 4 areas from each label
        """
        raw_file_per_label = {l: [fm for fm in raw_mat_files if l in fm.__str__()] for l in labels}
        areas_to_keep = {lbl:list(set([fm.parts[-2].split('_')[-1] for fm in files])) for (lbl, files) in raw_file_per_label.items()}
        areas_to_keep = {lbl: random.sample(areas, 4) for (lbl, areas) in areas_to_keep.items()}
        raw_files = [fm for (lbl, areas) in areas_to_keep.items() for area in areas for fm in raw_mat_files if area+"_" in fm.__str__() and lbl in fm.__str__()]
        """
        raw_files = raw_mat_files
        print(f"Keeping {len(raw_files)} areas over {len(labels)} labels.")
    else:
        raw_files = raw_mat_files
    # Uncomment if specific files to be converted to images are already saved as np files
    # npy_files = list(target_path.rglob("*_raw.npy"))
    # raw_files = [dataset_root.joinpath(fn.parts[-2]).joinpath(f"{fn.stem}.mat") for fn in npy_files]

    # Convert file format
    print("Saving mat files to individual images...")
    # Create new folder if needed
    if not img_root_path.is_dir():
        img_root_path.mkdir(parents=True, exist_ok=True)

    for i, f in enumerate(raw_files): # enumerate(raw_mat_files):
        print(f"File {i+1}/{len(raw_files)}: Working on {f.name}")
        # Load mat file
        mat_data = open_mat_file(f).T

        # Normalize data to fit as uint8
        # cv2.imshow('Original', img)
        mat_data = cv2.normalize(mat_data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        # cv2.imshow('Normalized', img_norm)
        # cv2.waitKey(0)

        # Create target subfolder if needed
        target_img_subdir = img_root_path.joinpath(f.parts[-2])

        # Create area dir if needed
        if not target_img_subdir.exists():
            target_img_subdir.mkdir(parents=True, exist_ok=True)
        # Find how many images will be generated
        split_idx = traj_cuts = [0] + [(i+1)*ascan_per_group for i in range(mat_data.shape[1]//ascan_per_group)] + [int(mat_data.shape[1])]

        # Save into individual images
        for j in tqdm(range(len(split_idx)-1)):
            img_file_name = target_img_subdir.joinpath(f"{'_'.join(f.stem.split('_')[:-2])}_{split_idx[j]}_{split_idx[j+1]}.jpg")
            img = mat_data[:, split_idx[j]:split_idx[j+1]]
            cv2.imwrite(img_file_name, img)

    print("Done!")







