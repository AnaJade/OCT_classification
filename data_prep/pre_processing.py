# Goal: Reproduce imagesc(movmean(mscan_real, 100, 2)) from Matlab
import argparse
import pathlib
from sys import platform
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy.io import loadmat
from PIL import Image

import utils
from utils_data import open_mat_file, movmean

if __name__ == '__main__':
    # Set up the argument parser
    """
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

    # Find all *_raw.mat files
    mat_file_root = pathlib.Path(r'D:\Masterarbeit\OCT_lab_data')
    raw_mat_files = list(mat_file_root.rglob("*_raw.mat"))
    raw_mat_files = [f for f in raw_mat_files if 'reject' not in f.__str__()]  # Remove files if they are in the rejected folder

    # Load one mat file
    # f = raw_mat_files[3]
    """

    f = pathlib.Path(r'D:\Masterarbeit\OCT_lab_data\chicken_heart_p1_area1\chicken_heart_p1_area1_sine_5x5y_s8_zamp05_1747041952022251_raw.mat')
    print(f)
    mat_data = open_mat_file(f).T
    # mat_data = cv2.normalize(mat_data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    mat_data = mat_data[:, :1000]

    # Apply movmean filter
    mat_data_filtered = movmean(mat_data, 100)
    mat_data_filtered_norm = cv2.normalize(mat_data_filtered, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Show before and after
    fig, (ax_og, ax_fil) = plt.subplots(2, 1, figsize=(6, 15))
    ax_og.imshow(mat_data, cmap='gray')
    ax_og.set_title('Original')
    ax_og.set_axis_off()
    ax_fil.imshow(np.absolute(mat_data_filtered), cmap='gray')
    ax_fil.set_title('movmean 100')
    ax_fil.set_axis_off()
    fig.show()
    plt.tight_layout()
    plt.show()


