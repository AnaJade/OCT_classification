import pathlib
import time
import numpy as np
import pandas as pd
from utils_data import open_mat_file

if __name__ == '__main__':

    # Dataset root
    ds_root_path = pathlib.Path(r"D:\OCT_lab_data_AJ")

    # Find all folders
    all_folders = list(ds_root_path.iterdir())
    # Find all files in each folder
    all_files = [d.iterdir() for d in all_folders]
    # Find all *_raw.mat files
    raw_mat_files = list(ds_root_path.rglob("*_raw.mat"))
    raw_npy_files = [f.with_suffix(".npy") for f in raw_mat_files]