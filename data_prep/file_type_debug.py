import pathlib
import time
import numpy as np
import pandas as pd
from utils_data import open_mat_file

if __name__ == '__main__':

    # Dataset root
    ds_root_path = pathlib.Path(r"C:\Users\anaja\OneDrive\Documents\Ecole\TUHH\Semester 6\Masterarbeit\OCT_sample_data")

    # Find all *_raw.mat files
    raw_mat_files = list(ds_root_path.rglob("*_raw.mat"))
    raw_npy_files = [f.with_suffix(".npy") for f in raw_mat_files]
    raw_csv_files = [f.with_suffix(".csv") for f in raw_mat_files]

    # Convert file format
    print("Converting mat files to csv...")
    exec_time_tracker = {}
    for fm, fn, fc in zip(raw_mat_files, raw_npy_files, raw_csv_files):
        exec_time_tracker[f'{fm.name}'] = {}
        # Load mat file
        print(f"Working on {fm.name}")
        time_start = time.time()
        mat_data = open_mat_file(fm)
        time_stop = time.time()
        mat_load_time = time_stop - time_start
        exec_time_tracker[f'{fm.name}']['col_count'] = mat_data.shape[0]

        # Convert mat to npy
        print("\tSaving to .npy...")
        time_start = time.time()
        np.save(fn, mat_data)
        time_stop = time.time()
        exec_time_tracker[f'{fm.name}']['npy_save_time'] = time_stop - time_start

        # Convert mat to csv
        print("\tSaving to .csv...")
        time_start = time.time()
        pd.DataFrame(mat_data).to_csv(fc, index=False, header=False)
        time_stop = time.time()
        exec_time_tracker[f'{fm.name}']['csv_save_time'] = time_stop - time_start

        # Check load speed from npy
        exec_time_tracker[f'{fm.name}']['mat_load_time'] = mat_load_time
        print("\tLoading from .npy...")
        time_start = time.time()
        mat_data = np.load(fn)
        time_stop = time.time()
        exec_time_tracker[f'{fm.name}']['npy_load_time'] = time_stop - time_start

        # Check load speed from csv
        print("\tLoading from .csv...")
        time_start = time.time()
        mat_data = pd.read_csv(fc)
        time_stop = time.time()
        exec_time_tracker[f'{fm.name}']['csv_load_time'] = time_stop - time_start

        # Add file size
        exec_time_tracker[f'{fm.name}']['mat_fs'] = fm.stat().st_size / (1024 ** 3)  # Convert bytes to GB
        exec_time_tracker[f'{fm.name}']['npy_fs'] = fn.stat().st_size / (1024 ** 3)  # Convert bytes to GB
        exec_time_tracker[f'{fm.name}']['csv_fs'] = fc.stat().st_size / (1024 ** 3)  # Convert bytes to GB


    exec_tt = pd.DataFrame.from_dict(exec_time_tracker, orient='index')
    exec_tt.to_excel(ds_root_path.joinpath("file_type_overview.xlsx"))
    print(exec_tt)







