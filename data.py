import pathlib
import time
import numpy as np
import torch

from utils_data import open_mat_file

if __name__ == '__main__':
    # Check if cuda is available
    # print(torch.cuda.is_available())

    # Load .dat file
    timesteps_file = pathlib.Path(r"C:\Users\anaja\OneDrive\Documents\Ecole\TUHH\Semester 6\Masterarbeit\OCT_sample_data\test_scans\test_1741858028367969_time.dat")
    # timesteps_op = open(timesteps_file, encoding="ieee-le")
    # timesteps = timesteps_op.read()
    # timesteps = np.genfromtxt(timesteps_file, dtype=np.uint64)
    # print(timesteps)

    # Load .mat file
    print("Loading mat file...")
    mat_file = pathlib.Path(r"C:\Users\anaja\OneDrive\Documents\Ecole\TUHH\Semester 6\Masterarbeit\OCT_sample_data\test_scans\test_1741858028367969_raw.mat")
    time_start = time.time()
    mat_data = open_mat_file(mat_file)
    time_stop = time.time()
    print(f"Execution time for opening a .mat file: {time_stop - time_start} ms")

