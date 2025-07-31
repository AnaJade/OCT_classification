import pathlib
import numpy as np
import pandas as pd



if __name__ == '__main__':
    # Find true video fps from time steps
    time_steps_path = pathlib.Path(r"C:\Users\anaja\OneDrive\Documents\Ecole\TUHH\Semester 6\Masterarbeit\OCT_sample_data\test_all\test_all_jitter_20p_2x_2y_2z_s8_framegrabber.txt")
    video_path = pathlib.Path(r"C:\Users\anaja\OneDrive\Documents\Ecole\TUHH\Semester 6\Masterarbeit\OCT_sample_data\test_all\test_all_jitter_20p_2x_2y_2z_s8_framegrabber.mp4")

    # Read time steps file
    with open(time_steps_path, 'r') as f:
        time_steps = [int(line.rstrip()) for line in f]
        f.close()
    # Calculate time diff between time steps
    time_diff = np.diff(time_steps)
    time_diff_mean = time_diff.mean()
    vid_duration = time_steps[-1] - time_steps[0]
    fps = 1/(time_diff_mean/1e10)   # 1e10 taken by matching vid_duration to the actual video duration
    print()

