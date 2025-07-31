import argparse
import pathlib
from tqdm import tqdm
import numpy as np
import pandas as pd

import utils


def get_pose_for_time(time, df) -> np.array:
    """
    Get the pose of the robot at desired frame time stamp
    :param time: Frame time stamp
    :param df: log of the robot positions for all time steps
    :return: pose as a np array
    """
    pose = df[df['time'] == time]
    if pose.empty:
        try:
            lower_pose = df.iloc[df[df['time'] < time]['time'].idxmax()]
        except ValueError:
            pose = np.empty(8)
            pose[:] = np.nan
            return pose
        try:
            upper_pose = df.iloc[df[df['time'] > time]['time'].idxmin()]
        except ValueError:
            pose = np.empty(8)
            pose[:] = np.nan
            return pose

        # Linear interpolation for the position
        lower_position = lower_pose[1:4].to_numpy()
        upper_position = upper_pose[1:4].to_numpy()
        position = upper_position - ((upper_pose['time'] - time)*(upper_position - lower_position)/(upper_pose['time'] - lower_pose['time']))

        # Get the quaternion
        # TODO: Redo using slerp
        if abs(lower_pose['time'] - time) < abs(upper_pose['time'] - time):
            orientation = lower_pose[-4:].to_numpy()
        else:
            orientation = upper_pose[-4:].to_numpy()

        pose = np.concatenate((np.array([time]), position, orientation))
    else:
        pose = pose.to_numpy().reshape(-1,)

    return pose


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
    log_file = traj_folder_path.joinpath(configs['annotations']['log_name'])
    frame_file = traj_folder_path.joinpath("time_stamps.csv")

    # Get robot and camera time stamps
    traj = utils.load_log_file(log_file)
    frame_timings = utils.load_frame_time_steps(frame_file)

    # Get robot pose at each camera time stamp
    col_names = traj.columns
    frame_pose_rows = []
    for i in tqdm(range(len(frame_timings))):
        pose = get_pose_for_time(frame_timings['time'][i], traj)
        row_dict = dict((col, pose[i]) for i, col in enumerate(col_names))
        frame_pose_rows.append(row_dict)
    frame_pose = pd.DataFrame(frame_pose_rows, columns=col_names)

    # Save as csv
    frame_pose_path = traj_folder_path.joinpath("frame_pose_shortcut.csv")  # Remove shortcut when using slerp
    frame_pose.to_csv(frame_pose_path, index=False)

    print(f"Results saved to {frame_pose_path}")
