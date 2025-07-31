import pathlib
import yaml
import numpy as np
import pandas as pd

from scipy.spatial.transform import Rotation


def load_configs(config_path: pathlib.Path) -> dict:
    """
    Load the configs in the yaml file, and returns them in a dictionary
    :param config_path: path to the config file as a pathlib path
    :return: dict with all configs in the file
    """
    # Read yaml config
    with open(config_path, 'r') as file:
        configs = yaml.safe_load(file)

    return configs


def load_frame_time_steps(csv_path: pathlib.Path) -> pd.DataFrame:
    """
    Load and format the image time frame csv file
    :param csv_path: path to the time_steps csv file
    :return: dataframe with the file data
    """
    csv = pd.read_csv(csv_path, header=None, names=['time'])
    return csv


def load_log_file(txt_path: pathlib.Path) -> pd.DataFrame:
    """
    Load and format the robot position log txt file
    :param txt_path: path to the robot pose log file
    :return: dataframe with the file data
    """
    col_names = ['time', 'x', 'y', 'z', 'q1', 'q2', 'q3', 'q4']
    txt = pd.read_table(txt_path, sep=',', header=None, names=col_names)
    return txt


def quaternion2euler(q: np.array) -> np.array:
    """
    Convert a 1D or 2D quaternion np array to the equivalent  [Z, Y, X] euler angles
    :param q: quaternions, as either a 1D or 2D array
    :return: euler angle equivalent for the given quaternions
    """
    reshape_output = False
    if np.isnan(q).any():
        rot_euler = np.empty(3)
        rot_euler[:] = np.nan
    else:
        # Scipy uses real last, so change the order
        if len(q.shape) == 1:
            reshape_output = True
            q = np.expand_dims(q, 0)
        q = q[:, [1, 2, 3, 0]]
        rot = Rotation.from_quat(q)
        rot_euler = rot.as_euler('zyx', degrees=True)
        # Reformat to get [Rx, Ry, Rz] based on the robot
        rot_euler = rot_euler[:, [2, 0, 1]]
        if reshape_output:
            rot_euler = np.squeeze(rot_euler, axis=0)
    return rot_euler


def pose_quaternion2euler(pose_q: np.ndarray) -> np.ndarray:
    q = pose_q[:, -4:]
    r_euler = quaternion2euler(q)
    pose_euler = np.concatenate([pose_q[:, :3], r_euler], axis=1)

    return pose_euler


def q_norm(q: np.ndarray) -> np.ndarray:
    """
    Normalize quaternion values to get unit quaternions (versors)
    :param q: [_, 4] np array with the quaternion values
    :return: normalized quaternions
    """
    n = np.sqrt((q*q).sum(axis=1))
    q = q/n[:, None]
    return q


def hamilton_prod(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Calculate the orientation difference (Hamilton product) between each quaternion in q1 and q2
    For unit quaternions, the reciprocal is the same as the conjugate.
    Based on https://math.stackexchange.com/a/3573308
    :param q1: [_, 4] unit quaternions (true values)
    :param q2: [_, 4] unit quaternions (predicted values)
    :return: Hamilton product of both quaternions
    """
    # [0] Delta q = q1q2 + x1x2 + y1y2 + z1z2
    dq = q1[:, 0]*q2[:, 0] + q1[:, 1]*q2[:, 1] + q1[:, 2]*q2[:, 2] + q1[:, 3]*q2[:, 3]

    # [1] Delta x = q1x2 − x1q2 + y1z2 − z1y2
    dx = q1[:, 0]*q2[:, 1] - q1[:, 1]*q2[:, 0] + q1[:, 2]*q2[:, 3] - q1[:, 3]*q2[:, 2]

    # [2] Delta y = q1y2 − y1q2 − x1z2 + z1x2
    dy = q1[:, 0]*q2[:, 2] - q1[:, 2]*q2[:, 0] - q1[:, 1]*q2[:, 3] + q1[:, 3]*q2[:, 1]

    # [3] Delta z = q1z2 − z1q2 + x1y2 − y1x2
    dz = q1[:, 0]*q2[:, 3] - q1[:, 3]*q2[:, 0] + q1[:, 1]*q2[:, 2] - q1[:, 2]*q2[:, 1]

    q = np.vstack([dq, dx, dy, dz]).T

    return q


if __name__ == '__main__':
    # Load data config file
    data_config_file = pathlib.Path("data_prep/data_config.yaml")
    data_configs = load_configs(data_config_file)

    data_folder_path = pathlib.Path(data_configs['annotations']['data_folder_path'])
    traj_name = data_configs['annotations']['traj_name']
    traj_folder_path = data_folder_path.joinpath(traj_name)
    log_file = traj_folder_path.joinpath(data_configs['annotations']['log_name'])
    frame_file = traj_folder_path.joinpath("time_stamps.csv")

    # Load files
    traj = load_log_file(log_file)
    frame_timings = load_frame_time_steps(frame_file)

    # Load config file
    config_file = pathlib.Path("siamese_net/config_windows.yaml")
    configs = load_configs(config_file)
    dataset_root = pathlib.Path(configs['data']['dataset_root'])
    anno_paths_test = configs['data']['trajectories_test']

    # Create full paths
    full_paths = [dataset_root.joinpath(anno_file) for anno_file in anno_paths_test]
    # Read annotation files
    anno_per_file = [pd.read_csv(anno_path, index_col='frame') for anno_path in full_paths]
    # Merge files
    annotations_test = pd.concat(anno_per_file, axis=0)

    # Test quaternion error
    test_q = annotations_test[[f'q{i}' for i in range(1, 5)]].to_numpy()
    test_q2 = q_norm(test_q)    # Slight difference to og values

    q_error = hamilton_prod(test_q, test_q2)

    # Convert error quaternion to Euler
    euler_error = quaternion2euler(q_error)

    print(euler_error)



    print()
