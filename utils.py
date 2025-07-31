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



if __name__ == '__main__':
    print()