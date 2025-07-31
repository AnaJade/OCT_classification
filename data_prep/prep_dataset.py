"""
Convert the video files to images, and remove frames that don't have an associated robot position
Create a csv file with the annotations
"""
import argparse
import pathlib

import numpy as np
from tqdm import tqdm

import pandas as pd
import cv2

import utils


def get_vid_paths(vid_folder: pathlib.Path) -> list:
    """
    Build the individual camera video files + annotation paths
    :param vid_folder: path to the trajectory folder
    :return: list with the left, right and side video paths + annotation path
    """
    base_vid_name = vid_folder.stem
    l_vid_path = vid_folder.joinpath(f'{base_vid_name}_l.avi')
    r_vid_path = vid_folder.joinpath(f'{base_vid_name}_r.avi')
    s_vid_path = vid_folder.joinpath(f'{base_vid_name}_s.avi')
    annotation_path = vid_folder.joinpath(f'frame_pose_shortcut.csv')   # Eventually remove shortcut
    return [[l_vid_path, r_vid_path, s_vid_path], annotation_path]


def convert_vid2imgs(vid_path: pathlib.Path, anno_path: pathlib.Path, img_base_dir: pathlib.Path,
                     remove_duplicates=True) -> None:
    """
    Save the video frames as images. Files that already exist will not be overwritten
    :param vid_path: Path to the video frames to be extracted
    :param anno_path: Annotation file path
    :param img_base_dir: Base directory for where to store the images
    :param remove_duplicates: Whether to remove frames where the pose is the same or not
    :return:
    """
    # Read annotation file
    annos = pd.read_csv(anno_path).set_index('time')
    col_names = annos.columns.tolist()

    # Replace duplicate values with NaN
    if remove_duplicates:
        # Find index (time stamp) of unique frames
        unique_idx = annos.drop_duplicates(keep='first').index.to_list()
        # Replace duplicate values with NaN in og df
        duplicate_idx = [idx for idx in annos.index.to_list() if idx not in unique_idx]
        annos.loc[duplicate_idx, :] = np.nan
        print(f'\nRemoved {len(duplicate_idx)} duplicate frames from {vid_path.stem}.')

    annos = annos.reset_index().drop('time', axis=1)

    # Check camera used
    cam = vid_path.stem[-1]
    cams = {'l': 'Left', 'r': 'Right', 's': 'Side'}

    # Load video file
    print(f"Loading {cams[cam]} video file...")
    vid = cv2.VideoCapture(vid_path.__str__())
    readable, img = vid.read()

    # Set img params
    img_dir = img_base_dir.joinpath(f'{cams[cam]}')
    traj_name = '_'.join(vid_path.stem.split('_')[:-1])

    # Set new annotation file params
    new_anno_path = img_base_dir.joinpath(f'{traj_name}.csv')
    new_anno_exists = new_anno_path.is_file()

    # Loop through all frames, and save them as images
    img_id = 0
    anno_new = []
    print(f"Saving {cams[cam]} camera frames as images...")
    for frame_id in tqdm(range(len(annos))):
        if not readable:
            break
        pose = annos.iloc[frame_id]
        # Check if pose is none
        if not pose.isnull().sum().any():
            # Save as image
            img_file = img_dir.joinpath(f'{traj_name}_{img_id}_{cam}.jpg')
            if not img_file.is_file():
                cv2.imwrite(img_file.__str__(), img)

            # Save corresponding annotation
            if not new_anno_exists:
                anno_img = dict((col, pose.iloc[i]) for i, col in enumerate(col_names))
                anno_img['frame'] = f'{traj_name}_{img_id}'
                anno_new.append(anno_img)

            # Update img id
            img_id = img_id + 1

        # Go to next frame
        readable, img = vid.read()

    if not new_anno_exists:
        # Create new annotation dataframe
        col_names.append('frame')
        anno_new = pd.DataFrame(anno_new, columns=col_names).set_index('frame')

        # Save new annotations
        anno_new.to_csv(img_base_dir.joinpath(f'{traj_name}.csv'), index=True)

    print(f"Finished {cams[cam]} camera")


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

    img_root_path = pathlib.Path(configs['images']['img_root'])

    [vid_paths, anno_path] = get_vid_paths(traj_folder_path)

    # Save imgs and annotations
    for vid_path in vid_paths:
        convert_vid2imgs(vid_path, anno_path, img_root_path)
