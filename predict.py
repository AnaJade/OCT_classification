import argparse
import pathlib

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
import torch
import torchvision
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from torchvision.io import read_image

# Disable warning for using transforms.v2
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2

import utils
import utils_data
from utils_data import MandibleDataset, NormTransform
from SiameseNet import SiameseNetwork, get_preds

pd.set_option('display.max_columns', None)

if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path',
                        help='Path to the config file',
                        type=str)

    args = parser.parse_args()
    config_file = pathlib.Path(args.config_path)
    # config_file = pathlib.Path("siamese_net/config.yaml")

    if not config_file.exists():
        print(f'Config file not found at {args.config_path}')
        raise SystemExit(1)

    # Load configs
    configs = utils.load_configs(config_file)
    dataset_root = pathlib.Path(configs['data']['dataset_root'])
    anno_paths_test = configs['data']['trajectories_test']
    resize_img_h = configs['data']['resize_img']['img_h']
    resize_img_w = configs['data']['resize_img']['img_w']
    grayscale = configs['data']['grayscale']
    rescale_pos = configs['data']['rescale_pos']

    subnet_name = configs['training']['sub_model']
    cam_inputs = configs['training']['cam_inputs']
    num_hidden = configs['training']['num_fc_hidden_units']
    test_bs = configs['training']['test_bs']
    weights_file_addon = configs['training']['weights_file_addon']
    rename_side = True if 'center_rmBackground' in cam_inputs else False
    real_bgnd = True if any('crop' in c for c in cam_inputs) else False

    if rename_side:
        cam_inputs[-1] = 'Side'
    cam_str = ''.join([c[0].lower() for c in cam_inputs])
    if weights_file_addon:
        weights_file = f"{subnet_name}_{cam_str}cams_{num_hidden}_{weights_file_addon}"
    else:
        weights_file = f"{subnet_name}_{cam_str}cams_{num_hidden}"
    print(f'Loading weights from: {weights_file}')

    if rescale_pos:
        # Set min and max XYZ position values: [[xmin, ymin, zmin], [xmax, ymax, zmax]
        # min_max_pos = [[299, 229, 279], [401, 311, 341]]
        # min_max_pos = [[254, 203, 234], [472, 335, 362]]    # Min and max values for all trajectories
        # min_max_pos = [[290, 235, 275], [410, 305, 345]]  # Min and max values for all trajectories
        min_max_pos = [[284.0, 234.0, 269.0], [406.0, 326.0, 351.0]]
        # min_max_pos = utils_data.get_dataset_min_max_pos(configs)
        print(f'min_max_pos: {min_max_pos}')
    else:
        min_max_pos = None

    # Set training device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Inference done on {device}')

    # Merge all annotation files based on config file
    print("Loading annotation files...")
    annotations_test = utils_data.merge_annotations(dataset_root, anno_paths_test)
    # Filter images
    annotations_test = utils_data.filter_imgs_per_position(annotations_test, [[310, 406], [245, 326], []], None)
    annotations_test = utils_data.filter_imgs_per_rotation_euler(annotations_test, None)

    # Get pred file name
    # Set file name
    if any('occ' in img for img in annotations_test.index.values) and not any(
            'slice' in img for img in annotations_test.index.values):
        pred_file = pathlib.Path(f"siamese_net/preds/{weights_file}_occ.csv")
    elif any('slice' in img for img in annotations_test.index.values) and not any(
            'occ' in img for img in annotations_test.index.values):
        pred_file = pathlib.Path(f"siamese_net/preds/{weights_file}_test_unoccluded.csv")
    elif any('slice' in img for img in annotations_test.index.values):
        pred_file = pathlib.Path(f"siamese_net/preds/{weights_file}_test.csv")
    else:
        pred_file = pathlib.Path(f"siamese_net/preds/{weights_file}.csv")

    if real_bgnd:
        pred_file = pathlib.Path(str(pred_file).replace(pred_file.stem, f'{pred_file.stem}_crop'))

    if pred_file.exists():
        print(f'Loading preds from {pred_file}')
        preds = pd.read_csv(pred_file).set_index('frame')
        preds = preds[preds.columns.intersection(preds.columns[:7])]
        preds = preds.loc[annotations_test.index, :]
    else:
        # Create dataset object
        print("Creating dataloader...")
        # Create dataset objects
        if rename_side:
            cam_inputs[-1] = 'Side'
        if grayscale:
            transforms = v2.Compose([torchvision.transforms.Resize((resize_img_h, resize_img_w)),
                                     torchvision.transforms.Grayscale(),
                                     NormTransform()])  # Remember to also change the annotations for other transforms
        else:
            transforms = v2.Compose([torchvision.transforms.Resize((resize_img_h, resize_img_w)),
                                     NormTransform()])
        dataset_test = MandibleDataset(dataset_root, cam_inputs, annotations_test, min_max_pos, transforms)
        # NOTE: shuffle has to be false, to be able to match the predictions to the right frames
        # dataloader_test = DataLoader(dataset_test, batch_size=test_bs, shuffle=False, num_workers=4)
        dataloader_test = DataLoader(dataset_test, batch_size=test_bs, shuffle=False, num_workers=0)

        # Define the model
        print("Loading model...")
        model = SiameseNetwork(configs)
        # Load trained weights
        model.load_state_dict(torch.load(f"siamese_net/model_weights/{weights_file}.pth"))
        model.to(device)

        print("Performing inference...")
        # Get predictions
        preds = get_preds(model, device, dataloader_test, min_max_pos)

    # Calculate the loss
    test_rmse = mean_squared_error(annotations_test.to_numpy(), preds.to_numpy(), squared=False)
    print(f'Test RMSE: {test_rmse}')

    # Calculate the loss per dimension
    # annotations_test_euler = utils.pose_quaternion2euler(annotations_test.to_numpy())
    # preds_euler = utils.pose_quaternion2euler(preds.to_numpy())
    rmse_per_dim = utils_data.get_loss_per_axis(annotations_test.to_numpy(), preds.to_numpy())
    print(f'RMSE per dimension: \n{rmse_per_dim}')

    mae_per_dim = utils_data.get_mae_per_axis(annotations_test.to_numpy(), preds.to_numpy())
    print(f'MAE per dimension: \n{mae_per_dim}')

    # Calculate the loss on the normalized data
    """
    norm_annotations = utils_data.normalize_position(torch.Tensor(annotations_test.to_numpy()),
                                                     np.array(min_max_pos[0]), np.array(min_max_pos[1]))
    norm_preds = utils_data.normalize_position(torch.Tensor(preds.to_numpy()),
                                               np.array(min_max_pos[0]), np.array(min_max_pos[1]))
    test_rmse = mean_squared_error(norm_annotations, norm_preds, squared=False)
    print(f'Test RMSE on normalized data: {test_rmse}')
    """

    # Calculate the rotation error
    rot_q_diff = utils.hamilton_prod(annotations_test.to_numpy()[:, -4:], preds.to_numpy()[:, -4:])

    # Convert error quaternion to Euler
    rot_euler_error = utils.quaternion2euler(rot_q_diff)
    rot_euler_error_rmse = np.sqrt(np.square(rot_euler_error))  # 'RMSE' on the euler eurer
    rot_euler_error_rmse = np.vstack([np.mean(rot_euler_error_rmse, axis=0), np.min(rot_euler_error_rmse, axis=0),
                                      np.max(rot_euler_error_rmse, axis=0), np.median(rot_euler_error_rmse, axis=0),
                                      (np.max(rot_euler_error_rmse, axis=0) - np.min(rot_euler_error_rmse, axis=0)) / 2,
                                      np.std(rot_euler_error_rmse, axis=0)])
    rot_euler_error_rmse = pd.DataFrame(rot_euler_error_rmse.T, index=['Rx_err', 'Ry_err', 'Rz_err'],
                                        columns=['Rot_err', 'Rot_err_min', 'Rot_err_max', 'Rot_err_median',
                                                 'Rot_err_range', 'Rot_err_std'])
    rot_mean = rot_euler_error_rmse.mean()
    rot_euler_avg_err = pd.concat([rot_euler_error_rmse, rot_mean.to_frame(name='rot_mean').T])
    print(f'Average orientation error:\n{rot_euler_avg_err}')

    # Rot error MAE
    rot_euler_error_mae = np.abs(rot_euler_error)
    rot_euler_error_mae = np.vstack([np.mean(rot_euler_error_mae, axis=0), np.min(rot_euler_error_mae, axis=0),
                                     np.max(rot_euler_error_mae, axis=0), np.median(rot_euler_error_mae, axis=0),
                                     (np.max(rot_euler_error_mae, axis=0) - np.min(rot_euler_error_mae, axis=0)) / 2,
                                     np.std(rot_euler_error_mae, axis=0)])
    rot_euler_error_mae = pd.DataFrame(rot_euler_error_mae.T, index=['Rx_err', 'Ry_err', 'Rz_err'],
                                       columns=['Rot_MAE', 'Rot_MAE_min', 'Rot_MAE_max', 'Rot_MAE_median',
                                                'Rot_MAE_range', 'Rot_MAE_std'])
    rot_mean = rot_euler_error_mae.mean()
    rot_euler_error_mae = pd.concat([rot_euler_error_mae, rot_mean.to_frame(name='rot_mean').T])
    print(f'Average orientation MAE:\n{rot_euler_error_mae}')

    if not pred_file.exists():
        # Format to pandas df
        preds_df = annotations_test.copy()
        preds_df.iloc[:, :] = preds

        # Append position difference
        pos_diff = annotations_test.to_numpy()[:, :3] - preds.to_numpy()[:, :3]
        pos_diff = pd.DataFrame(pos_diff, columns=['delta_x', 'delta_y', 'delta_z'], index=preds_df.index)
        preds_df = pd.concat([preds_df, pos_diff], axis=1)

        # Append the position, orientation and total RMSE for each image
        rmse_per_image = utils_data.get_loss_per_img(annotations_test.to_numpy(), preds.to_numpy())
        rmse_per_image.index = preds_df.index
        preds_df = pd.concat([preds_df, rmse_per_image], axis=1)

        # Add euler error for each image
        rot_err_per_image = pd.DataFrame(rot_euler_error, columns=['Rx_err', 'Ry_err', 'Rz_err'])
        rot_err_per_image.index = preds_df.index
        preds_df = pd.concat([preds_df, rot_err_per_image], axis=1)

        # Save preds as csv
        print(f"Saving results in {pred_file}...")
        preds_df.to_csv(pred_file)
        # print(preds_df.head(5))

        # Reset pred df to kepp only predictions
        preds = preds_df[preds_df.columns.intersection(annotations_test.columns[:7])]

    # Get Pearson product-moment correlation coefficients
    pcc_per_axis = utils_data.get_pcc_per_axis(annotations_test, preds)
    print(f'Pearson correlation coefficient per axis:\n{pcc_per_axis}')

    # Get the graph plotting true vs predicted values
    # Convert from quaternion to euler
    annotations_test_euler = utils_data.get_euler_annotations([annotations_test])[0]
    preds_euler = utils_data.get_euler_annotations([preds])[0]
    for traj_file in anno_paths_test:
        traj = pathlib.Path(traj_file).stem
        anno_traj = annotations_test_euler.filter(like=traj, axis=0)
        preds_traj = preds_euler.filter(like=traj, axis=0)

        # Split position and orientation + shift values to relative pos/rot
        anno_pos = anno_traj[['x', 'y', 'z']].to_numpy() - np.min(anno_traj[['x', 'y', 'z']].to_numpy(), axis=0)
        preds_pos = preds_traj[['x', 'y', 'z']].to_numpy() - np.min(anno_traj[['x', 'y', 'z']].to_numpy(), axis=0)
        annos_ori = anno_traj[['Rx', 'Ry', 'Rz']].to_numpy() - np.array([-90, 90, 0])
        preds_ori = preds_traj[['Rx', 'Ry', 'Rz']].to_numpy() - np.array([-90, 90, 0])

        # Plot
        frame_id = np.arange(0, len(anno_traj))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
        line_colours = ['r', 'g', 'b']
        colour_cycler = cycler(color=line_colours)
        ax1.set_prop_cycle(colour_cycler)
        ax2.set_prop_cycle(colour_cycler)
        ax1.plot(frame_id, anno_pos, label=['gt_x', 'gt_y', 'gt_z'], linestyle='-')
        ax1.plot(frame_id, preds_pos, label=['pred_x', 'pred_y', 'pred_z'], linestyle='--')
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax1.set_ylabel('Displacement [mm]')
        ax2.plot(frame_id, annos_ori, label=['gt_Rx', 'gt_Ry', 'gt_Rz'], linestyle='-')
        ax2.plot(frame_id, preds_ori, label=['pred_Rx', 'pred_Ry', 'pred_Rz'], linestyle='--')
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax2.set_xlabel('Frame number')
        ax2.set_ylabel('Rotation [°]')
        fig.suptitle(f'True vs predicted relative displacement and rotation for {traj}')
        plt.tight_layout()
        plt.subplots_adjust(left=0.075, bottom=0.05, right=0.875, top=0.95, wspace=0.15, hspace=0.15)
        plt.show()
