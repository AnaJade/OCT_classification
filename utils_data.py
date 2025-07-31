import copy
import pathlib
import warnings
import platform
from random import randint
import h5py

import cv2
import matplotlib
from tqdm import tqdm
from typing import Union

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from scipy.io import loadmat
from PIL import Image
from sklearn.metrics import mean_squared_error

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
# Disable warning for using transforms.v2
torchvision.disable_beta_transforms_warning()
from torchvision.io import read_image
from torchvision import transforms
from torchvision.transforms import v2

import utils
from SPICE.fixmatch.datasets.augmentation.randaugment import RandAugment
from SPICE.fixmatch.datasets.data_utils import get_onehot


class OCTDataset(Dataset): # Used in train_moco
    def __init__(self, root: pathlib.Path, split: str, map_df_paths: dict, labels_dict: dict, transforms=None, preload_data=False):
        """
        Dataset object used to pass images to a siamese network
        :param root: dataset root path
        :param split: dataset split ('train', 'valid', 'test')
        :param map_df_paths: Path to the mapping dataframes of start and stop idx of M-Scans (train, valid, test)
        :param labels_dict: int to string label conversion
        :param transforms: Set of transforms that will be applied to each image before being used as input by the model
        :param preload_data: Whether to preload and save all the data into class variables
        """
        self.root = root
        # self.map_df_paths = map_df_paths
        self.transforms = transforms
        self.map_df = map_df_paths[split]
        self.label_dict = labels_dict
        self.map_df = pd.read_csv(self.map_df)
        self.preload_data = preload_data
        self.data = None
        self.labels = None

        # Update relative path to image paths
        ascan_per_group = self.map_df['idx_end'].iloc[0]
        img_subdir = pathlib.Path(f"{ascan_per_group}mscans")
        self.map_df.loc[:, 'img_relative_path'] = [pathlib.Path(p) for p in self.map_df.loc[:,'img_relative_path']]

        # Restructure mapping df
        self.map_df = self.map_df.rename(columns={'label': 'label_str'})
        for i, lbl in self.label_dict.items():
            self.map_df.loc[self.map_df['label_str'] == lbl, 'label'] = i
        missing_labels = [l for l in labels_dict.values() if l not in self.map_df['label_str'].unique()]
        extra_labels = self.map_df[self.map_df['label'].isna()]['label_str'].unique().tolist()
        if len(missing_labels) > 0:
            print(f"{missing_labels} not found in mapping dataset")
        if len(extra_labels) > 0:
            print(f"{extra_labels} found in mapping dataset but not in label dict")
            print(f"Removing labels...")
            self.map_df = self.map_df.dropna(axis=0)

        if self.preload_data:
            self.data = [cv2.imread(self.root.joinpath(self.map_df['img_relative_path'].iloc[i])) for i in range(len(self.map_df))]
            self.labels = [self.map_df['label'].iloc[i] for i in range(len(self.map_df))]

    def __len__(self):
        return len(self.map_df)

    def __getitem__(self, idx):
        if self.preload_data:
            data = self.data[idx]
            label = self.labels[idx]
        else:
            # scan_info = self.map_df[idx]
            scan_path = self.root.joinpath(self.map_df['img_relative_path'].iloc[idx])
            scan_idx_start = self.map_df['idx_start'].iloc[idx]
            scan_idx_end = self.map_df['idx_end'].iloc[idx]
            label = int(self.map_df['label'].iloc[idx])

            # Read M-Scan
            # data = np.load(scan_path)
            # data = np.expand_dims(data[scan_idx_start:scan_idx_end, :].T, 0)
            # data = np.expand_dims(data[0:10000, :].T, 0)
            # data.shape = (512, 5000, 3)
            data = cv2.imread(scan_path)# , cv2.IMREAD_GRAYSCALE)

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            data = Image.fromarray(data)
            """
            try:
                data = Image.fromarray(data) # , mode='L')
            except AttributeError:
                print(f"Problem with file {scan_path}")
                # TODO: REMOVE AFTER DEBUG
                idx = idx + 1
                scan_path = self.root.joinpath(self.map_df['img_relative_path'].iloc[idx])
                label = int(self.map_df['label'].iloc[idx])
                data = cv2.imread(scan_path)
                data = Image.fromarray(data)
            """

        # Apply transforms
        if self.transforms is not None:
            data = self.transforms(data)

        return data, label


class OCTDataset2Trans(Dataset): # Used in pre_compute_embedding (no embedding), train_self_v2 (with embedding), local_consistency (no embedding)
    def __init__(self, root: pathlib.Path, split: str, map_df_paths: dict, labels_dict: dict, transform1=None, transform2=None, embedding=None, show=False, preload_data=False):
        """
        Dataset object used to pass images to a siamese network
        :param root: dataset root path
        :param split: dataset split ('train', 'valid', 'test')
        :param map_df_paths: Path to the mapping dataframes of start and stop idx of M-Scans (train, valid, test)
        :param transform1: First set of transforms that will be applied to each image before being used as input by the model
        :param transform2: Second set of transforms that will be applied to each image before being used as input by the model
        :param embedding: npy embedding file
        :param show: Whether to show the resulting transformed images or not
        """
        self.root = root
        # self.map_df_paths = map_df_paths
        self.transform1 = transform1
        self.transform2 = transform2
        self.embedding = embedding
        self.show = show
        self.map_df = map_df_paths[split]
        self.label_dict = labels_dict
        self.map_df = pd.read_csv(self.map_df)
        self.preload_data = preload_data
        self.data = None
        self.labels = None

        # Update relative path to image paths
        ascan_per_group = self.map_df['idx_end'].iloc[0]
        img_subdir = pathlib.Path(f"{ascan_per_group}mscans")
        self.map_df.loc[:, 'img_relative_path'] = [pathlib.Path(p) for p in self.map_df.loc[:, 'img_relative_path']]

        # Restructure mapping df
        self.map_df = self.map_df.rename(columns={'label': 'label_str'})
        for i, lbl in self.label_dict.items():
            self.map_df.loc[self.map_df['label_str'] == lbl, 'label'] = i
        missing_labels = [l for l in labels_dict.values() if l not in self.map_df['label_str'].unique()]
        extra_labels = self.map_df[self.map_df['label'].isna()]['label_str'].unique().tolist()
        if len(missing_labels) > 0:
            print(f"{missing_labels} not found in mapping dataset")
        if len(extra_labels) > 0:
            print(f"{extra_labels} found in mapping dataset but not in label dict")
            print(f"Removing labels...")
            self.map_df = self.map_df.dropna(axis=0)

        if self.preload_data:
            self.data = [cv2.imread(self.root.joinpath(self.map_df['img_relative_path'].iloc[i])) for i in range(len(self.map_df))]
            self.labels = [self.map_df['label'].iloc[i] for i in range(len(self.map_df))]

        if embedding is not None:
            self.embedding = np.load(embedding)
        else:
            self.embedding = None


    def __len__(self):
        return len(self.map_df)

    def __getitem__(self, idx):
        if self.preload_data:
            data = self.data[idx]
            label = self.labels[idx]
        else:
            # scan_info = self.map_df[idx]
            scan_path = self.root.joinpath(self.map_df['img_relative_path'].iloc[idx])
            scan_idx_start = self.map_df['idx_start'].iloc[idx]
            scan_idx_end = self.map_df['idx_end'].iloc[idx]
            label = int(self.map_df['label'].iloc[idx])

            # Read M-Scan
            # data = np.load(scan_path)
            # data = np.expand_dims(data[scan_idx_start:scan_idx_end, :].T, 0)
            # data = np.expand_dims(data[0:10000, :].T, 0)
            # data.shape = (512, 5000, 3)
            data = cv2.imread(scan_path)# , cv2.IMREAD_GRAYSCALE)

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            data = Image.fromarray(data)
            """
            try:
                data = Image.fromarray(data) # , mode='L')
            except AttributeError:
                print(f"Problem with file {scan_path}")
                # TODO: REMOVE AFTER DEBUG
                idx = idx + 1
                scan_path = self.root.joinpath(self.map_df['img_relative_path'].iloc[idx])
                label = int(self.map_df['label'].iloc[idx])
                data = cv2.imread(scan_path)
                data = Image.fromarray(data)
            """

        if self.embedding is not None:
            emb = self.embedding[idx]
        else:
            emb = None

        # Apply transforms
        if self.transform1 is not None:
            img_trans1 = self.transform1(data)
        else:
            img_trans1 = data

        if self.transform2 is not None:
            img_trans2 = self.transform2(data)
        else:
            img_trans2 = data

        if self.show:
            # TODO: Double check dimensions
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])

            # img_trans1.numpy().shape: (3, 512, 5000)
            # img_trans1.numpy().transpose([1, 2, 0]).shape: (512, 5000, 3)
            img_trans1 = img_trans1.numpy().transpose([1, 2, 0]) * std + mean
            # img_trans1 = img_trans1.numpy().transpose([1, 2, 0])
            # img_trans1 = (img_trans1 - img_trans1.min()) / (img_trans1.max() - img_trans1.min())
            plt.figure()
            plt.imshow(img_trans1)

            img_trans2 = img_trans2.numpy().transpose([1, 2, 0]) * std + mean
            plt.figure()
            plt.imshow(img_trans2)
            plt.show()

        if emb is not None:
            return img_trans1, img_trans2, emb, label, idx
        else:
            return img_trans1, img_trans2, label, idx


class OCTDatasetSSL(Dataset): # train_semi (ssl)
    def __init__(self, root: pathlib.Path, split: str, map_df_paths: dict, labels_dict: dict,
                 reliable_label_idxs: Union[np.array, None], return_just_reliable: bool,
                 use_strong_transform: bool, strong_transforms=None,
                 one_hot=False):
        """
        Dataset object used to pass images to a siamese network
        :param root: dataset root path
        :param split: dataset split ('train', 'valid', 'test')
        :param map_df_paths: Path to the mapping dataframes of start and stop idx of M-Scans (train, valid, test)
        :param labels_dict: int to string label conversion
        :param reliable_label_idxs: Indexes of images with reliable pseudo-labels
        :param return_just_reliable: Whether to use just images with just reliable pseudo-labels or all images
        :param use_strong_transform: Whether to also apply the strong transforms
        :param strong_transforms: List of transformations included in the strong transforms
        :param one_hot: Whether to apply a one-hot transformation to the target labels
        """
        self.root = root
        self.split = split
        self.map_df_paths = map_df_paths
        self.map_df = pd.read_csv(map_df_paths[split])
        self.label_dict = labels_dict
        self.num_classes = len(self.label_dict.keys())
        self.reliable_label_idxs = reliable_label_idxs
        self.return_just_reliable = return_just_reliable
        self.use_strong_transform = use_strong_transform
        self.strong_transform = strong_transforms
        self.one_hot = one_hot
        self.transforms = None

        # Update relative path to image paths
        ascan_per_group = self.map_df['idx_end'].iloc[0]
        img_subdir = pathlib.Path(f"{ascan_per_group}mscans")
        self.map_df.loc[:, 'img_relative_path'] = [pathlib.Path(p) for p in self.map_df.loc[:, 'img_relative_path']]
        """
        self.map_df.loc[:, 'relative_path'] = [pathlib.Path(p) for p in self.map_df.loc[:,'relative_path']]
        self.map_df['img_relative_path'] = [img_subdir.joinpath(p.parts[0]).joinpath(
            f"{'_'.join(p.stem.split('_')[:-2])}_{self.map_df['idx_start'].iloc[i]}_{self.map_df['idx_end'].iloc[i]}.jpg")
                                            for i, p in enumerate(self.map_df['relative_path'])]
        """

        # Restructure mapping df
        self.map_df = self.map_df.rename(columns={'label': 'label_str'})
        for i, lbl in self.label_dict.items():
            self.map_df.loc[self.map_df['label_str'] == lbl, 'label'] = i
        missing_labels = [l for l in labels_dict.values() if l not in self.map_df['label_str'].unique()]
        extra_labels = self.map_df[self.map_df['label'].isna()]['label_str'].unique().tolist()
        if len(missing_labels) > 0:
            print(f"{missing_labels} not found in mapping dataset")
        if len(extra_labels) > 0:
            print(f"{extra_labels} found in mapping dataset but not in label dict")
            print(f"Removing labels...")
            self.map_df = self.map_df.dropna(axis=0)

        # Setup transforms
        # mean, std = self.get_mean_std()
        mean = np.array([150.17267018, 150.17267018, 150.17267018])
        std = np.array([11.74984744, 11.74984744, 11.74984744])
        crop_size = (512, ascan_per_group)
        if self.split == 'train':
            self.transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.RandomCrop(crop_size, padding=4),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean, std)])
        else:
            self.transforms = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean, std)])
        # Setup strong transform
        if self.use_strong_transform and self.strong_transform is None:
            self.strong_transform = copy.deepcopy(self.transforms)
            self.strong_transform.transforms.insert(0, RandAugment(3, 5))

        # Image selection
        if self.return_just_reliable and self.reliable_label_idxs is not None:
            self.map_df = self.map_df.iloc[self.reliable_label_idxs, :].reset_index(drop=True)

    def __len__(self):
        return len(self.map_df)

    def __getitem__(self, idx):
        # scan_info = self.map_df[idx]
        scan_path = self.root.joinpath(self.map_df['img_relative_path'].iloc[idx])
        scan_idx_start = self.map_df['idx_start'].iloc[idx]
        scan_idx_end = self.map_df['idx_end'].iloc[idx]
        label = int(self.map_df['label'].iloc[idx])
        label = label if not self.one_hot else get_onehot(len(self.labels_dict.keys()), label)


        # Read M-Scan
        # data = np.load(scan_path)
        # data = np.expand_dims(data[scan_idx_start:scan_idx_end, :].T, 0)
        # data = np.expand_dims(data[0:10000, :].T, 0)
        # data.shape = (512, 5000, 3)
        data = cv2.imread(scan_path)# , cv2.IMREAD_GRAYSCALE)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        data = Image.fromarray(data)
        """
        try:
            data = Image.fromarray(data) # , mode='L')
        except AttributeError:
            print(f"Problem with file {scan_path}")
            # TODO: REMOVE AFTER DEBUG
            idx = idx + 1
            scan_path = self.root.joinpath(self.map_df['img_relative_path'].iloc[idx])
            label = int(self.map_df['label'].iloc[idx])
            data = cv2.imread(scan_path)
            data = Image.fromarray(data)
        """

        # Apply transforms
        if self.transforms is not None:
            data = self.transforms(data)

        return data, label

    def get_mean_std(self):
        mean = []
        std = []
        for i in range(len(self.map_df)):
            # data.shape = (512, 5000, 3)
            data = cv2.imread(self.root.joinpath(self.map_df['img_relative_path'].iloc[i]))
            mean.append(np.mean(data, (0, 1)))
            std.append(np.std(data, (0, 1)))

        mean = np.mean(np.stack(mean, axis=0), 0)
        std = np.mean(np.stack(std, axis=0), 0)
        return mean, std


def open_mat_file(file: pathlib.Path):
    with open(file, 'rb') as f:
        header = f.read(128).decode('utf-8')

    if "MATLAB 7.3" in header:
        with h5py.File(file, 'r') as mat_file:
            data = mat_file["mscan_real"][:]
    else:
        mat_contents = loadmat(file)
        data = mat_contents["mscan_real"]
    return data


def find_keywords(df: pd.DataFrame, col: str, keywords: list) -> pd.DataFrame:
    """
    Check whether each row in column col contains one of the given keywords.
    :param df: dataframe to be evaluated
    :param col: column in which to search for the keywords
    :param keywords: list of keywords to be found
    :return: Returns the df with extra columns: total keywords found + one per keyword
    """
    # Init new cols
    df.loc[:, 'keyword_count'] = 0
    df.loc[:, [w for w in keywords]] = 0

    # Check if keyword is present
    for word in keywords:
        df.loc[df[col].str.contains(word, case=False), word] = 1

    # Count total number of keywords found
    df['keyword_count'] = np.sum(df[[w for w in keywords]], axis=1)

    return df


class NormTransform(torch.nn.Module):
    """
    Convert tensor type from uint8 to float32, and divide by 255
    """
    def forward(self, img):
        return img.astype(np.float64) / 255
        # return img.float()/255


if __name__ == '__main__':
    config_file = pathlib.Path("config_windows.yaml")
    configs = utils.load_configs(config_file)
    dataset_root = pathlib.Path(configs['data']['dataset_root'])
    labels = configs['data']['labels']
    labels_dict = {i: lbl for i, lbl in enumerate(labels)}
    ascan_per_group = configs['data']['ascan_per_group']
    use_mini_dataset = configs['data']['use_mini_dataset']
    map_df_paths ={split: dataset_root.joinpath(f"{split}{'Mini' if use_mini_dataset else ''}_mapping_{ascan_per_group}scans.csv") for split in ['train', 'valid', 'test']}



    # Create dataset object
    print("Initializing dataset object...")
    split = 'train'
    # Create dataset objects
    transforms = v2.Compose([NormTransform()])
    dataset_train = OCTDataset(dataset_root, split, map_df_paths, labels_dict, transforms)
    dataset_valid = OCTDataset(dataset_root, split, map_df_paths, labels_dict, transforms)
    dataset_test = OCTDataset(dataset_root, split, map_df_paths, labels_dict, transforms)

    print("Creating dataloader...")
    dataloader_train = DataLoader(dataset_train, batch_size=4, shuffle=True, num_workers=0)
    dataloader_valid = DataLoader(dataset_valid, batch_size=4, shuffle=False, num_workers=0)
    dataloader_test = DataLoader(dataset_test, batch_size=4, shuffle=False, num_workers=0)

    print("Fetching data...")
    for imgs, lbl in dataloader_train:
        # print(type(imgs))
        # print(type(lbl))
        img = (imgs[0, :, :, :].numpy().transpose((1, 2, 0)) * 255)[:, :, 0].astype('uint8')
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.title(labels_dict[lbl[0].item()])
        plt.tight_layout()
        plt.show(block=True)
        # Keep only first image in the batch
        # imgs = [img[0, :, :, :].squeeze(0) for img in imgs]
        # lbl = lbl[0, :].squeeze(0).numpy()

        # Plot images
        """
        plt.figure(figsize=(3*len(cam_inputs), 3))
        for i, (img, cam) in enumerate(zip(imgs, cam_inputs)):
            img = (img.numpy().transpose((1, 2, 0)) * 255).astype('uint8')
            plt.subplot(1, len(cam_inputs), i+1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(cam)
        plt_title = f'Position: {np.round(lbl[:3], 3)}\nRx: {euler[0]}   Ry: {euler[1]}    Rz: {euler[2]}'
        plt.suptitle(plt_title)
        plt.tight_layout()
        plt.show(block=True)
        """
