
import pathlib
import sys

import cv2
import torch
from torchvision import models

import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import v2, InterpolationMode
import torch.nn as nn

# Import utils
parent_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
import utils
from utils_data import get_supervised_oct_data_loaders, build_image_root, RandomWrapAround, NormTransform, get_cross_valid_splits, get_lab_data_splits

class LivePrediction:
    def __init__(self, weights_path: pathlib.Path, device: torch.device):
        self.weights_path = weights_path
        self.device = device
        self.weights = torch.load(self.weights_path, map_location=self.device)
        self.model = models.resnet18('DEFAULT')
        self.sigmoid = nn.Sigmoid()
        # Update first layer for grayscale images
        self.model = utils.update_backbone_channel(self.model, 1)
        # Set classifier head
        self.model = utils.set_classifier_head(self.model, 1)
        self.model = self.model.to(self.device)
        # Load model weights
        self.model.load_state_dict(self.weights, strict=False)
        self.model.eval()

    def predict(self, img:np.ndarray):
        with torch.no_grad():
            pred = self.model(img)
            # print(pred)
            prob = self.sigmoid(pred)
        return prob


def prep_img(img: np.ndarray, img_transforms: transforms.Compose, device: torch.device):
    img = Image.fromarray(img)
    img = img_transforms(img)
    img = torch.unsqueeze(img, 0)
    img = img.to(device)
    return img

def main():
    # Init weights path + corresponding labels
    weights_path = pathlib.Path(r"C:\Users\Ana Jade\OneDrive\Documents\Ecole\TUHH\Semester 6\Masterarbeit\OCT_classification\BYOL\weights_resnet18\finetune_best_loss_oct_sup010_split_chm_lhf_lhm.pt")
    labels = {0: 'lamb_liver', 1: 'lamb_testicle'}

    # Init other vars
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{0}')
    else:
        device = torch.device('cpu')
    # img_transforms = v2.Compose([transforms.RandomEqualize(p=0.0),
    #     v2.ToTensor(),
    #     # v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
    #     v2.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
    #     NormTransform(),
    #     transforms.Grayscale()
    # ])
    img_transforms = transforms.Compose([transforms.RandomEqualize(p=0.0),
                      v2.ToTensor(),
                      v2.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
                      NormTransform(),
                      transforms.Grayscale()
                      ])
    img_path_lamb_liver = pathlib.Path(r"C:\Users\Ana Jade\OneDrive\Documents\Ecole\TUHH\Semester 6\Masterarbeit\OCT_lab_data\512mscans_noNoise_sample20\lamb_liver_p1_area11\lamb_liver_p1_area11_sine_5x5y_s3_zamp05_29697_30208.png")
    img_path_lamb_testicle = pathlib.Path(r"C:\Users\Ana Jade\OneDrive\Documents\Ecole\TUHH\Semester 6\Masterarbeit\OCT_lab_data\512mscans_noNoise_sample20\lamb_testicle_area4\lamb_testicle_area4_sine_5x5y_s8_zamp05_24577_25088.png")

    # img_root_lamb_liver = pathlib.Path(
    #     r"C:\Users\Ana Jade\OneDrive\Documents\Ecole\TUHH\Semester 6\Masterarbeit\OCT_lab_data\512mscans_noNoise_sample20\lamb_liver_p1_area11")
    # img_root_lamb_testicle = pathlib.Path(
    #     r"C:\Users\Ana Jade\OneDrive\Documents\Ecole\TUHH\Semester 6\Masterarbeit\OCT_lab_data\512mscans_noNoise_sample20\lamb_testicle_area4")
    # img_paths_lamb_liver = list(img_root_lamb_liver.glob("*.png"))
    # img_paths_lamb_testicle = list(img_root_lamb_testicle.glob("*.png"))
    # random_imgs = img_paths_lamb_liver[:5] + img_paths_lamb_testicle[:5]



    # Init model
    model = LivePrediction(weights_path, device)

    # Setup image
    for img_path in [img_path_lamb_liver, img_path_lamb_testicle]: # random_imgs
        lbl = 'liver' if 'liver' in img_path.__str__() else 'testicle'
        img = cv2.imread(img_path)
        img = prep_img(img, img_transforms, device)
        prob = model.predict(img)
        pred = prob > 0.5
        print(f"Label: {lbl} || Prediction: {labels[pred.item()]} (logit: {prob.item()})")




if __name__ == "__main__":
    main()
