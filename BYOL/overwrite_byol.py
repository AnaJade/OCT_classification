import copy
import pathlib
import sys
import torch
from torch import nn
from byol_pytorch import BYOL
from byol_pytorch.byol_pytorch import default, get_module_device, loss_fn, RandomApply, EMA, MLP, NetWrapper
from torchvision import transforms as T

# Import utils
parent_dir = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))
import utils


class BYOL_custom(BYOL):
    def __init__(
        self,
        net,
        image_size,
        ch_in = 3,
        use_iipp=False,
        hidden_layer = -2,
        projection_size = 256,
        projection_hidden_size = 4096,
        augment_fn = None,
        augment_fn2 = None,
        moving_average_decay = 0.99,
        use_momentum = True,
        sync_batchnorm = None
    ):
        # Reset channel to 3 and iipp to false for super init
        if ch_in !=  3:
            net_3ch = utils.update_backbone_channel(copy.deepcopy(net), 3)
            """
            if net.__class__.__name__ == 'ResNet':
                net.conv1.in_channels = 3
                net.conv1.weight = nn.Parameter(torch.concat(3*[net.conv1.weight], dim=1))
            if net.__class__.__name__ == 'VisionTransformer':
                net.conv_proj.in_channels = 3
                net.conv_proj.weight = nn.Parameter(torch.concat(3*[net.conv_proj.weight], dim=1))
            """
        else:
            net_3ch = copy.deepcopy(net)
        self.ch_in = 3
        self.use_iipp = False
        super().__init__(net_3ch, image_size, hidden_layer, projection_size, projection_hidden_size, augment_fn,
                         augment_fn2, moving_average_decay, use_momentum, sync_batchnorm)
        # Reset to desired values
        self.net = net
        self.ch_in = ch_in
        self.use_iipp = use_iipp

        # default SimCLR augmentation
        default_aug_mean = [0.485, 0.456, 0.406]
        default_aug_std = [0.229, 0.224, 0.225]
        if self.ch_in == 1:
            default_aug_mean = [sum(default_aug_mean) / len(default_aug_mean)]
            default_aug_std = [sum(default_aug_std) / len(default_aug_std)]
        DEFAULT_AUG = torch.nn.Sequential(
            RandomApply(
                T.ColorJitter(0.8, 0.8, 0.8, 0.2),
                p=0.3
            ),
            T.RandomGrayscale(p=0.2),
            T.RandomHorizontalFlip(),
            RandomApply(
                T.GaussianBlur((3, 3), (1.0, 2.0)),
                p=0.2
            ),
            T.RandomResizedCrop((image_size, image_size)),
            T.Normalize(
                mean=torch.tensor(default_aug_mean),
                std=torch.tensor(default_aug_std)),
        )

        self.augment1 = default(augment_fn, DEFAULT_AUG)
        self.augment2 = default(augment_fn2, self.augment1)

        self.online_encoder = NetWrapper(
            self.net,
            projection_size,
            projection_hidden_size,
            layer=hidden_layer,
            use_simsiam_mlp=not use_momentum,
            sync_batchnorm=sync_batchnorm
        )

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        if self.use_iipp:
            self.forward(torch.randn(2, 2 * self.ch_in, image_size, image_size, device=device))
        else:
            self.forward(torch.randn(2, self.ch_in, image_size, image_size, device=device))


    def forward(
                self,
                x,
                return_embedding=False,
                return_projection=True
        ):
            assert not (self.training and x.shape[
                0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

            if return_embedding:
                return self.online_encoder(x, return_projection=return_projection)

            if self.use_iipp:
                # Unstack images along the channel dim
                image_one = self.augment1(x[:, :self.ch_in, :, :])
                image_two = self.augment2(x[:, self.ch_in:, :, :])
            else:
                image_one, image_two = self.augment1(x), self.augment2(x)

            images = torch.cat((image_one, image_two), dim=0)

            online_projections, _ = self.online_encoder(images)  # resnet18 shape: [2b, 512]
            online_predictions = self.online_predictor(online_projections)

            online_pred_one, online_pred_two = online_predictions.chunk(2, dim=0)

            with torch.no_grad():
                target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder

                target_projections, _ = target_encoder(images)
                target_projections = target_projections.detach()

                target_proj_one, target_proj_two = target_projections.chunk(2, dim=0)

            loss_one = loss_fn(online_pred_one, target_proj_two.detach())
            loss_two = loss_fn(online_pred_two, target_proj_one.detach())

            loss = loss_one + loss_two
            return loss.mean()

