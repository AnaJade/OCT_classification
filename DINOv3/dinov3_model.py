import pathlib
import sys
from tqdm import tqdm
import torch
import timm
from peft import LoraModel, LoraConfig, TaskType, PeftModel
from torchvision import models
import torch.nn as nn


# Import utils
parent_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
import utils


class DINO_LoRA(torch.nn.Module):
    def __init__(
            self,
            args,
            classifier_ckp_path=None,
            lora_ckp_path=None,
    ):
        super().__init__()
        self.args = args
        self.classifier_best_weights_path = classifier_ckp_path
        self.lora_best_weights_path = lora_ckp_path
        # Define other variables
        self.arch = args.arch
        self.device = args.device
        self.use_lora = args.use_lora
        self.ch_in = args.img_channel
        self.save_folder = args.save_folder
        self.dino_model = None

        # Load dino model minus classification layer
        self.load_model()
        self.arch = self.dino_model.__class__.__name__

        # Update input layer dim
        if self.ch_in == 1:
            self.dino_model = utils.update_backbone_channel(self.dino_model, self.ch_in)

        # Add lora adapter
        if self.use_lora:
            self.add_lora()

        # Add linear layer
        self.add_linear_layer()
        self.dino_model.to(args.device)

        # Define classifier weights path
        if self.classifier_best_weights_path is None:
            self.classifier_best_weights_path = self.args.save_folder.joinpath(f'{self.args.dataset_name}_classifier_best_loss.pt')

        # Define lora weights path
        if self.lora_best_weights_path is None:
            self.lora_best_weights_path = self.args.save_folder.joinpath(f'{self.args.dataset_name}_lora_best_loss.pt')


    def load_model(self):
        # Documentation: https://huggingface.co/collections/timm/timm-dinov3
        weight_dict = {'convnextt': 'convnext_tiny.dinov3_lvd1689m',
                       'vits': 'vit_small_patch16_dinov3_qkvb.lvd1689m'}
        num_classes = 1 if len(self.args.labels_dict.keys()) == 2 else len(self.args.labels_dict.keys())
        self.dino_model = timm.create_model(weight_dict[self.arch],
                                            pretrained=True,
                                            num_classes=0) # set to 0 to remove linear layer

    def add_lora(self):
        # Add LoRA
        # For transformer: target attention blocks
        # For CNN: conv layers
        if 'ConvNeXt' in self.arch:
            target_modules = ['stem.0', 'conv_dw']
            dino_modules = dict([*self.dino_model.named_modules()])
            dino_conv_modules = {l: dino_modules[l] for l in
                                 [la for la in dino_modules.keys() if la.endswith('conv_dw')]}
            rank = max([l.groups for l in dino_conv_modules.values()])
            alpha = rank
        elif 'Eva' in self.arch:
            target_modules = ['qkv']
            rank = 8
            alpha = 16
        else:
            target_modules = None
            rank = 8
            alpha = rank
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
        )
        self.dino_model = LoraModel(self.dino_model, lora_config, "lora")

        trainable_params = sum(
            p.numel() for p in self.dino_model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.dino_model.parameters())
        print(f"Trainable params in backbone: {trainable_params:,} / {total_params:,}")

    def add_linear_layer(self, num_classes=None):
        if num_classes is None:
            num_classes = 1 if len(self.args.labels_dict.keys()) == 2 else len(self.args.labels_dict.keys())
        if 'ConvNeXt' in self.arch:
            dim_mlp = self.dino_model.model.head.in_features
            self.dino_model.model.head.fc = nn.Linear(dim_mlp, num_classes)
        elif 'Eva' in self.arch:  # DINOv3 ViT
            dim_mlp = self.dino_model.model.blocks[-1].mlp.fc2.out_features
            self.dino_model.model.head = nn.Linear(dim_mlp, num_classes)
        # Load weights if file exists
        # if self.classifier_best_weights_path is not None and self.classifier_best_weights_path.exists():
        #     self.dino_model.model.head.load_state_dict(torch.load(self.classifier_best_weights_path, weights_only=True))

        trainable_params = sum(
            p.numel() for p in self.dino_model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.dino_model.parameters())
        print(f"Total trainable params: {trainable_params:,} / {total_params:,}")

    def train(self, train_loader, valid_loader, criterion, opt, wandb_log=False, project_name=None):
        if wandb_log:
            utils.wandb_init(project_name, hyperparams=vars(self.args))
        best_epoch = 0
        best_valid_loss = 1e6
        for epoch in range(self.args.epochs):
            print(f"\n================================\n"
                  f"Epoch {epoch}")
            if (epoch - best_epoch) >= self.args.patience:
                print(f'Loss has not improved for {self.args.patience} epochs. Training has stopped')
                print(f'Best loss was {best_valid_loss} @ epoch {best_epoch}')
                break
            avg_epoch_train_loss = []
            avg_epoch_valid_loss = []
            self.dino_model.train()
            # with torch.autocast(device_type=f'cuda:{self.args.gpu_index}', dtype=torch.float16):
            for images, labels in tqdm(train_loader, desc='Training'):
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)
                if len(labels.shape) == 1:
                    labels = labels.unsqueeze(1)
                opt.zero_grad()
                preds = self.dino_model(images)
                batch_loss = criterion(preds, labels)
                batch_loss.backward()
                opt.step()
                avg_epoch_train_loss.append(batch_loss)
                if wandb_log:
                    utils.wandb_log('batch', loss=batch_loss)
            avg_epoch_train_loss = float(torch.mean(torch.stack(avg_epoch_train_loss)).cpu().detach().numpy())
            print(f"Average epoch train loss: {avg_epoch_train_loss}")
            if wandb_log:
                utils.wandb_log('epoch', train_loss=avg_epoch_train_loss)

            # Get validation loss
            self.dino_model.eval()
            with torch.no_grad():
                for images, labels in tqdm(valid_loader, desc='Validation'):
                    images = images.to(self.args.device)
                    labels = labels.to(self.args.device)
                    preds = self.dino_model(images)
                    batch_loss = criterion(preds, labels)
                    avg_epoch_valid_loss.append(batch_loss)
                avg_epoch_valid_loss = float(torch.mean(torch.stack(avg_epoch_valid_loss)).cpu().detach().numpy())
                print(f"Average epoch valid loss: {avg_epoch_valid_loss}")
                if wandb_log:
                    utils.wandb_log('epoch', valid_loss=avg_epoch_valid_loss)

            if avg_epoch_valid_loss < best_valid_loss:
                print(f'New best loss achieved @ epoch {epoch}: {avg_epoch_valid_loss}')
                best_epoch = epoch
                best_valid_loss = avg_epoch_valid_loss
                # classifier_weights = self.dino_model.model.head.state_dict()
                torch.save(self.dino_model.model.head.state_dict(), self.classifier_best_weights_path)
                if self.use_lora:
                    lora_weights = self.dino_model.state_dict()
                    lora_weights = {n: w for n, w in lora_weights.items() if 'lora' in n}
                    torch.save(lora_weights, self.lora_best_weights_path)

    def test(self, test_loader):
        # Update model weights
        print(f'Loading best model weghts...')
        self.dino_model.model.head.load_state_dict(torch.load(self.classifier_best_weights_path, map_location=self.device, weights_only=True))
        if self.use_lora:
            # Load original state dict
            model_weights = self.dino_model.state_dict()
            # Overwrite with saved weights
            lora_weights = torch.load(self.lora_best_weights_path, map_location=self.device, weights_only=True)
            # Double check weights
            lora_layers = list(lora_weights.keys())
            # print([model_weights[l].equal(lora_weights[l]) for l in lora_layers])
            # Load into model
            updated_weights = {l:w if w not in list(lora_weights.keys()) else lora_layers[l] for l, w in model_weights.items()}
            self.dino_model.load_state_dict(updated_weights, strict=False)
        preds_all = []
        labels_all = []
        self.dino_model.eval()
        print(f'Getting test set predictions...')
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Testing'):
                images = images.to(self.args.device)
                pred = self.dino_model(images)
                preds_all.append(pred)
                labels_all.append(labels)
        preds_all = torch.concat(preds_all, dim=0).detach().to('cpu')
        labels_all = torch.concat(labels_all, dim=0).detach().to('cpu')
        return preds_all, labels_all

