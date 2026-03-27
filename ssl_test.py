import torch

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn import preprocessing
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, precision_score, recall_score, adjusted_rand_score, normalized_mutual_info_score, confusion_matrix, ConfusionMatrixDisplay

from SimCLR.models.resnet_simclr import FeatureModelSimCLR
from BYOL.feature_model import get_backbone

import utils

class FeatureExtractor(object):
    def __init__(self, args, approach, ckp_file):
        self.ckp_file = ckp_file
        self.approach = approach
        self.args = args

        # Define model
        if self.approach == 'simclr':
            self.model = FeatureModelSimCLR(arch=args.arch, out_dim=args.out_dim, pretrained=False, img_channel=args.img_channel)
        elif self.approach == 'byol':
            self.model, _ = get_backbone(args.arch, False)
            # Change first layer to take grayscale image
            if args.img_channel == 1:
                self.model = utils.update_backbone_channel(self.model, args.img_channel)

        # Load weights
        print(f"Loading weights from {self.ckp_file}...")
        state_dict = torch.load(self.ckp_file, map_location=self.args.device)
        if 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(args.device)

    def _inference(self, loader):
        feature_vector = []
        labels_vector = []
        for batch_x, batch_y in tqdm(loader, desc='Extracting features'):
            batch_x = batch_x.to(self.args.device)
            labels_vector.extend(batch_y)

            features = self.model(batch_x)
            feature_vector.extend(features.cpu().detach().numpy())

        feature_vector = np.array(feature_vector)
        labels_vector = np.array(labels_vector)

        print("Features shape {}".format(feature_vector.shape))
        return feature_vector, labels_vector

    def get_features(self, train_loader, test_loader):
        X_train_feature, y_train = self._inference(train_loader)
        X_test_feature, y_test = self._inference(test_loader)

        return X_train_feature, y_train, X_test_feature, y_test


class LogisticRegressionSSL(nn.Module):

    def __init__(self, n_features, n_classes):
        super().__init__()
        self.model = nn.Linear(n_features, n_classes)
        # self.model = nn.Sequential(nn.Linear(n_features, n_classes),
        #                            nn.Linear(n_classes, n_classes))

    def forward(self, x):
        return self.model(x)


class LogisticRegressionEvaluator(object):
    def __init__(self, n_features, n_classes, args):
        self.args = args
        self.log_regression = LogisticRegressionSSL(n_features, n_classes).to(self.args.device)
        self.scaler = preprocessing.StandardScaler()

    def _normalize_dataset(self, X_train, X_test):
        print("Standard Scaling Normalizer")
        self.scaler.fit(X_train)
        X_train = self.scaler.transform(X_train)
        X_test = self.scaler.transform(X_test)
        return X_train, X_test

    @staticmethod
    def _sample_weight_decay():
        # We selected the l2 regularization parameter from a range of 45 logarithmically spaced values between 10−6 and 105
        weight_decay = np.logspace(-6, 5, num=45, base=10.0)
        weight_decay = np.random.choice(weight_decay)
        print("Sampled weight decay:", weight_decay)
        return weight_decay

    def eval(self, test_loader):
        correct = 0
        total = 0

        logits_epoch = []
        y_true_epoch = []
        with torch.no_grad():
            self.log_regression.eval()
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(self.args.device), batch_y.to(self.args.device)
                outputs = self.log_regression(batch_x)

                predicted = torch.argmax(outputs, dim=1)
                labels = torch.argmax(labels, dim=1)
                batch_y = torch.argmax(batch_y, dim=1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

                # Save values
                logits_epoch.append(outputs)
                y_true_epoch.append(batch_y)

            final_acc = 100 * correct / total
            logits_epoch = torch.concat(logits_epoch)
            y_true_epoch = torch.concat(y_true_epoch)
            self.log_regression.train()
            return final_acc, logits_epoch, y_true_epoch

    def create_data_loaders_from_arrays(self, X_train, y_train, X_test, y_test):
        X_train, X_test = self._normalize_dataset(X_train, X_test)

        train = torch.utils.data.TensorDataset(torch.from_numpy(X_train),
                                               torch.from_numpy(y_train).type(torch.long))
        train_loader = torch.utils.data.DataLoader(train, batch_size=396, shuffle=False)

        test = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test).type(torch.long))
        test_loader = torch.utils.data.DataLoader(test, batch_size=512, shuffle=False)
        return train_loader, test_loader

    def train(self, X_train, y_train, X_test, y_test, save_folder):

        train_loader, test_loader = self.create_data_loaders_from_arrays(X_train, y_train, X_test, y_test)

        weight_decay = self._sample_weight_decay()

        criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
        optimizer = torch.optim.AdamW(self.log_regression.parameters(), lr=self.args.lr, weight_decay=weight_decay)

        best_nmi = 0
        best_epoch_acc = 0
        best_epoch_precision = 0
        best_epoch_recall = 0
        best_epoch_ari = 0
        best_epoch = 0
        best_labels = None
        best_preds = None
        print("Training regression model...")
        for e in tqdm(range(200)):
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.args.device), batch_y.to(self.args.device)
                optimizer.zero_grad()
                logits = self.log_regression(batch_x)
                batch_y = torch.argmax(batch_y, dim=1)
                loss = criterion(logits.type(torch.float32), batch_y.type(torch.float32))
                loss.backward()
                optimizer.step()

            acc, logits_epoch, y_true_epoch = self.eval(test_loader)

            # Get other metrics
            test_probs = F.softmax(logits_epoch, dim=1)
            preds_epoch = test_probs.argmax(1)
            eval_df_epoch = pd.DataFrame(torch.vstack((y_true_epoch.cpu(), preds_epoch.cpu())).T, columns=['label', 'pred'])
            # Calculate other metrics
            # Precision and recall
            # https://medium.com/data-science-in-your-pocket/calculating-precision-recall-for-multi-class-classification-9055931ee229
            # Precision (tp/(tp+fp)) and recall (tp/(tp+fn)), macro gives better results as micro
            precision = precision_score(eval_df_epoch['label'], eval_df_epoch['pred'], average='macro', zero_division=0)
            recall = recall_score(eval_df_epoch['label'], eval_df_epoch['pred'], average='macro')
            # Adjusted rand index (ARI)
            ari = adjusted_rand_score(eval_df_epoch['label'], eval_df_epoch['pred'])
            # Normalized mutual information (NMI)
            nmi = normalized_mutual_info_score(eval_df_epoch['label'], eval_df_epoch['pred'])
            if nmi > best_nmi:
                # print("Saving new model with accuracy {}".format(epoch_acc))
                best_nmi = nmi
                best_epoch = e
                # best_epoch_acc = acc
                # best_epoch_precision = precision
                # best_epoch_recall = recall
                best_epoch_ari = ari
                best_labels = eval_df_epoch['label']
                best_preds = eval_df_epoch['pred']
                torch.save(self.log_regression.state_dict(), save_folder.joinpath('log_regression.pth'))

        print("--------------")
        print("Done training")
        print(f"Best nmi @ epoch {best_epoch}: {best_nmi}")
        # print(f"Accuracy @ epoch {best_epoch}: {best_epoch_acc}")
        # print(f"Precision @ epoch {best_epoch}: {best_epoch_precision}")
        # print(f"Recall @ epoch {best_epoch}: {best_epoch_recall}")
        print(f"ARI @ epoch {best_epoch}: {best_epoch_ari}")
        print(f"Classification report @ epoch {best_epoch}")
        report = classification_report(best_labels, best_preds,
                                            target_names=self.args.labels_dict.values(),
                                            digits=4, zero_division=np.nan)
        print(report)
        # Get confusion matrix display
        cm = confusion_matrix(best_labels, best_preds)
        cm_plot = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.args.labels_dict.values())
        cm_plot.plot()
        plt.title('Confusion matrix')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        cm_path = f"confusion_matrix_{self.args.dataset_name}.png"
        plt.savefig(save_folder.joinpath(cm_path))
        plt.show()
