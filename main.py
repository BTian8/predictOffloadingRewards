import numpy as np
import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt

import torch
from torchvision.ops import roi_align, roi_pool
from torch.utils.data import DataLoader
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error as mse

from net import EdgeDetectionDataset, EdgeDetectionNet

"""Train a regression model that maps the weak detector's intermediate feature map to the offloading reward."""


def roi_padding(x):
    """
    Pad the feature map and set it as an roi for pooling.
    :param x: the feature map for padding.
    :return: the feature map and its coordinates.
    """
    c, h, w = x.shape
    if h < w:
        padding = ((0, 0), (0, w - h), (0, 0))
    else:
        padding = ((0, 0), (0, 0), (0, h - w))
    # Padding value does not matter, use the default zero padding mode.
    feature_map = np.pad(x, padding)
    coord = [[0, 0, w, h]]
    return feature_map, coord


def load_feature(path, stage, pool=True, batch_size=128, func="avg", size=8):
    """
    Load the feature maps.
    :param path: path to the folder where the feature maps are stored.
    :param stage: the stage number of the feature map.
    :param pool: whether the feature maps should be resized with roi pooling.
    :param batch_size: batch size for pooling image, only active when "pool" is enabled.
    :param func: function for pooling image, "avg" or "max", only active when "pool" is enabled.
    :param size: size (H, W) of the feature map after pooling, only active when "pool" is enabled.
    :return: the loaded (pooled) feature maps as a list or ndarray.
    """
    # The stage names of yolov5 detectors.
    v5_names = ['Conv', 'Conv', 'C3', 'Conv', 'C3', 'Conv', 'C3', 'Conv', 'C3', 'SPPF', 'Conv', 'Upsample', 'Concat',
                'C3', 'Conv', 'Upsample', 'Concat', 'C3', 'Conv', 'Concat', 'C3', 'Conv', 'Concat', 'C3']
    data = list()
    # Read the data from each npy file.
    images = [f for f in os.listdir(path) if not os.path.isfile(os.path.join(path, f))]
    if pool:
        pool_func = roi_align if func == "avg" else roi_pool
        # Split images into batches and resize the feature maps through roi pooling.
        for ndx in range(0, len(images), batch_size):
            features, coords = list(), list()
            for img_name in images[ndx:min(ndx + batch_size, len(images))]:
                file_path = os.path.join(path, img_name, f"stage{stage}_{v5_names[stage]}_features.npy")
                file_data = np.load(file_path)
                # Pad the feature maps to set the original feature map as an roi for the pooling operation.
                feature_map, coord = roi_padding(file_data)
                features.append(feature_map)
                coords.append(torch.tensor(coord, dtype=torch.float))
            data.append(pool_func(torch.from_numpy(np.array(features)), coords, size).numpy())
        data = np.concatenate(data)
        # # Visualize the feature maps.
        # for img_name, feature_map in zip(images, data):
        #     file_path = os.path.join(path, img_name, f"stage{stage}_{v5_names[stage]}_features_pooled.png")
        #     n = 32
        #     fig, ax = plt.subplots(4, 8, tight_layout=True)
        #     ax = ax.ravel()
        #     plt.subplots_adjust(wspace=0.05, hspace=0.05)
        #     for i in range(n):
        #         ax[i].imshow(feature_map[i])
        #         ax[i].axis('off')
        #     plt.savefig(file_path, dpi=300, bbox_inches='tight')
        #     plt.close()
    else:
        # Directly load the feature maps without modification.
        for img_name in images:
            file_path = os.path.join(path, img_name, f"stage{stage}_{v5_names[stage]}_features.npy")
            file_data = np.load(file_path)
            data.append(file_data)
    return data


@dataclass
class BROpt:
    """Options for the Bayesian ridge regression model."""
    alpha_1: float = 1e-6  # Shape parameter for the Gamma distribution prior over the alpha parameter.
    alpha_2: float = 1e-6  # Rate parameter for the Gamma distribution prior over the alpha parameter.
    lambda_1: float = 1e-6  # Shape parameter for the Gamma distribution prior over the lambda parameter.
    lambda_2: float = 1e-6  # Rate parameter for the Gamma distribution prior over the lambda parameter.


_BROPT = BROpt()


def fit_BR(train_feature, val_feature, train_reward, val_reward, opts=_BROPT):
    """
    Fit a Bayesian ridge regression model that predicts offloading reward based on weak detector feature map.
    :param train_feature: weak detector feature maps for the training dataset.
    :param val_feature: weak detector feature maps for the validation dataset.
    :param train_reward: offloading rewards for the training dataset.
    :param val_reward: offloading rewards for the validation dataset.
    :param opts: options for fitting the regression model.
    :return: the estimated offloading reward for the training and validation dataset.
    """
    train_feature = [x.flatten() for x in train_feature]
    val_feature = [x.flatten() for x in val_feature]
    reg = BayesianRidge(alpha_1=opts.alpha_1, alpha_2=opts.alpha_2, lambda_1=opts.lambda_1, lambda_2=opts.lambda_2).fit(
        train_feature, train_reward)
    train_est, val_est = reg.predict(train_feature), reg.predict(val_feature)
    train_mse, val_mse = mse(train_reward, train_est), mse(val_reward, val_est)
    print(f"Trained Bayesian Ridge model with training MSE:{train_mse}, validation MSE:{val_mse}")
    return train_est, val_est


def fit_LR(train_feature, val_feature, train_reward, val_reward, opts=None):
    """Fit a linear regression model to predict offloading reward."""
    train_feature = [x.flatten() for x in train_feature]
    val_feature = [x.flatten() for x in val_feature]
    reg = LinearRegression().fit(train_feature, train_reward)
    train_est, val_est = reg.predict(train_feature), reg.predict(val_feature)
    train_mse, val_mse = mse(train_reward, train_est), mse(val_reward, val_est)
    print(f"Trained Linear Regression model with training MSE:{train_mse}, validation MSE:{val_mse}")
    return train_est, val_est


@dataclass
class ENOpt:
    """Options for the Elastic net regression model."""
    alpha: float = 1.0  # Constant that multiplies the penalty terms.
    l1_ratio: float = 0.5  # The ElasticNet mixing parameter.


_ENOPT = ENOpt()


def fit_EN(train_feature, val_feature, train_reward, val_reward, opts=_ENOPT):
    """Fit an elastic net model to predict offloading reward."""
    train_feature = [x.flatten() for x in train_feature]
    val_feature = [x.flatten() for x in val_feature]
    reg = ElasticNet(alpha=opts.alpha, l1_ratio=opts.l1_ratio).fit(train_feature, train_reward)
    train_est, val_est = reg.predict(train_feature), reg.predict(val_feature)
    train_mse, val_mse = mse(train_reward, train_est), mse(val_reward, val_est)
    print(f"Trained Elastic Net model with training MSE:{train_mse}, validation MSE:{val_mse}")
    return train_est, val_est


@dataclass
class SVROpt:
    """Options for the support vector regression model."""
    kernel: str = 'rbf'  # Specifies the kernel type to be used in the algorithm.
    gamma: str = 'scale'  # Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
    C: float = 1.0  # Regularization parameter.


_SVROPT = SVROpt()


def fit_SVR(train_feature, val_feature, train_reward, val_reward, opts=_SVROPT):
    """Fit a support vector regression model to predict offloading reward."""
    train_feature = [x.flatten() for x in train_feature]
    val_feature = [x.flatten() for x in val_feature]
    reg = SVR(kernel=opts.kernel, gamma=opts.gamma, C=opts.C).fit(train_feature, train_reward)
    train_est, val_est = reg.predict(train_feature), reg.predict(val_feature)
    train_mse, val_mse = mse(train_reward, train_est), mse(val_reward, val_est)
    print(f"Trained Support Vector Regression model with training MSE:{train_mse}, validation MSE:{val_mse}")
    return train_est, val_est


@dataclass
class GBROpt:
    """Options for the Gradient Boosting regression model."""
    learning_rate: float = 0.1
    n_estimators: int = 100  # The number of boosting stages to perform.
    subsample: float = 1.0  # The fraction of samples to be used for fitting the individual base learners.


_GBROPT = GBROpt()


def fit_GBR(train_feature, val_feature, train_reward, val_reward, opts=_GBROPT):
    """Fit a Gradient Boosting Regressor to predict offloading reward."""
    train_feature = [x.flatten() for x in train_feature]
    val_feature = [x.flatten() for x in val_feature]
    reg = GradientBoostingRegressor(learning_rate=opts.learning_rate, n_estimators=opts.n_estimators,
                                    subsample=opts.subsample).fit(train_feature, train_reward)
    train_est, val_est = reg.predict(train_feature), reg.predict(val_feature)
    train_mse, val_mse = mse(train_reward, train_est), mse(val_reward, val_est)
    print(f"Trained Gradient Boosting Regressor with training MSE:{train_mse}, validation MSE:{val_mse}")
    return train_est, val_est


@dataclass
class CNNOpt:
    """Options for the Gradient Boosting regression model."""
    learning_rate: float = 1e-3  # Initial learning rate.
    gamma: float = 0.1  # Scale for updating learning rate at each milestone.
    milestones: List = field(default_factory=lambda: [10, 15, 20])  # Epochs to update the learning rate.
    max_epoch: int = 25  # Maximum number of epochs for training.
    channels: List = field(default_factory=lambda: [256, 128, 32])  # Number of channels in each conv layer.
    kernels: List = field(default_factory=lambda: [3, 3])  # Kernel size for each conv layer.
    pools: List = field(default_factory=lambda: [True, True])  # Whether max-pooling each conv layer.
    linear: List = field(default_factory=lambda: [])  # Number of features in each linear after the conv layers.


@dataclass
class CNNOpt:
    """Options for the Gradient Boosting regression model."""
    learning_rate: float = 1e-3  # Initial learning rate.
    gamma: float = 0.1  # Scale for updating learning rate at each milestone.
    milestones: List = field(default_factory=lambda: [10, 15, 20])  # Epochs to update the learning rate.
    max_epoch: int = 25  # Maximum number of epochs for training.
    batch_size: int = 64  # Batch size for model training.
    channels: List = field(default_factory=lambda: [256, 128, 32])  # Number of channels in each conv layer.
    kernels: List = field(default_factory=lambda: [3, 3])  # Kernel size for each conv layer.
    pools: List = field(default_factory=lambda: [True, True])  # Whether max-pooling each conv layer.
    linear: List = field(default_factory=lambda: [])  # Number of features in each linear after the conv layers.
    test_epoch: int = 5  # Number of epochs for periodic test using the validation set.
    model_dir: str = ''  # Directory to save the model weights.
    load: bool = False  # If model is loaded from pre-trained weights.
    save: bool = True  # If model weights need to be saved after training.


_CNNOPT = CNNOpt()


def fit_CNN(train_feature, val_feature, train_reward, val_reward, opts=_CNNOPT):
    """Fit a Convolutional Neural Network to predict offloading reward."""
    # Import pytorch.
    import torch
    from torch.utils.data import DataLoader
    # Prepare the dataset.
    train_data = EdgeDetectionDataset(train_feature, train_reward)
    val_data = EdgeDetectionDataset(val_feature, val_reward)
    train_dataloader = DataLoader(train_data, batch_size=opts.batch_size)
    val_dataloader = DataLoader(val_data, batch_size=opts.batch_size)
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    # Build the CNN model.
    model = EdgeDetectionNet(opts.channels, opts.kernels, opts.pools, opts.linear).to(device)
    print(model)
    # Load weights if specified.
    if opts.load and opts.model_dir != '':
        model.load_state_dict(torch.load(opts.model_dir))
    # Declare loss function, optimizer, and scheduler.
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opts.milestones, gamma=0.1)

    # Define the training and test function.
    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(dataloader, model, loss_fn):
        num_batches = len(dataloader)
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
        test_loss /= num_batches
        print(f"Avg Test Loss: {test_loss:>8f} \n")

    # The training loop.
    for t in range(opts.max_epoch):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        if t % opts.test_epoch == 0:
            test(val_dataloader, model, loss_fn)
        scheduler.step()
    # Estimate the offloading reward for both training and validation set.
    with torch.no_grad():
        train_est, val_est = list(), list()
        for X, y in train_dataloader:
            X, y = X.to(device), y.to(device)
            train_est.append(model(X).cpu().numpy())
        train_est = np.concatenate(train_est)
        for X, y in val_dataloader:
            X, y = X.to(device), y.to(device)
            val_est.append(model(X).cpu().numpy())
        val_est = np.concatenate(val_est)
    # Save model if specified.
    if opts.save and opts.model_dir != '':
        torch.save(model.state_dict(), opts.model_dir)
    return train_est, val_est


def main(opts):
    # Load the weak detector feature maps for the training and validation dataset.
    ifpool = opts.pool_size > 0
    train_feature = load_feature(opts.train_dir, opts.stage, pool=ifpool, size=opts.pool_size)
    val_feature = load_feature(opts.val_dir, opts.stage, pool=ifpool, size=opts.pool_size)
    # Load the offloading rewards for the training dataset.
    train_reward = np.load(opts.train_label)
    val_reward = np.load(opts.val_label)
    assert len(train_feature) == len(
        train_reward), "Inconsistent number of training feature maps and offloading rewards."
    assert len(val_feature) == len(
        val_reward), "Inconsistent number of validation feature maps and offloading rewards."
    # Select and fit the regression model.
    model_names = ['BR', 'LR', 'EN', 'SVR', 'GBR', 'CNN']
    models = [fit_BR, fit_LR, fit_EN, fit_SVR, fit_GBR, fit_CNN]
    try:
        model_idx = model_names.index(opts.model)
        model = models[model_idx]
    except ValueError:
        print("Please select a regression model from 'BR' (Bayesian Ridge), 'LR' (Linear Regression), " +
              "'EN' (Elastic Net), 'SVR' (Support Vector Regression), 'GBR' (Gradient Boosting Regressor), " +
              "and 'CNN' (Convolutional Neural Network).")
    if not ifpool:
        assert opts.model == 'CNN', "Only fully convolutional NN can take input with different shapes. " + \
                                    "Please set model to 'CNN' if you choose to skip the RoI pooling step."
        _CNNOPT.batch_size = 1
    _CNNOPT.model_dir = opts.model_dir
    train_est, val_est = model(train_feature, val_feature, train_reward, val_reward)
    # Save the estimated offloading reward.
    Path(opts.save_dir).mkdir(parents=True, exist_ok=True)
    np.savez(os.path.join(opts.save_dir, f'{opts.model}_estimate.npz'), train=train_est, val=val_est)
    return


def getargs():
    """Parse command line arguments."""
    args = argparse.ArgumentParser()
    args.add_argument('train_dir', help="Directory that saves the weak detector feature maps for the training set.")
    args.add_argument('val_dir', help="Directory that saves the weak detector feature maps for the validation set.")
    args.add_argument('train_label', help="Path to the offloading reward for the training set.")
    args.add_argument('val_label', help="Path to the offloading reward for the validation set.")
    args.add_argument('save_dir', help="Directory to save the estimated offloading reward.")
    args.add_argument('--stage', type=int, default=23,
                      help="Stage number of the selected feature map. For yolov5 detectors, " +
                           "this should be a number between [0, 23].")
    args.add_argument('--pool_size', type=int, default=8,
                      help="Size (H,W) of the feature maps after using RoI pooling. If 0, skip RoI pooling.")
    args.add_argument('--model', type=str, default='LR',
                      help="Type of the regression model. Available choices include 'BR' (Bayesian Ridge), " +
                           "'LR' (Linear Regression), 'EN' (Elastic Net), 'SVR' (Support Vector Regression), " +
                           "'GBR' (Gradient Boosting Regressor), and 'CNN' (Convolutional Neural Network).")
    args.add_argument('--model_dir', type=str, default='', help="Directory to save the (CNN) model weights.")
    return args.parse_args()


if __name__ == '__main__':
    main(getargs())
