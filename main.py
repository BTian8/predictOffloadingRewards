import numpy as np
import argparse
import os
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt

import torch
from torchvision.ops import roi_align, roi_pool
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet
from sklearn.svm import SVR

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


def load_feature(path, stage, pool=True, batch_size=128, func="avg", size=4):
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


def fit_BR(train_feature, val_feature, train_reward, opts=_BROPT):
    """
    Fit a Bayesian ridge regression model that predicts offloading reward based on weak detector feature map.
    :param train_feature: weak detector feature maps for the training dataset.
    :param val_feature: weak detector feature maps for the validation dataset.
    :param train_reward: offloading rewards for the training dataset.
    :param opts: options for fitting the regression model.
    :return: the estimated offloading reward for the training and validation dataset.
    """
    train_feature = [x.flatten() for x in train_feature]
    val_feature = [x.flatten() for x in val_feature]
    reg = BayesianRidge(alpha_1=opts.alpha_1, alpha_2=opts.alpha_2, lambda_1=opts.lambda_1, lambda_2=opts.lambda_2).fit(
        train_feature, train_reward)
    train_est, val_est = reg.predict(train_feature), reg.predict(val_feature)
    return train_est, val_est


def fit_LR(train_feature, val_feature, train_reward, opts=None):
    """Fit a linear regression model to predict offloading reward."""
    train_feature = [x.flatten() for x in train_feature]
    val_feature = [x.flatten() for x in val_feature]
    reg = LinearRegression().fit(train_feature, train_reward)
    train_est, val_est = reg.predict(train_feature), reg.predict(val_feature)
    return train_est, val_est


@dataclass
class ENOpt:
    """Options for the Elastic net regression model."""
    alpha: float = 1.0  # Constant that multiplies the penalty terms.
    l1_ratio: float = 0.5  # The ElasticNet mixing parameter.


_ENOPT = ENOpt()


def fit_EN(train_feature, val_feature, train_reward, opts=_ENOPT):
    """Fit an elastic net model to predict offloading reward."""
    train_feature = [x.flatten() for x in train_feature]
    val_feature = [x.flatten() for x in val_feature]
    reg = ElasticNet(alpha=opts.alpha, l1_ratio=opts.l1_ratio).fit(train_feature, train_reward)
    train_est, val_est = reg.predict(train_feature), reg.predict(val_feature)
    return train_est, val_est


@dataclass
class SVROpt:
    """Options for the support vector regression model."""
    kernel: str = 'rbf'  # Specifies the kernel type to be used in the algorithm.
    gamma: str = 'scale'  # Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
    C: float = 1.0  # Regularization parameter.


_SVROPT = SVROpt()


def fit_SVR(train_feature, val_feature, train_reward, opts=_SVROPT):
    """Fit a support vector regression model to predict offloading reward."""
    train_feature = [x.flatten() for x in train_feature]
    val_feature = [x.flatten() for x in val_feature]
    reg = SVR(kernel=opts.kernel, gamma=opts.gamma, C=opts.C).fit(train_feature, train_reward)
    train_est, val_est = reg.predict(train_feature), reg.predict(val_feature)
    return train_est, val_est


@dataclass
class GBROpt:
    """Options for the Gradient Boosting regression model."""
    learning_rate: float = 0.1
    n_estimators: int = 100  # The number of boosting stages to perform.
    subsample: float = 1.0  # The fraction of samples to be used for fitting the individual base learners。


_GBROPT = GBROpt()


def fit_GBR(train_feature, val_feature, train_reward, opts=_GBROPT):
    """Fit a Gradient Boosting Regressor to predict offloading reward."""
    train_feature = [x.flatten() for x in train_feature]
    val_feature = [x.flatten() for x in val_feature]
    reg = GradientBoostingRegressor(learning_rate=opts.learning_rate, n_estimators=opts.n_estimators,
                                    subsample=opts.subsample).fit(train_feature, train_reward)
    train_est, val_est = reg.predict(train_feature), reg.predict(val_feature)
    return train_est, val_est


def main(opts):
    # Load the weak detector feature maps for the training and validation dataset.
    train_feature = load_feature(opts.train_dir, opts.stage, size=4)
    val_feature = load_feature(opts.val_dir, opts.stage, size=4)
    # Load the offloading rewards for the training dataset.
    train_reward = np.load(opts.label)
    assert len(train_feature) == len(
        train_reward), "Inconsistent number of training feature maps and offloading rewards."
    # Select and fit the regression model.
    model_names = ['BR', 'LR', 'EN', 'SVR', 'GBR']
    models = [fit_BR, fit_LR, fit_EN, fit_SVR, fit_GBR]
    try:
        model_idx = model_names.index(opts.model)
        model = models[model_idx]
    except ValueError:
        print("Please select a regression model from 'BR' (Bayesian Ridge), 'LR' (Linear Regression), " +
              "'EN' (Elastic Net), 'SVR' (Support Vector Regression), and 'GBR' (Gradient Boosting Regressor).")
    train_est, val_est = model(train_feature, val_feature, train_reward)
    # Save the estimated offloading reward.
    Path(opts.save_dir).mkdir(parents=True, exist_ok=True)
    np.savez(os.path.join(opts.save_dir, f'{opts.model}_estimate.npz'), train=train_est, val=val_est)
    return


def getargs():
    """Parse command line arguments."""
    args = argparse.ArgumentParser()
    args.add_argument('train_dir', help="Directory that saves the weak detector feature maps for the training set.")
    args.add_argument('val_dir', help="Directory that saves the weak detector feature maps for the validation set.")
    args.add_argument('label', help="Path to the offloading reward for the training set.")
    args.add_argument('save_dir', help="Directory to save the estimated offloading reward.")
    args.add_argument('--stage', type=int, default=23,
                      help="Stage number of the selected feature map. For yolov5 detectors, " +
                           "this should be a number between [0, 23].")
    args.add_argument('--model', type=str, default='LR',
                      help="Type of the regression model. Available choices include 'BR' (Bayesian Ridge), " +
                           "'LR' (Linear Regression), 'EN' (Elastic Net), 'SVR' (Support Vector Regression), " +
                           "and 'GBR' (Gradient Boosting Regressor).")
    return args.parse_args()


if __name__ == '__main__':
    main(getargs())
