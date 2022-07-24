import numpy as np
import pandas as pd
import skimage
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet
from sklearn.svm import SVR

from sklearn.model_selection import cross_val_score  # 交叉验证
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score

import argparse
from argparse import ArgumentParser

plt.rc("font", size=14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

# Read data to dataframe

stageWeight = np.load('./stage0_Conv_features.npy', allow_pickle=True, encoding="latin1")


# 建立贝叶斯岭回归模型
br_model = BayesianRidge()

# 普通线性回归
lr_model = LinearRegression()

# 弹性网络回归模型
etc_model = ElasticNet()

# 支持向量机回归
svr_model = SVR()

# 梯度增强回归模型对象
gbr_model = GradientBoostingRegressor()

# 不同模型的名称列表
model_names = ['BayesianRidge', 'LinearRegression', 'ElasticNet', 'SVR', 'GBR']
# 不同回归模型
model_dic = [br_model, lr_model, etc_model, svr_model, gbr_model]


def load_offloading_reward():
    coco_val = np.load('./offloading_reward/coco_val.npy', allow_pickle=True, encoding="latin1")
    coco_train = np.load('./offloading_reward/coco_train.npy', allow_pickle=True, encoding="latin1")
    voc_val = np.load('./offloading_reward/voc_val.npy', allow_pickle=True, encoding="latin1")
    voc_train = np.load('./offloading_reward/voc_train.npy', allow_pickle=True, encoding="latin1")
    return coco_val, coco_train, voc_val, voc_train


# 计算回归系数
def regression_beta(stage_weight, offloading_reward):
    Xtx = np.dot(stage_weight.T, stage_weight)
    Xty = np.dot(stage_weight.T, offloading_reward)
    beta = np.linalg.solve(Xtx, Xty)

coco_val = np.load('./offloading_reward/coco_val.npy', allow_pickle=True, encoding="latin1")


# stageWeight = skimage.measure.block_reduce(stageWeight.reshape(-1), block_size=(5000,), func=np.mean)
regression_beta(stageWeight, coco_val)

def global_max_pooling_forward(z):
    """
    全局最大池化前向过程
    :param z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :return:
    """
    return np.max(np.max(z, axis=-1), -1)


res = global_max_pooling_forward(stageWeight)
print(res.ndim)

