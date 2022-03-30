import json
import matplotlib.pyplot as plt
import neptune.new as neptune
import numpy as np
import numpy.random as nr
# search hyperparameter
import optuna
import pandas as pd
import seaborn as sns
import torch  # For building the networks
import torch.nn as nn
import torch.nn.functional as F
from neptune.new.types import File
from numba import jit
from numpy.random import binomial, multivariate_normal
from scipy.integrate import simps
from scipy.linalg.special_matrices import toeplitz
from scipy.stats import gaussian_kde
from scipy.stats.stats import mode
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm

from pycaus.survcaus.losses.loss import *
from apps.surv_files.simulationnew import *
from apps.surv_files.utils import *

scaler = preprocessing.MinMaxScaler()


args_cuda = False  # torch.cuda.is_available()


"""
Model architecture
"""


class NetCFRSurv(nn.Module):

    def __init__(self, in_features, encoded_features, out_features, alpha=1):
        super().__init__()

        self.psi = nn.Sequential(
            nn.Linear(in_features-1, 32),  nn.LeakyReLU(),
            nn.Linear(32, 32),  nn.ReLU(),
            nn.Linear(32, 28),  nn.LeakyReLU(),
            nn.Linear(28, encoded_features),
        )

        self.surv_net = nn.Sequential(
            nn.Linear(encoded_features + 1, 128), nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 50),  nn.ReLU(),
            nn.Linear(50, out_features),
        )

        self.alpha = alpha
        self.loss_surv = NLLPMFLoss()  # NLLLogistiHazardLoss()
        self.loss_wass = WassLoss()  # IPM

    def forward(self, input):
        """
        The forward function takes the input and computes the output of the network.
        The output is computed by applying a linear transformation to the input, then
        applying a logistic sigmoid activation function. The computation is done in one
        line of code!

            Args:
                x (Tensor): Input tensor with shape (batch_size, num_features)

            Returns:
                y (Tensor): Output tensor with shape (batch_size, 1)

        :param self: Used to access the variables and methods of the class.
        :param input: Used to pass the input data.
        :return: the output of the network, which is a tensor containing the values of (t)$ for each time step.

        """
        x, t = get_data(input)
        self.input = input
        psi = self.psi(x)
        # psi_inv = self.psi_inv(psi)
        psi_t = torch.cat((psi, t), 1)
        phi = self.surv_net(psi_t)
        return phi, psi_t

    def get_repr(self, input):
        x, t = get_data(input)
        return torch.cat((self.psi(x), t), 1)

    def predict(self, input):
        """
        The predict function returns the predicted class for a single data point.

        Args:
            input (numpy.array): A representation of a piece of data in numpy array format.  This can be in either image or vector format, so long as the function is able to accept it as input and operate on it!  If you are using images, this will be an array consisting of 3 subarrays, each containing the R G B values for each pixel location.

        :param self: Used to access the attributes and methods of the class in python.
        :param input: Used to pass the input data to the function.
        :return: the inner product of the input and the weight vector.

        :doc-author: Trelent
        """
        psi_t = self.get_repr(input)
        return self.surv_net(psi_t)