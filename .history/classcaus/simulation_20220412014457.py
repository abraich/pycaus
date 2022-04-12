
import pandas as pd
import torch
from utils import *
import numpy as np
from scipy.linalg.special_matrices import toeplitz
from numpy.random import binomial, multivariate_normal
import matplotlib.pyplot as plt


class Simulation:
    def __init__(self, param_sim):
        self.n_features = param_sim['n_features']
        self.n_classes = param_sim['n_classes']
        self.n_samples = param_sim['n_samples']
        self.wd_para = param_sim['wd_para']
        self.beta = param_sim['beta']
        self.coef_tt = param_sim['coef_tt']
        self.rho = param_sim['rho']
        self.cov = toeplitz(self.rho ** np.arange(0, self.n_features))
        # multivariate normal
        self.X = multivariate_normal(
            np.zeros(self.n_features), self.cov, self.n_samples)

        self.path_data = param_sim['path_data']

    def simule(self):
        """
        The simule function simulates a dataset with the following characteristics:
            - n_features : number of features (default = 10)
            - wd_para : weight decay parameter for the treatment covariates (default = 0.1)
            - beta : coefficient vector for the true model (default = [0, 1, ... , 0])

        :param self: Used to reference a class instance.
        :return: the data_frame with the generated data and the Wasserstein distance between X and tt.
        """
        # treatment simulation
        idx = np.arange(self.n_features)
        params_tt = np.exp(-idx / 10.)

        p_tt = sigmoid(self.X.dot(params_tt))
        tt = binomial(1, p_tt)  # treatment

        for j in range(self.n_features):
            self.X[:, j] -= self.wd_para/2 * tt * params_tt[j]
            self.X[:, j] += self.wd_para/2 * (1-tt) * (1-params_tt[j])
            
        # transform treatment specific covariates in tensor and compute the Wasserstein distance
        def get_wd(tt,X) :
            mask_1, mask_0 = (tt == 1), (tt == 0)
            X_tesnor = torch.tensor(X).float()
            x_1 = X_tesnor[mask_1]
            x_0 = X_tesnor[mask_0]
            m = max(x_0.shape, x_1.shape)
            z0 = torch.zeros(m)
            m0 = x_0.shape[0]
            z0[:m0, ] = x_0
            z1 = torch.zeros(m)
            m1 = x_1.shape[0]
            z1[:m1, ] = x_1
            wd = SinkhornDistance(eps=0.001, max_iter=100,
                              reduction=None)(z0, z1).item()
            return wd
        wd = get_wd(tt,self.X)   
            
            
            
        # normalize 
        self.X = self.X / np.linalg.norm(self.X, axis=1)[:, None]
        
        # pi (x,t) =P(Y=1|X=x,T=t)
        s_f = self.X.dot(self.beta) + self.coef_tt * tt
        s_cf = self.X.dot(self.beta) + self.coef_tt * (1-tt)
        pi_f = sigmoid(s_f/np.linalg.norm(s_f))
        pi_cf = sigmoid(s_cf/np.linalg.norm(s_cf))
        Y_f = binomial(1, pi_f)
        Y_cf = binomial(1, pi_cf)
        Y_0 = (1-tt) * Y_f + tt * Y_cf
        Y_1 = tt * Y_f + (1-tt) * Y_cf
        pi_0 = (1-tt) * pi_f + tt * pi_cf
        pi_1 = tt * pi_f + (1-tt) * pi_cf
        KLD = KL(pi_0, pi_1)

        

        # data_frame construction
        colmns = ["X" + str(j) for j in range(1, self.n_features + 1)]
        data_sim = pd.DataFrame(data=self.X, columns=colmns)

        # scaling
        #data_sim = pd.DataFrame(scaler.fit_transform(data_sim),columns=colmns)

        data_sim["tt"] = tt
        data_sim["Y_f"] = Y_f
        data_sim["Y_cf"] = Y_cf
        data_sim["Y_0"] = Y_0
        data_sim["Y_1"] = Y_1
        data_sim["pi_0"] = pi_0
        data_sim["pi_1"] = pi_1
        data_sim["pi_f"] = pi_f
        data_sim["pi_cf"] = pi_cf

        self.data_sim = data_sim

        #  treatment proportions
        self.perc_treatement = int((data_sim["tt"].mean() * 100))
        self.y_0_perc = int((data_sim["Y_0"].mean() * 100))
        self.y_1_perc = int((data_sim["Y_1"].mean() * 100))
        
        # Wasserstein distances
        print("WD = ", wd)
        print(f"tt = 1 : {self.perc_treatement} % ")
        print(f"Y_0 = 1 : {self.y_0_perc} % ")
        print(f"Y_1 = 1 : {self.y_1_perc} % ")
        print(f"KLD = {KLD}")
        self.wd = wd
        self.kld = KLD
        data_sim.to_csv(self.path_data, index=False, header=True)
        return data_sim

