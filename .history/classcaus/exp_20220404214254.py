import warnings
from matplotlib.pyplot import cla, xlabel
from sklearn.manifold import TSNE
import seaborn as sns

from utils import *
from classes import *
from simulation import *
import streamlit as st
import numpy as np
import pickle as pkl
import sys
sys.path.append('../')

#import SessionState
# ignore warnings
warnings.filterwarnings("ignore")


param_sim = {
    'n_features': 25,
    'n_classes': 2,
    'n_samples': 1000,
    'wd_para': 2.,
    'beta': [0.1, 0.1, 0.3],
    'coef_tt': 2.8,
    'rho': 0.1,
    'path_data': './dataclassif.csv'
}

params_classifcaus = {
    "encoded_features": 25,
    "alpha_wass": 0.01,
    "batch_size": 128,
    "epochs": 30,
    "lr": 0.001,
    "patience": 10,
}

list_models = ["ClassCaus", "lgbm", "xgb"]

num_exp = 50
mode = "onemodel"
# class for Experiment: for evry wd_para simulate new data and luanch benchmark


class Experiment:
    def __init__(self, param_sim, params_classifcaus, list_wd_param, num_exp, list_models, mode="onemodel"):
        self.param_sim = param_sim
        self.params_classifcaus = params_classifcaus
        self.list_wd_param = list_wd_param
        self.num_exp = num_exp
        self.list_models = list_models
        self.sim = Simulation(param_sim)
        self.classifcaus = ClassifCaus(params_classifcaus)
        self.mode = mode

    def run(self):
        d_sim = {}
        for wd_para in self.list_wd_param:
            for exp in range(self.num_exp):
                self.param_sim['wd_para'] = wd_para
                self.sim.simule()

                # get param
                d_sim['wd_para'] = wd_para
                d_sim['wd'] = self.sim.wd
                d_sim['y_0_perc'] = self.sim.y_0_perc
                d_sim['y_1_perc'] = self.sim.y_1_perc

                # Bnechmark

                Bench = BenchmarkClassif(params_classifcaus, list_models)

                df_results, dic_fig, d_all = Bench.evall_all_bench(
                    mode=self.mode)
                df_results = df_results.drop(['f1_0', 'f1_1', 'mae'], axis=1)
                # rename columns
                df_results.columns.values[-1] = 'DTV_1'
                df_results.columns.values[-2] = 'DTV_0'
                # round values to 3 decimals
                df_results = df_results.round(3)
                # display results
                st.write(df_results)
                # save results as csv
                df_results.to_csv(f"results_bench_{exp}_{mode}.csv")

                fig_0 = boxplot_F(
                    list_models, d_all, ylabel='diff proba 0', option=0)
                fig_1 = boxplot_F(list_models, d_all,
                                  ylabel='diff proba 1', option=1)
                st.pyplot(fig_0)
                st.pyplot(fig_1)

                fig_dist_0_list = dist_F(
                    list_models, d_all, ylabel='', option=0)
                fig_dist_1_list = dist_F(
                    list_models, d_all, ylabel='', option=1)
