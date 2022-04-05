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


# class for Experiment: for evry wd_para simulate new data and luanch benchmark

class Experiment:
    def __init__(self, param_sim, params_classifcaus):
        self.param_sim = param_sim
        self.params_classifcaus = params_classifcaus
        self.sim = Simulation(param_sim)
        self.classifcaus = ClassifCaus(params_classifcaus)
        self.classifcaus.load_data(self.sim.data)
        self.classifcaus.build_model()
        self.classifcaus.train_model()
        self.classifcaus.save_model()
        self.classifcaus.load_model()
        self.classifcaus.predict_model()
        self.classifcaus.save_prediction()
        self.classifcaus.load_prediction()
        self.classifcaus.plot_prediction()

    def run(self):
        self.sim.run()
        self.classifcaus.run()
    
    def save(self):
        self.sim.save()
        self.classifcaus.save()
    def load(self):
        self.sim.load()
        self.classifcaus.load()
    def plot(self):
        self.sim.plot()
        self.classifcaus.plot()
    def plot_prediction(self):
        self.classifcaus.plot_prediction()
    