
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.tensorboard import SummaryWriter
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgbm
from sklearn import metrics
import torch.nn as nn
from torch import Tensor
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pandas as pd
# Importing the utils.py file from the class_files app.
from utils import *
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import classification_report
import torchtuples as tt
import warnings
warnings.filterwarnings("ignore")


# A class that defines the loss function.
class Loss(nn.Module):

    def __init__(self,  alpha, beta=1):
        super().__init__()
        self.beta = beta
        self.alpha = alpha
        self.loss_classif = nn.BCELoss()
        self.loss_wass = WassLoss()  # IPM

    def forward(self, phi_t, sigma, y_train):
        y_train = y_train.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        loss_classif = self.loss_classif(sigma, y_train)
        loss_wass = self.loss_wass(phi_t)  # Wasserstein Loss
        self.wd = loss_wass.item()
        self.cl = loss_classif.item()
        return self.beta*loss_classif + self.alpha * loss_wass  #


# It creates a class that inherits from nn.Module.
class WassLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, psi: Tensor) -> Tensor:
        a, b = sepr_repr(psi)
        self.psi0 = a
        self.psi1 = b
        return SinkhornDistance(eps=0.001, max_iter=100, reduction=None)(a, b)


# It creates a class that inherits from nn.Module.
class NetClassif(nn.Module):
    def __init__(self, in_features, encoded_features):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(in_features-1, 32),  nn.LeakyReLU(),
            nn.Linear(32, 32),  nn.ReLU(),
            nn.Linear(32, 28),  nn.LeakyReLU(),
            nn.Linear(28, encoded_features),
        )
        self.psi = nn.Sequential(
            nn.Linear(encoded_features + 1, 128), nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 50),  nn.ReLU(),
            nn.Linear(50, 1),
        )
        self.loss_classif = nn.BCELoss()
        self.loss_wass = WassLoss()  # IPM

    def forward(self, input):
        x, t = get_data(input)
        self.input = input
        t = t.reshape(-1, 1)
        phi = self.phi(x)
        phi_t = torch.cat((phi, t), 1)
        sigma = nn.Sigmoid()(self.psi(phi_t))
        return phi_t, sigma


# The class ClassifBase is a subclass of tt.Model
class ClassifBase(tt.Model):
    """Base class for classification models.
    Essentially same as torchtuples.Model,
    """

    def __init__(self, net, loss=None, optimizer=None, device=None):
        super().__init__(net, loss, optimizer, device)

    def predict_proba(self, input,  **kwargs):
        x, t = get_data(input)
        self.input = input
        t = t.reshape(-1, 1)
        phi = self.net.phi(x)
        phi_t = torch.cat((phi, t), 1)
        sigma = nn.Sigmoid()(self.net.psi(phi_t))
        return sigma.detach().cpu().numpy()


# The DataLoader class is used to load the data from the data_dir,
# process the data into batches and then return the batches.
class DataLoader():

    def __init__(self):
        super().__init__()
        self.path = "dataclassif"  # params_sim['path_data']

    def load_data_sim_benchmark(self):

        df = pd.read_csv(self.path + ".csv")
        df = reduce_mem_usage(df)
        self.df = df
        dim = df.shape[1]-9

        x_z_list = ["X" + str(i) for i in range(1, dim + 1)] + ["tt"]
        leave = x_z_list + ["Y_f", "Y_cf", "Y_0", "Y_1", "pi_0", "pi_1"]

        ##
        rs = ShuffleSplit(test_size=.4, random_state=0)
        df_ = df[leave].copy()

        for train_index, test_index in rs.split(df_):
            df_train = df_.drop(test_index)
            df_test = df_.drop(train_index)
            df_val = df_test.sample(frac=0.2)
            df_test = df_test.drop(df_val.index)

        counter_list = ["Y_cf", "Y_0", "Y_1", "pi_0", "pi_1"]
        y_train_classif = df_train['Y_f'].values.astype("float32")
        y_val_classif = df_val['Y_f'].values.astype("float32")
        y_test_classif = df_test['Y_f'].values.astype("float32")

        counter_train = df_train[counter_list].values.astype("float32")
        counter_val = df_val[counter_list].values.astype("float32")
        counter_test = df_test[counter_list].values.astype("float32")

        train = (df_train[x_z_list].values.astype("float32"), y_train_classif)
        val = (
            df_val[x_z_list].values.astype("float32"),
            y_val_classif,
        )

        x_test = df_test[x_z_list].values.astype("float32")

        # SPlit data for OURS
        self.x_train, self.y_train, self.train, self.val,\
            self.y_test, self.x_test, self.counter_train, self.counter_test, self.counter_val = \
            train[0], train[1], train, val, y_test_classif, x_test, counter_train, counter_test, counter_val

        self.x_train = torch.from_numpy(self.x_train).float()
        self.y_train = torch.from_numpy(self.y_train).float().view(-1, 1)
        self.x_val = torch.from_numpy(self.val[0]).float()
        self.y_val = torch.from_numpy(self.val[1]).float().view(-1, 1)
        self.x_test = torch.from_numpy(self.x_test).float()
        self.y_test = torch.from_numpy(self.y_test).float().view(-1, 1)

        #  SPlit data for benchmarking

        def get_separ_data(x):
            mask_1 = x["tt"] == 1
            mask_0 = x["tt"] == 0
            x_1 = x[mask_1].drop(columns="tt")
            x_0 = x[mask_0].drop(columns="tt")
            return x_0, x_1

        df_train_0,  df_train_1 = get_separ_data(df_train)
        df_val_0, df_val_1 = get_separ_data(df_val)
        x_z_list = ["X" + str(i) for i in range(1, dim + 1)]
        self.x_train_0 = torch.from_numpy(
            df_train_0[x_z_list].values.astype("float32")).float()
        self.x_train_1 = torch.from_numpy(
            df_train_1[x_z_list].values.astype("float32")).float()
        self.x_val_0 = torch.from_numpy(
            df_val_0[x_z_list].values.astype("float32")).float()
        self.x_val_1 = torch.from_numpy(
            df_val_1[x_z_list].values.astype("float32")).float()

        self.y_train_0 = torch.from_numpy(
            df_train_0["Y_f"].values.astype("float32")).float().view(-1, 1)
        self.y_train_1 = torch.from_numpy(
            df_train_1["Y_f"].values.astype("float32")).float().view(-1, 1)
        self.y_val_0 = torch.from_numpy(
            df_val_0["Y_f"].values.astype("float32")).float().view(-1, 1)
        self.y_val_1 = torch.from_numpy(
            df_val_1["Y_f"].values.astype("float32")).float().view(-1, 1)

        self.counter_train_0 = torch.from_numpy(
            df_train_0[counter_list].values.astype("float32")).float()
        self.counter_train_1 = torch.from_numpy(
            df_train_1[counter_list].values.astype("float32")).float()
        self.counter_val_0 = torch.from_numpy(
            df_val_0[counter_list].values.astype("float32")).float()
        self.counter_val_1 = torch.from_numpy(
            df_val_1[counter_list].values.astype("float32")).float()

    def get_data(self):
        self.load_data_sim_benchmark()
        return self


#
# The class ClassifCaus
class ClassifCaus(nn.Module):

    def __init__(self, params_classifcaus):
        super().__init__()
        encoded_features = params_classifcaus['encoded_features']
        alpha_wass = params_classifcaus['alpha_wass']
        batch_size = params_classifcaus['batch_size']
        epochs = params_classifcaus['epochs']
        lr = params_classifcaus['lr']
        patience = params_classifcaus['patience']

        self.data = DataLoader().get_data()

        self.in_features = self.data.x_train.shape[1]
        self.encoded_features = encoded_features
        self.net = NetClassif(self.in_features, self.encoded_features)

        self.alpha_wass = alpha_wass
        self.lr = lr
        self.loss = Loss(self.alpha_wass, 1)
        self.metrics = dict(loss_classif=Loss(0, 1), loss_wass=Loss(1, 0))
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = ClassifBase(
            net=self.net, loss=self.loss, optimizer=torch.optim.Adam, device=None)

        self.patence = patience
        self.callbacks = [tt.cb.EarlyStopping(patience=patience)]
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def fit_model(self):
        self.model.fit(input=self.data.x_train, target=self.data.y_train,
                       val_data=(self.data.x_val, self.data.y_val),
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       callbacks=self.callbacks,
                       metrics=self.metrics)
        return self

    def pred_t(self, x_test, t, y_test, p_test):
        # drop x_test[:,-1] and replace it by t * 1.
        tt = torch.ones(x_test.shape[0], 1) * t
        x_test_t = np.concatenate((x_test[:, :-1], tt.reshape(-1, 1)), axis=1)
        x_test_t = torch.from_numpy(x_test_t).float()
        y_test = y_test.squeeze().numpy()
        p_pred_t = self.model.predict_proba(x_test_t).squeeze()
        y_pred_t = (p_pred_t > 0.5) * 1.0
        acc = metrics.accuracy_score(y_test, y_pred_t)
        cf_m = metrics.confusion_matrix(y_test, y_pred_t)
        f1_s = metrics.f1_score(y_test, y_pred_t)
        auc = metrics.roc_auc_score(y_test, p_pred_t)
        kl = KL(p_test, p_pred_t)
        report = classification_report(y_test, y_pred_t)
        return acc, cf_m, f1_s, auc, kl, p_pred_t, report

    def eval_all_test(self, N):
        """
        It evaluates the model on the test set.
        :return: a dictionary with the accuracy, confusion matrix, f1-score, and AUC for class 0 and
        class 1.
        """
        d_exp = {}
        y_test_0 = torch.from_numpy(self.data.counter_test[:, 1]).float()
        y_test_1 = torch.from_numpy(self.data.counter_test[:, 2]).float()
        p_test_0 = torch.from_numpy(self.data.counter_test[:, 3]).float()
        p_test_1 = torch.from_numpy(self.data.counter_test[:, 4]).float()
        acc_0, cfm_m_0, f_1_0, auc_0, kl_0, p_pred_0, report_0 = self.pred_t(
            self.data.x_test, 0, y_test_0, p_test_0)
        acc_1, cfm_m_1, f_1_1, auc_1, kl_1, p_pred_1, report_1 = self.pred_t(
            self.data.x_test, 1, y_test_1, p_test_1)
        #
        p_pred_0 = np.asarray(p_pred_0, dtype=np.float)
        p_pred_1 = np.asarray(p_pred_1, dtype=np.float)
        p_test_0 = np.asarray(p_test_0, dtype=np.float)
        p_test_1 = np.asarray(p_test_1, dtype=np.float)
        y_pred_0 = (p_pred_0 > 0.5) * 1.0
        y_pred_1 = (p_pred_1 > 0.5) * 1.0
        cate_true = p_test_1 - p_test_0
        cate_pred = p_pred_1 - p_pred_0
        ord_cate_true = np.argsort(cate_true)
        ord_cate_pred = np.argsort(cate_pred)

        fig_log = plt.figure(figsize=(14, 10), dpi=120)
        log_cate = np.log(cate_true[ord_cate_true]
                          [:N]/cate_pred[ord_cate_true][:N])
        plt.plot(log_cate, label='log(cate_true/cate_pred)')
        plt.legend()
        plt.close(fig_log)

        fig_cate = plt.figure(figsize=(14, 10), dpi=120)
        ord_cate_true = np.argsort(cate_true)
        ord_cate_pred = np.argsort(cate_pred)
        diff_cate = cate_true - cate_pred
        ord_diff_cate = np.argsort(diff_cate)
        plt.plot(diff_cate[ord_diff_cate][:N], label='cate_true - cate_pred')

        # plt.plot(cate_true[ord_cate_true][:N], cate_pred[ord_cate_
        #plt.scatter(cate_pred[ord_cate_true][:N], marker='^',color='r',label='cate_pred')
        #plt.plot(diff_cate[ord_diff_cate], 'g', label='diff cate')

        #plt.plot(diff_cate[ord_diff_cate], 'g', label='diff cate')
        plt.plot(np.mean(cate_true[ord_cate_true][:N]) * np.ones(cate_true[:N].shape),
                 'k--', label='mean cate true')
        plt.plot(np.mean(cate_pred[ord_cate_true][:N]) * np.ones(cate_pred[:N].shape),
                 'g--', label='mean cate pred')

        plt.legend()
        plt.close(fig_cate)
        # roc curve
        fig_roc = plt.figure(figsize=(14, 10), dpi=120)
        fpr_1, tpr_1, thresholds = metrics.roc_curve(y_test_1, p_pred_1)
        fpr_0, tpr_0, thresholds = metrics.roc_curve(y_test_0, p_pred_0)
        plt.plot(fpr_1, tpr_1, label=f'tt = 1 with AUC = {auc_1.round(3)}')
        plt.plot(fpr_0, tpr_0, label=f'tt = 0 with AUC = {auc_0.round(3)}')
        # add title
        plt.title(
            f'ROC curve : KL_0 = {kl_0.round(3)} & KL_1 = {kl_1.round(3)}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        # save figure as svg
        plt.savefig('curv_roc.svg')
        plt.close(fig_roc)

        pehe = np.sqrt(metrics.mean_squared_error(cate_true, cate_pred))
        mae = metrics.mean_absolute_error(cate_true, cate_pred)
        d_0 = {'acc': acc_0, 'f1': f_1_0, 'auc': auc_0,
               'cf_m': cfm_m_0, 'pehe': pehe, 'mae': mae, 'kl': kl_0}
        d_1 = {'acc': acc_1, 'f1': f_1_1, 'auc': auc_1,
               'cf_m': cfm_m_1, 'pehe': pehe, 'mae': mae, 'kl': kl_1}

        print(f' Report for tt = 0 : \n {report_0}')
        print(f' Report for tt = 1 : \n {report_1}')

        d_exp = {'d_0': d_0, 'd_1': d_1, 'fig_roc': fig_roc, 'fig_cate': fig_cate,
                 'fig_log': fig_log, 'p_pred_0': p_pred_0, 'p_pred_1': p_pred_1, 'y_pred_0': y_pred_0, 'y_pred_1': y_pred_1,
                 'cate_true': cate_true, 'cate_pred': cate_pred, 'y_test_0': y_test_0, 'y_test_1': y_test_1,
                 'p_test_0': p_test_0, 'p_test_1': p_test_1}

        return d_0, d_1, fig_roc, fig_cate, fig_log, d_exp

    def eval(self, x_test, y_test):
        pred_proba = self.model.predict_proba(input=x_test).squeeze()
        y_pred = (pred_proba > 0.5) * 1.0
        y_test = y_test.squeeze().numpy()
        acc = metrics.accuracy_score(y_test, y_pred)
        cf_m = metrics.confusion_matrix(y_test, y_pred)
        f1_s = metrics.f1_score(y_test, y_pred)
        auc = metrics.roc_auc_score(y_test, pred_proba)

        print(
            f" accuracy: {acc}, confusion matrix: {cf_m}, f1 score: {f1_s}, auc: {auc}")
        self.model.log.to_pandas().to_csv("log_classif_caus.csv")
        # return acc, y_pred, pred_proba

    def eval_test(self):
        x_test, y_test = self.data.x_test, self.data.y_test
        return self.eval(x_test, y_test)


""" class for clasification benchmark
 we use models from sklearn, lgbm and xgboost for classification
 we train two modelson the same data x_0,y_0 and x_1,y_1
    we use the same data for training and testing
"""


class BenchmarkClassif():

    def __init__(self, params_classifcaus, list_models):
        """
        params_classifcaus : dict
        """
        self.params_classifcaus = params_classifcaus
        self.classifcaus = ClassifCaus(self.params_classifcaus)
        self.data = self.classifcaus.data
        self.list_models = list_models

    def fit_model(self, model_name):
        model_base_0, model_base_1 = None, None
        if model_name == "lgbm":
            model_base_0, model_base_1 = lgbm.LGBMClassifier(), lgbm.LGBMClassifier()
        elif model_name == "xgb":
            model_base_0, model_base_1 = xgb.XGBClassifier(), xgb.XGBClassifier()
        elif model_name == "rf":
            model_base_0, model_base_1 = RandomForestClassifier(), RandomForestClassifier()
        elif model_name == "svm":
            model_base_0, model_base_1 = SVC(), SVC()
        elif model_name == "knn":
            model_base_0, model_base_1 = KNeighborsClassifier(), KNeighborsClassifier()
        elif model_name == "mlp":
            model_base_0, model_base_1 = MLPClassifier(), MLPClassifier()
        elif model_name == "dt":
            model_base_0, model_base_1 = DecisionTreeClassifier(), DecisionTreeClassifier()
        elif model_name == "lgr":
            model_base_0, model_base_1 = LogisticRegression(), LogisticRegression()

        # copy
        class_weight_0 = compute_sample_weight('balanced', self.data.y_train_0)
        class_weight_1 = compute_sample_weight('balanced', self.data.y_train_1)

        model_base_0.fit(self.data.x_train_0, self.data.y_train_0,
                         sample_weight=class_weight_0)
        model_base_1.fit(self.data.x_train_1, self.data.y_train_1,
                         sample_weight=class_weight_1)

        return model_base_0, model_base_1

    def eval_model(self, model_name, N):
        y_test_0 = np.asarray(self.data.counter_test[:, 1], dtype=np.float)
        y_test_1 = np.asarray(self.data.counter_test[:, 2], dtype=np.float)
        p_test_0 = np.asarray(self.data.counter_test[:, 3], dtype=np.float)
        p_test_1 = np.asarray(self.data.counter_test[:, 4], dtype=np.float)

        d_exp = {}
        model_base_0, model_base_1 = self.fit_model(model_name)

        p_pred_0 = model_base_0.predict_proba(
            self.data.x_test[:, :-1])[:, 1]
        y_pred_0 = (p_test_0 > 0.5) * 1.0
        p_pred_1 = model_base_1.predict_proba(
            self.data.x_test[:, :-1])[:, 1]
        y_pred_1 = (p_test_1 > 0.5) * 1.0

        p_pred_0 = np.asarray(p_pred_0, dtype=np.float)
        p_pred_1 = np.asarray(p_pred_1, dtype=np.float)
        cate_true = p_test_1 - p_test_0
        cate_pred = p_pred_1 - p_pred_0
        ord_cate_true = np.argsort(cate_true)
        ord_cate_pred = np.argsort(cate_pred)

        fig_log = plt.figure(figsize=(14, 10), dpi=120)
        log_cate = np.log(cate_true[ord_cate_true]
                          [:N]/cate_pred[ord_cate_true][:N])
        plt.plot(log_cate, label='log(cate_true/cate_pred)')
        plt.legend()
        plt.close(fig_log)

        fig_cate = plt.figure(figsize=(14, 10), dpi=120)

        diff_cate = cate_true - cate_pred
        ord_diff_cate = np.argsort(diff_cate)
        plt.plot(diff_cate[ord_diff_cate][:N], label='cate_true - cate_pred')
        plt.plot(np.mean(cate_true[ord_cate_true][:N]) * np.ones(cate_true[:N].shape),
                 'k--', label='mean cate true')
        plt.plot(np.mean(cate_pred[ord_cate_true][:N]) * np.ones(cate_pred[:N].shape),
                 'g--', label='mean cate pred')
        # plt.plot(cate_true[ord_cate_true][:N],marker='o',color='b',label='cate_true')
        #plt.plot(cate_pred[ord_cate_true][:N], marker='^',color='r',label='cate_pred')
        #plt.plot(diff_cate[ord_diff_cate], 'g', label='diff cate')

        plt.legend()
        plt.close(fig_cate)

        pehe = np.sqrt(metrics.mean_squared_error(cate_true, cate_pred))
        mae = metrics.mean_absolute_error(cate_true, cate_pred)
        acc_0 = metrics.accuracy_score(y_test_0, y_pred_0)
        acc_1 = metrics.accuracy_score(y_test_1, y_pred_1)
        f_1_0 = metrics.f1_score(y_test_0, y_pred_0)
        f_1_1 = metrics.f1_score(y_test_1, y_pred_1)
        cfm_m_0 = metrics.confusion_matrix(y_test_0, y_pred_0)
        cfm_m_1 = metrics.confusion_matrix(y_test_1, y_pred_1)
        auc_0 = metrics.roc_auc_score(y_test_0, p_pred_0)
        auc_1 = metrics.roc_auc_score(y_test_1, p_pred_1)
        kl_0 = KL(p_test_0, p_pred_0)
        kl_1 = KL(p_test_1, p_pred_1)
        d_0 = {'acc': acc_0, 'f1': f_1_0, 'auc': auc_0,
               'cf_m': cfm_m_0, 'pehe': pehe, 'mae': mae, 'kl': kl_0}
        d_1 = {'acc': acc_1, 'f1': f_1_1, 'auc': auc_1,
               'cf_m': cfm_m_1, 'pehe': pehe, 'mae': mae, 'kl': kl_1}
        print(f' Report for tt = 0 : \n {d_0}')
        print(f' Report for tt = 1 : \n {d_1}')
        # roc curve
        fig_roc = plt.figure(figsize=(14, 10), dpi=120)
        fpr_1, tpr_1, thresholds = metrics.roc_curve(y_test_1, p_pred_1)
        fpr_0, tpr_0, thresholds = metrics.roc_curve(y_test_0, p_pred_0)
        plt.plot(fpr_1, tpr_1, label=f'tt = 1 with AUC = {auc_1.round(3)}')
        plt.plot(fpr_0, tpr_0, label=f'tt = 0 with AUC = {auc_0.round(3)}')
        plt.title(
            f'ROC curve : KL_0 = {kl_0.round(3)} & KL_1 = {kl_1.round(3)}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.close(fig_roc)

        d_exp = {'d_0': d_0, 'd_1': d_1, 'fig_roc': fig_roc, 'fig_cate': fig_cate,
                 'fig_log': fig_log, 'p_pred_0': p_pred_0, 'p_pred_1': p_pred_1, 'y_pred_0': y_pred_0, 'y_pred_1': y_pred_1,
                 'cate_true': cate_true, 'cate_pred': cate_pred, 'y_test_0': y_test_0, 'y_test_1': y_test_1,
                 'p_test_0': p_test_0, 'p_test_1': p_test_1}

        return d_0, d_1, fig_roc, fig_cate, fig_log, d_exp

    def evall_all_bench(self, N=400, mode='two_models'):
        results = {}
        dic_fig = {}
        d_all = {}
        for model_name in self.list_models:
            print(f'Evaluation for model {model_name}')

            if model_name == "ClassCaus":

                self.classifcaus.fit_model()
                self.res = self.classifcaus.model.log.to_pandas()
                d_0, d_1, fig_roc, fig_cate, fig_log, d_exp = self.classifcaus.eval_all_test(
                    N)

            else:
                if mode == 'two_models':
                    d_0, d_1, fig_roc, fig_cate, fig_log, d_exp = self.eval_model(
                        model_name, N)
                else:
                    d_0, d_1, fig_roc, fig_cate, fig_log, d_exp = self.eval_model_one(
                        model_name, N)

            dic_fig[model_name] = [fig_roc, fig_cate, fig_log]

            results[model_name] = {'acc_0': d_0['acc'], 'acc_1': d_1['acc'], 'f1_0': d_0['f1'], 'f1_1': d_1['f1'],
                                   'auc_0': d_0['auc'], 'auc_1': d_1['auc'],  'pehe': d_0['pehe'], 'mae': d_0['mae'],
                                   'kl_0': d_0['kl'], 'kl_1': d_1['kl']}  # 'cfm_m_0': d_0['cf_m'],'cfm_m_1': d_1['cf_m'],
            print(f'Evaluation for model {model_name} done')
            d_all[model_name] = d_exp

        # to dataframe
        df_results = pd.DataFrame(results).transpose()
        df_results.to_csv(f'results_bench.csv')
        return df_results, dic_fig, d_all

    def fit_model_one(self, model_name):

        model_base = None
        if model_name == "lgbm":
            model_base = lgbm.LGBMClassifier()
        elif model_name == "xgb":
            model_base = xgb.XGBClassifier()
        elif model_name == "rf":
            model_base = RandomForestClassifier()
        elif model_name == "svm":
            model_base = SVC()
        elif model_name == "knn":
            model_base = KNeighborsClassifier()
        elif model_name == "mlp":
            model_base = MLPClassifier()
        elif model_name == "dt":
            model_base = DecisionTreeClassifier()
        elif model_name == "lgr":
            model_base = LogisticRegression()

        class_weight = compute_sample_weight('balanced', self.data.y_train)

        model_base.fit(self.data.x_train,
                       self.data.y_train.view(-1), sample_weight=class_weight)
        return model_base

    def eval_model_one(self, model_name, N):
        y_test_0 = np.asarray(self.data.counter_test[:, 1], dtype=np.float)
        y_test_1 = np.asarray(self.data.counter_test[:, 2], dtype=np.float)
        p_test_0 = np.asarray(self.data.counter_test[:, 3], dtype=np.float)
        p_test_1 = np.asarray(self.data.counter_test[:, 4], dtype=np.float)
        d_exp = {}
        model_base = self.fit_model_one(model_name)

        x_test_tt_0 = get_x_tt(self.data.x_test, 0)
        x_test_tt_1 = get_x_tt(self.data.x_test, 1)
        p_pred_0 = model_base.predict_proba(x_test_tt_0)[:, 1]
        p_pred_0 = np.asarray(p_pred_0, dtype=np.float)
        y_pred_0 = (p_test_0 > 0.5) * 1.0
        p_pred_1 = model_base.predict_proba(x_test_tt_1)[:, 1]

        p_pred_1 = np.asarray(p_pred_1, dtype=np.float)

        y_pred_1 = (p_test_1 > 0.5) * 1.0
        cate_true = p_test_1 - p_test_0
        cate_pred = p_pred_1 - p_pred_0
        ord_cate_true = np.argsort(cate_true)
        ord_cate_pred = np.argsort(cate_pred)

        fig_log = plt.figure(figsize=(14, 10), dpi=120)
        log_cate = np.log(cate_true[ord_cate_true]
                          [:N]/cate_pred[ord_cate_true][:N])
        plt.plot(log_cate, label='log(cate_true/cate_pred)')
        plt.legend()
        plt.close(fig_log)

        fig_cate = plt.figure(figsize=(14, 10), dpi=120)

        diff_cate = cate_true - cate_pred
        ord_diff_cate = np.argsort(diff_cate)
        plt.plot(diff_cate[ord_diff_cate][:N], label='cate_true - cate_pred')
        plt.plot(np.mean(cate_true[ord_cate_true][:N]) * np.ones(cate_true[:N].shape),
                 'k--', label='mean cate true')
        plt.plot(np.mean(cate_pred[ord_cate_true][:N]) * np.ones(cate_pred[:N].shape),
                 'g--', label='mean cate pred')

        plt.legend()
        plt.close(fig_cate)

        pehe = np.sqrt(metrics.mean_squared_error(cate_true, cate_pred))
        mae = metrics.mean_absolute_error(cate_true, cate_pred)
        acc_0 = metrics.accuracy_score(y_test_0, y_pred_0)
        acc_1 = metrics.accuracy_score(y_test_1, y_pred_1)
        f_1_0 = metrics.f1_score(y_test_0, y_pred_0)
        f_1_1 = metrics.f1_score(y_test_1, y_pred_1)
        cfm_m_0 = metrics.confusion_matrix(y_test_0, y_pred_0)
        cfm_m_1 = metrics.confusion_matrix(y_test_1, y_pred_1)
        auc_0 = metrics.roc_auc_score(y_test_0, p_pred_0)
        auc_1 = metrics.roc_auc_score(y_test_1, p_pred_1)
        kl_0 = KL(p_test_0, p_pred_0)
        kl_1 = KL(p_test_1, p_pred_1)
        d_0 = {'acc': acc_0, 'f1': f_1_0, 'auc': auc_0,
               'cf_m': cfm_m_0, 'pehe': pehe, 'mae': mae, 'kl': kl_0}
        d_1 = {'acc': acc_1, 'f1': f_1_1, 'auc': auc_1,
               'cf_m': cfm_m_1, 'pehe': pehe, 'mae': mae, 'kl': kl_1}
        print(f' Report for tt = 0 : \n {d_0}')
        print(f' Report for tt = 1 : \n {d_1}')
        # roc curve
        fig_roc = plt.figure(figsize=(14, 10), dpi=120)
        fpr_1, tpr_1, thresholds = metrics.roc_curve(y_test_1, p_pred_1)
        fpr_0, tpr_0, thresholds = metrics.roc_curve(y_test_0, p_pred_0)
        plt.plot(fpr_1, tpr_1, label=f'tt = 1 with AUC = {auc_1.round(3)}')
        plt.plot(fpr_0, tpr_0, label=f'tt = 0 with AUC = {auc_0.round(3)}')
        plt.title(
            f'ROC curve : KL_0 = {kl_0.round(3)} & KL_1 = {kl_1.round(3)}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.close(fig_roc)

        d_exp = {'d_0': d_0, 'd_1': d_1, 'fig_roc': fig_roc, 'fig_cate': fig_cate,
                 'fig_log': fig_log, 'p_pred_0': p_pred_0, 'p_pred_1': p_pred_1, 'y_pred_0': y_pred_0, 'y_pred_1': y_pred_1,
                 'cate_true': cate_true, 'cate_pred': cate_pred, 'y_test_0': y_test_0, 'y_test_1': y_test_1,
                 'p_test_0': p_test_0, 'p_test_1': p_test_1}

        return d_0, d_1, fig_roc, fig_cate, fig_log, d_exp


import livejson

def save_json(Exp_num, wd_param, wd, model_name, pehe, dt0,dt1):
    d = livejson.File("test.json")
    if d == None:
        l_exp = [{
                f'{model_name}':{
                    'pehe': pehe,
                    'dt0': dt0,
                    'dt1': dt1,
                    'wd': wd
                }
            }]
        d[f'wd_{wd_param}']={
            f'{Exp_num}': l_exp
        }
    else:
        l_exp = d[f'wd_{wd_param}'][f'{Exp_num}']
        l_exp.append({
            f'{model_name}':{
                'pehe': pehe,
                'dt0': dt0,
                'dt1': dt1,
                'wd': wd
            }
        })
        d[f'wd_{wd_param}'][f'{Exp_num}'] = l_exp
    
   
            


Exp_num = 'Exp_2'
wd_param = 0.1
wd = 0.1
model_name = 'model_2'
pehe = 0.1
dt0 = 0.1
dt1 = 0.1
save_json(Exp_num, wd_param, wd, model_name, pehe, dt0,dt1)


   