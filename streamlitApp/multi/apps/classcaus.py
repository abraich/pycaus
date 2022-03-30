import warnings
from matplotlib.pyplot import xlabel
from sklearn.manifold import TSNE
import seaborn as sns

from apps.class_files.utils import *
from apps.class_files.classes import *
from apps.class_files.simulation import *
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


def app():
    st.title("ClassCaus")
    st.write("ClassCaus is a python package for binary causal classification ")
    tasks_choices = ['Simulation',  'Benchmarking', 'WDvsDTV']
    task = st.sidebar.selectbox("Choose a task", tasks_choices)
    if task == 'Simulation':
        n_samples = st.sidebar.number_input(
            "n_samples", min_value=2000, max_value=10000)
        n_features = st.sidebar.number_input(
            "n_features", min_value=2, max_value=30, value=25)
        coef_tt = st.sidebar.number_input("coef_tt", value=2.8)
        wd_param = st.sidebar.number_input("wd_param", value=1.)

        rho = st.sidebar.number_input("rho", value=0.1)
        param_sim['rho'] = rho
        param_sim['n_samples'] = n_samples
        param_sim['n_features'] = n_features
        param_sim['coef_tt'] = coef_tt
        param_sim['wd_para'] = wd_param

        idx = np.arange(param_sim['n_features'])
        param_sim['beta'] = (-1) ** idx * np.exp(-idx / 20.)
        #beta_norm = np.linalg.norm(param_sim['beta'])
        #param_sim['beta'] = param_sim['beta'] / beta_norm
        if st.sidebar.button("Run"):
            sim = Simulation(param_sim)
            sim.simule()

            st.write({"Shape": sim.data_sim.shape,
                      'WD': sim.wd, "% treatement": sim.perc_treatement})
            # print head datasim
            df = sim.data_sim
            st.write(df.head(5))
            st.write({'Y_0 mean': df['Y_0'].mean(),
                      'Y_1 mean': df['Y_1'].mean()})
            st.write(df.describe())
            # matrix of correlation
            st.write("Matrix of correlation")
            st.write(df.iloc[:, :param_sim['n_features']].corr())
            # print path
            st.write(f"Data saved in {sim.path_data}")
            # TSNE
            """st.write("TSNE")
            x = df.iloc[:, :param_sim['n_features']]
            tsne = TSNE(n_components=2, verbose=1, random_state=123)
            z = tsne.fit_transform(x)
            d = pd.DataFrame()
            d["tt"] = df[['tt']].values.squeeze()
            d["comp-1"] = z[:, 0]
            d["comp-2"] = z[:, 1]

            fig = plt.figure()
            sns.scatterplot(x="comp-1", y="comp-2", hue=d.tt.tolist(),
                            palette=sns.color_palette("hls", 2),
                            data=d).set(title="Sampled data T-SNE projection")
            fig.show()
            st.pyplot(fig)"""

    if task == 'Benchmarking':
        list_models = st.sidebar.multiselect('List of models', [
            "ClassCaus", "lgbm", "xgb", "rf", "mlp", "dt"], default=["ClassCaus","lgbm", "xgb"])

        params_classifcaus = {
            "encoded_features": 25,
            "alpha_wass": 0.01,
            "batch_size": 128,
            "epochs": 100,
            "lr": 0.001,
            "patience": 10,
        }
        encoded_features = st.sidebar.number_input(
            "encoded_features", value=25)
        alpha_wass = st.sidebar.number_input("alpha_wass", value=0.01)
        batch_size = st.sidebar.number_input("batch_size", value=128)
        epochs = st.sidebar.number_input("epochs", value=30)
        patience = st.sidebar.number_input("patience", value=7)
        N = st.sidebar.number_input("N_test", value=200)
        mode = st.sidebar.selectbox("Mode", ["two_models", "one_model"])
        num_experiments = st.sidebar.number_input("num_experiments", value=1)
        params_classifcaus['encoded_features'] = encoded_features
        params_classifcaus['alpha_wass'] = alpha_wass
        params_classifcaus['batch_size'] = batch_size
        params_classifcaus['epochs'] = epochs
        params_classifcaus['patience'] = patience

        if st.sidebar.button("Run"):
            for exp in range(num_experiments):
                Bench = BenchmarkClassif(params_classifcaus, list_models)
                # print shape datasim
                st.write(f'Shape of data: {Bench.data.df.shape}')
                # Start training
                st.write("Start training")
                # progress bar
                progress = st.progress(0)
                df_results, dic_fig, d_all = Bench.evall_all_bench(N, mode)
                progress.progress(100)
                # end training
                st.write("End training")
                st.write("Losses")
                st.write(Bench.res)
                # plots
                st.write("Plots")
                st.line_chart(Bench.res[['train_loss', 'val_loss']])

                # display results : big and centered
                st.write("#Results")
                # delete f_1_1 and f_1_0 and mae columns
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

                # plot in 2 columns
                for i, fig in enumerate(fig_dist_0_list):
                    st.pyplot(fig)
                    st.pyplot(fig_dist_1_list[i])

                # save results and figures , dic_fig , d_all
                path_save = './results/'
                pkl.dump(dic_fig,  open(path_save + 'dic_fig.pkl', 'wb'))
                pkl.dump(d_all,  open(path_save + 'd_all.pkl', 'wb'))
                pkl.dump(df_results, open(path_save + 'df_results.pkl', 'wb'))
                st.write(f"Results saved in {path_save}")
                # st.json(d_all)
                row1_col1, row1_col2 = st.columns(2)
                for key, fig in dic_fig.items():
                    # big markdown
                    row1_col1.markdown(f"# AUC ROC - {key}")
                    row1_col2.markdown(f"# CATE - {key}")
                    row1_col1.pyplot(fig[0])
                    row1_col2.pyplot(fig[1])
                    # add label
                # st.write(d_all)


    if task == 'WDvsDTV':
        n_samples = st.sidebar.number_input(
            "n_samples", min_value=2000, max_value=10000)
        n_features = st.sidebar.number_input(
            "n_features", min_value=2, max_value=30, value=25)
        coef_tt = st.sidebar.number_input("coef_tt", value=2.8)
        wd_param = st.sidebar.number_input("wd_param", value=1.)

        params_classifcaus = {
                "encoded_features": 25,
                "alpha_wass": 0.01,
                "batch_size": 128,
                "epochs": 100,
                "lr": 0.001,
                "patience": 10,
            }
        rho = st.sidebar.number_input("rho", value=0.1)
        param_sim['rho'] = rho
        param_sim['n_samples'] = n_samples
        param_sim['n_features'] = n_features
        param_sim['coef_tt'] = coef_tt

        idx = np.arange(param_sim['n_features'])
        param_sim['beta'] = (-1) ** idx * np.exp(-idx / 20.)
        
        # 
        encoded_features = st.sidebar.number_input(
                "encoded_features", value=25)
        alpha_wass = st.sidebar.number_input("alpha_wass", value=0.01)
        batch_size = st.sidebar.number_input("batch_size", value=128)
        epochs = st.sidebar.number_input("epochs", value=30)
        patience = st.sidebar.number_input("patience", value=7)
        N = st.sidebar.number_input("N_test", value=200)
        mode = st.sidebar.selectbox("Mode", ["one_model","two_models"])
        #num_experiments = st.sidebar.number_input("num_experiments", value=1)
        # list input
        list_wd_param = np.linspace(0., 3., 10)
        d = {}
        d['wd_list'] = []
        d['df_wd'] = []
        if st.sidebar.button("Run"):
            for wd_param in list_wd_param:
                    param_sim['wd_para'] = wd_param
                    sim = Simulation(param_sim)
                    sim.simule()
                    d['wd_list'].append(sim.wd)
                    st.write('Simulation done')
                    list_models = ["ClassCaus", "lgbm", "xgb"]
                    #st.sidebar.multiselect('List of models', [
                    #"ClassCaus", "lgbm", "xgb", "rf", "mlp", "dt"], default=["ClassCaus"])
                    params_classifcaus['encoded_features'] = encoded_features
                    params_classifcaus['alpha_wass'] = alpha_wass
                    params_classifcaus['batch_size'] = batch_size
                    params_classifcaus['epochs'] = epochs
                    params_classifcaus['patience'] = patience


                    Bench = BenchmarkClassif(params_classifcaus, list_models)
                    
                    st.write(f"Start training for wd_param = {wd_param}")
                    progress=  st.progress(0)
                    df_results, dic_fig, d_all = Bench.evall_all_bench(N, mode)
                    progress.progress(100)
                    # end training
                    st.write(f"End training for wd_param = {wd_param}")
                    
                    df_wd = get_dtv(list_models, d_all)
                    # save as csv
                    df_wd.to_csv(f"results_bench_wd_{wd_param}_{mode}.csv")
                    
                    
                    d['df_wd'].append(df_wd)
                    
                    st.write(df_wd)
            st.write(d['wd_list'] )