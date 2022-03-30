

"""
# class for Neptune logs
class Neptune: 
    def __init__(self, experiment_name):
        self.project ="SurvCaus/RUNS"
        self.api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NTllZGVjNy1jMWVmLTRjNzktYTIyNi0yM2JiNjIwZDkyZjgifQ=="
        self.experiment_name = experiment_name
        self.experiment = None
        
        self.num_runs = 0
        self.list_runs = []
        self.p_survcaus_best = None
        self.p_bart_best = None
        
        self.p_survcaus_best_list = []
        self.p_bart_best_list = []
        
        self.data_sim = None
        self.list_models =  ["SurvCaus", "SurvCaus_0", 'CoxPH', 'BART']
        self.list_EV = []

        
    def create_experiment(self):
        # create experiment
        self.experiment = neptune.init(project=self.project, api_token=self.api_token)
        # increase number of runs
        self.num_runs += 1
        # add run to list
        self.list_runs.append(self.experiment)
        return self.experiment
    
    def set_p_survcaus_best(self,p_survcaus_best):
        self.p_survcaus_best = p_survcaus_best
        self.p_survcaus_best_list.append(p_survcaus_best)
        
    def set_p_bart_best(self,p_bart_best):
        self.p_bart_best = p_bart_best
        self.p_bart_best_list.append(p_bart_best)
        
        
    def send_data(self, df, name,num_run):
        df.to_csv("./data_exp/"+name+"_"+str(num_run)+".csv")
        self.experiment["data_exp/"+name+"_"+str(num_run)].upload(File("./data_exp/"+name+"_"+str(num_run)+".csv"))
        
        
    def send_param(self, param, name,num_run):
        param.to_csv("./param_exp/"+name+"_"+str(num_run)+".csv")
        self.experiment["param_exp/"+name+"_"+str(num_run)].upload(File("./param_exp/"+name+"_"+str(num_run)+".csv"))
        
    def send_plot(self, fig, name,num_run):
        fig.savefig("./plot_exp/"+name+"_"+str(num_run)+".png")
        self.experiment["plot_exp/"+name+"_"+str(num_run)].upload(File("./plot_exp/"+name+"_"+str(num_run)+".png"))
        
        
    # 'Simulation'
    def run_simulation(self, p_sim):
        simu = SimulationNew(p_sim)
        data = simu.simulation_surv()
        self.data_sim = data
        self.p_sim = p_sim
        # TSNE
        x = data.iloc[:, :p_sim['n_features']]
        tsne = TSNE(n_components=2, verbose=1, random_state=123)
        z = tsne.fit_transform(x)
        d = pd.DataFrame()
        d["tt"] = data[['tt']].values.squeeze()
        d["comp-1"] = z[:, 0]
        d["comp-2"] = z[:, 1]

        fig = plt.figure()
        sns.scatterplot(x="comp-1", y="comp-2", hue=d.tt.tolist(),
                    palette=sns.color_palette("hls", 2),
                    data=d).set(title="Sampled data T-SNE projection")
        plt.close()
        # send to neptune
        self.send_plot(fig, "TSNE",self.num_runs)
        self.send_data(data, "data_sim",self.num_runs)
        
        surv_true = simu.plot_surv_true(patient=0)
        self.send_plot(surv_true, "surv_true",self.num_runs)
        
        return data
    
    def get_simulation_data(self):
        return self.data_sim
    
    
    #Tunning Survcaus 

    def run_tunning_survcaus(self,n_trials):
        self.tunning = Tunning(self.p_sim)
        p_survcaus_best = self.tunning.get_best_hyperparameter_survcaus(n_trials=n_trials)
        self.set_p_survcaus_best(p_survcaus_best)
        return p_survcaus_best
    def run_tunning_bart(self,n_trials):
        p_bart_best = self.tunning.get_best_hyperparameter_bart(n_trials=n_trials)
        self.set_p_bart_best(p_bart_best)
        return p_bart_best
    
    def get_evaluation(self):
        self.Ev = Evaluation(self.p_sim, self.p_survcaus_best)
        self.list_EV.append(self.Ev)
        return self.Ev
    def launch_benchmark(self):
        Ev = self.get_evaluation()
        Ev.All_Results(list_models=self.list_models,
                        is_train=False,params_bart=self.p_bart_best)
        
        bilan_csv = Ev.bilan_benchmark
        bilan_csv.to_csv("./bilan_exp/"+self.experiment_name+"_"+str(self.num_runs)+".csv")
        
        self.experiment["bilan_exp/"+self.experiment_name+"_"+str(self.num_runs)].upload(File("./bilan_exp/"+self.experiment_name+"_"+str(self.num_runs)+".csv"))
        # box plots
        box_plot_cate = Ev.box_plot_cate
        box_plot_cate.savefig("./box_plot_exp/"+self.experiment_name+"_"+str(self.num_runs)+".png")
        self.experiment["box_plot_exp/"+self.experiment_name+"_"+str(self.num_runs)].upload(File("./box_plot_exp/"+self.experiment_name+"_"+str(self.num_runs)+".png"))
        
       """ 