
from survcaus.utils import *


is_tcga = True
is_support = False
is_metabric = False

class DataLoader():
    def __init__(self, params_sim, params_survcaus):
        super().__init__()
        self.path = params_sim['path_data']
        self.pmf = True
        self.scheme_subd = "equidistant"  # "quantiles"  # "equidistant"  # "quantiles"
        self.num_durations = params_survcaus['num_durations']

    def load_data_sim_benchmark(self):
        """
        The load_data_sim_benchmark function loads the simulated data from the csv file and splits it into training, validation, and test sets.
        The function also transforms the labels to a more usable format for our purposes.

        :param self: Used to reference the object that is being created.
        :return: the following:.

        :doc-author: Trelent
        """
        if is_tcga:
            self.path = './final_data_1_simu'
        if is_support:
            self.path = './support_simu'
        if is_metabric:
            self.path = './metabric_simu'

        df = pd.read_csv(self.path + ".csv")
        dim = df.shape[1]-8

        x_z_list = ["X" + str(i) for i in range(1, dim + 1)] + ["tt"]
        leave = x_z_list + ["event", "T_f_cens"]

        ##
        rs = ShuffleSplit(test_size=.4, random_state=0)
        df_ = df[leave].copy()

        for train_index, test_index in rs.split(df_):
            df_train = df_.drop(test_index)
            df_test = df_.drop(train_index)
            df_val = df_test.sample(frac=0.2)
            df_test = df_test.drop(df_val.index)

        if self.pmf:
            labtrans = PMF.label_transform(
                self.num_durations, scheme=self.scheme_subd)

        def get_target(df):
            return (df["T_f_cens"].values, df["event"].values)

        y_train_surv = labtrans.fit_transform(*get_target(df_train))
        y_val_surv = labtrans.transform(*get_target(df_val))

        train = (df_train[x_z_list].values.astype("float32"), y_train_surv)
        val = (
            df_val[x_z_list].values.astype("float32"),
            y_val_surv,
        )
        # We don't need to transform the test labels
        durations_test, events_test = get_target(df_test)
        x_test = df_test[x_z_list].values.astype("float32")

        # SPlit data for OURS
        self.x_train, self.y_train, self.train, self.val,\
            self.durations_test, self.events_test, self.labtrans, self.x_test = train[
                0], train[1], train, val, durations_test, events_test, labtrans, x_test

        #  SPlit data for benchmarking

        def get_separ_data(x):
            """
            The get_separ_data function takes in a dataframe and returns two separate dataframes, one for the treatment group and one for the control group.

            :param x: Used to separate the data into two different sets of data.
            :return: a dataframe with the rows of x where tt = 1 and a dataframe with the rows of x where tt = 0.

            """
            mask_1 = x["tt"] == 1
            mask_0 = x["tt"] == 0
            x_1 = x[mask_1].drop(columns="tt")
            x_0 = x[mask_0].drop(columns="tt")
            return x_0, x_1

        df_train_0,  df_train_1 = get_separ_data(df_train)
        df_test_0, df_test_1 = get_separ_data(df_test)

        self.x_0_train = df_train_0.iloc[:, :-2].values
        self.e_0_train = df_train_0.iloc[:, -2].values

        self.x_1_train = df_train_1.iloc[:, :-2].values
        self.e_1_train = df_train_1.iloc[:, -2].values

        self.T_f_0_train = df_train_0.iloc[:, -1].values
        self.T_f_1_train = df_train_1.iloc[:, -1].values

        self.x_0_test = df_train_0.iloc[:, :-2].values
        self.e_0_test = df_train_0.iloc[:, -2].values

        self.x_1_test = df_test_1.iloc[:, :-2].values
        self.e_1_test = df_test_1.iloc[:, -2].values

        self.T_f_0_test = df_test_0.iloc[:, -1].values
        self.T_f_1_test = df_test_1.iloc[:, -1].values

        self.df_train = df_train
        self.df_test = df_test
        self.df_val = df_val

    def get_data(self):
        self.load_data_sim_benchmark()
        return self
