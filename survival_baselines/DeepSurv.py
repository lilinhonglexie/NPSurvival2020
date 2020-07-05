
import numpy as np
import pandas as pd

# For preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 

import torch # For building the networks 
import torchtuples as tt # Some useful functions

from pycox.models import CoxPH

# cox ph with non-linear interactions approximated by a neural network

class DeepSurv_pycox():
    def __init__(self, layers, nodes_per_layer, dropout, weight_decay, batch_size, 
                 lr=0.01, seed=47):
        # set seed
        np.random.seed(seed)
        _ = torch.manual_seed(seed)
        self.standardalizer = None
        self.standardize_data = True

        self._duration_col = "duration"
        self._event_col = "event"

        self.in_features = None
        self.out_features = 1
        self.batch_norm = True
        self.output_bias = False
        self.activation = torch.nn.ReLU
        self.epochs = 512
        self.num_workers = 2
        self.callbacks = [tt.callbacks.EarlyStopping()]

        # parameters tuned
        self.num_nodes = [int(nodes_per_layer) for _ in range(int(layers))]
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.batch_size = int(batch_size)

    def set_standardize(self, standardize_bool):
        self.standardize_data = standardize_bool

    def _format_to_pycox(self, X, Y, F):
        # from numpy to pandas df
        df = pd.DataFrame(data=X, columns=F)
        if Y is not None:
            df[self._duration_col] = Y[:, 0]
            df[self._event_col] = Y[:, 1]
        return df

    def _standardize_df(self, df, flag):
        # if flag = test, the df passed in does not contain Y labels
        if self.standardize_data:
            df_x = df if flag == 'test' else df.drop(columns=[self._duration_col, self._event_col])
            if flag == "train":
                cols_leave = []
                cols_standardize = []
                for column in df_x.columns:
                    if set(pd.unique(df[column])) == set([0,1]):
                        cols_leave.append(column)
                    else:
                        cols_standardize.append(column)
                standardize = [([col], StandardScaler()) for col in cols_standardize]
                leave = [(col, None) for col in cols_leave]
                self.standardalizer = DataFrameMapper(standardize + leave)

                x = self.standardalizer.fit_transform(df_x).astype('float32')
                y = (df[self._duration_col].values.astype('float32'), df[self._event_col].values.astype('float32'))

            elif flag == "val":
                x = self.standardalizer.transform(df_x).astype('float32')
                y = (df[self._duration_col].values.astype('float32'), df[self._event_col].values.astype('float32'))

            elif flag == "test":
                x = self.standardalizer.transform(df_x).astype('float32')
                y = None

            else:
                raise NotImplementedError

            return x,y
        else:
            raise NotImplementedError

    def fit(self, X, y, column_names):
        # format data
        self.column_names = column_names
        full_df = self._format_to_pycox(X, y, self.column_names)
        val_df = full_df.sample(frac=0.2)
        train_df = full_df.drop(val_df.index)
        train_x, train_y = self._standardize_df(train_df, "train")
        val_x, val_y = self._standardize_df(val_df, "val")
        # configure model 
        self.in_features = train_x.shape[1]
        net = tt.practical.MLPVanilla(in_features=self.in_features, num_nodes=self.num_nodes, 
            out_features=self.out_features, batch_norm=self.batch_norm, dropout=self.dropout,
            activation=self.activation, output_bias=self.output_bias)
        self.model = CoxPH(net, tt.optim.Adam(lr=self.lr, weight_decay=self.weight_decay))
        # self.model.optimizer.set_lr(self.lr)

        n_train = train_x.shape[0]
        while n_train % self.batch_size == 1: # this will cause issues in batch norm
            self.batch_size += 1

        self.model.fit(train_x, train_y, self.batch_size, self.epochs, self.callbacks, 
                       verbose=True, val_data=(val_x, val_y), val_batch_size=self.batch_size,
                       num_workers=self.num_workers)
        self.model.compute_baseline_hazards()

    def predict(self, test_x, time_list):
        # format data
        test_df = self._format_to_pycox(test_x, None, self.column_names)
        test_x, _ = self._standardize_df(test_df, "test")

        proba_matrix_ = self.model.predict_surv_df(test_x)
        proba_matrix = np.transpose(proba_matrix_.values)
        pred_medians = []
        median_time = max(time_list)
        # if the predicted proba never goes below 0.5, predict the largest seen value
        for test_idx, survival_proba in enumerate(proba_matrix):
            # the survival_proba is in descending order
            for col, proba in enumerate(survival_proba):
                if proba > 0.5:
                    continue
                if proba == 0.5 or col == 0:
                    median_time = time_list[col]
                else:
                    median_time = (time_list[col - 1] + time_list[col]) / 2
                break
            pred_medians.append(median_time)

        return np.array(pred_medians), proba_matrix_

