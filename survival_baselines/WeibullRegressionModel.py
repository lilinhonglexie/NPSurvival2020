from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 

import rpy2.robjects as robjects
from rpy2.robjects import FloatVector
from rpy2.robjects.packages import importr

import pandas as pd
import numpy as np

'''
weibull: Weibull regression by R, no hyperparams for tuning
weibull_pca: Weibull regression with pca dimensionality reduction, params = {n_components(for pca)}

default:
    - feature standardization

implemented by George Chen & Ren Zuo, last modified by Lexie Li @ 2019.12
'''
def compute_proba_from_quantiles(quantiles, time_list):
    probabilities = np.linspace(0.999, 0.001, num=999)
    result_probas = np.zeros(len(time_list))

    q_i = 0
    for t_i, t in enumerate(time_list):

        while True:
            if q_i >= 999:
                result_probas[t_i] = 0
                break

            curr_qt= quantiles[q_i]
            curr_qp = probabilities[q_i]

            if t == curr_qt:
                result_probas[t_i] = curr_qp
                break

            elif t < curr_qt:
                last_qp = probabilities[max(0,q_i - 1)]
                result_probas[t_i] = (last_qp + curr_qp)/2
                break 

            q_i += 1

    return result_probas

class WeibullRegressionModel:

    def __init__(self):
        """
        This is a python wrapper of R's Weibull Regression Model (in Survival
        package). rpy2 is used to call R's functions.

        """
        # prepare R's packages
        self.survival = importr("survival")
        self.base = importr('base')

        self.standardalizer = StandardScaler()
        self.standardize_data = True

    def set_standardize(self, standardize_bool):
        self.standardize_data = standardize_bool

    def _standardize_df(self, df, flag):
        # if flag = test, the df passed in does not contain Y labels
        if self.standardize_data:
            df_x = df.drop(columns=[self._duration_col, self._event_col]) if flag == 'train' else df
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
                standardized_values = self.standardalizer.fit_transform(df_x)
                standardized_df = pd.DataFrame(data=standardized_values).set_index(df_x.index)
                standardized_df.columns = df_x.columns
                standardized_df[self._duration_col] = df[self._duration_col]
                standardized_df[self._event_col] = df[self._event_col]
            else:
                standardized_values = self.standardalizer.transform(df_x)
                standardized_df = pd.DataFrame(data=standardized_values).set_index(df_x.index)
                standardized_df.columns = df_x.columns
            return standardized_df
        else:
            return df

    def fit(self, train_x, train_y, feature_names, duration_col='LOS', event_col='OUT'):
        """
        As we only need to run a couple of lines in R, here we directly generate
        R command as python string, and call R using rpy2. There should be a
        more elegant way of using rpy2.

        :param train_df: DataFrame, with the duration and the event column
        :param duration_col: the column name for duration
        :param event_col: the column name for event
        """
        train_df = pd.DataFrame(data=train_x, columns=feature_names)
        train_df[duration_col] = train_y[:, 0]
        train_df[event_col] = train_y[:, 1]

        self._feature_names = feature_names
        self._duration_col = duration_col
        self._event_col = event_col

        reduced_train_df = self._standardize_df(train_df, flag='train')
        self.column_ref = {}

        # Generate R command.
        # Basically we first generate a vector for each feature we have:
        # C.i <- c(1,0,2,1,...)
        # Then we combine those vectors into a dataframe in R
        # train.df <- data.frame(C.1, C.2, ..., OUT, LOS)
        # Then we fit the model
        # s <- Surv(train.df$LOS, train.df$OUT)
        # survregWeibull <- survreg(s ~ C1 + C2 + ..., train.df, dist="weibull")
        df_command = 'train.df <- data.frame('
        sr_command = 'survregWeibull <- survreg(s ~ '
        columns = \
            reduced_train_df.drop(columns=[duration_col, event_col]).columns
        for index, column in enumerate(columns):
            # we don't actually care about the column name in R
            r_var_name = "C." + str(index)
            self.column_ref[column] = r_var_name
            # generate a R vector for each feature
            robjects.globalenv[r_var_name] = \
                FloatVector(list(reduced_train_df[column]))
            df_command += r_var_name + ', '
            sr_command += r_var_name
            if index != len(columns) - 1:
                sr_command += ' + '
        # generate a R vector for LOS
        robjects.globalenv[duration_col] = \
            FloatVector(list(reduced_train_df[duration_col]))
        # generate a R vector for OUT
        robjects.globalenv[event_col] = \
            FloatVector(list(reduced_train_df[event_col]))
        df_command += duration_col + ', ' + event_col + ')'
        sr_command += ', train.df, dist = "weibull")'

        robjects.r(df_command)
        robjects.r('s <- Surv(train.df$' + duration_col + ', train.df$' +
                   event_col + ')')
        robjects.r(sr_command)

    def predict(self, test_x, time_list):
        """
        :param test_df: DataFrame
        :param time_list: checkpoint time to calculate probability on
        :return: the list of median survival time and the probability matrix
        """
        test_df = pd.DataFrame(data=test_x, columns=self._feature_names)
        test_df = self._standardize_df(test_df, flag='test')
  
        if isinstance(time_list, int) or isinstance(time_list, float):
            timestamps = [time_list]
        else:
            timestamps = time_list

        # generate a sequence of survival probabilities, and predict the
        # corresponding survival time for each probability
        robjects.r('pct <- seq(.001,.999,by=.001)')
        proba_matrix = []
        pred_median = []
        # perform the predictionn for each test timepoint
        for _, row in test_df.iterrows():
            # generate R command:
            # result <- predict(survregWeibull, newdata=list(C.1=0,C.2=1,...),
            # type="quantile", p=pct)
            pd_command = 'result <- predict(survregWeibull, newdata=list('
            for index, column in enumerate(self.column_ref):
                r_var_name = self.column_ref[column]
                pd_command += r_var_name + '=' + str(row[column])
                if index != len(self.column_ref) - 1:
                    pd_command += ','
                else:
                    pd_command += '), type="quantile", p=pct)'
            # result is a list of time that has the survival probability from
            # 0.999 to 0.001. Notice that time in result list is in ascending
            # order
            result = np.array(list(robjects.r(pd_command)))
            curr_proba = compute_proba_from_quantiles(result, time_list)
            proba_matrix.append(curr_proba)
            # the 500th time is the time that has a survival probability of 0.5
            curr_median = result[499]
            if curr_median == np.inf:
                curr_median = np.max(time_list)
            pred_median.append(curr_median)

        return np.array(pred_median), pd.DataFrame(np.transpose(proba_matrix), index=np.array(time_list))

class WeibullRegressionModel_PCA:

    def __init__(self, n_components):
        """
        This is a python wrapper of R's Weibull Regression Model (in Survival
        package). rpy2 is used to call R's functions.

        :param n_components: number of principle components for PCA
        """
        # prepare R's packages
        self.survival = importr("survival")
        self.base = importr('base')

        self.pca = PCA(n_components=int(n_components),random_state=47) 
        self.standardalizer = StandardScaler()
        self.standardize_data = True

    def set_standardize(self, standardize_bool):
        self.standardize_data = standardize_bool

    def _standardize_df(self, df, flag):
        # if flag = test, the df passed in does not contain Y labels
        if self.standardize_data:
            df_x = df.drop(columns=[self._duration_col, self._event_col]) if flag == 'train' else df
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
                standardized_values = self.standardalizer.fit_transform(df_x)
                standardized_df = pd.DataFrame(data=standardized_values).set_index(df_x.index)
                standardized_df.columns = df_x.columns
                standardized_df[self._duration_col] = df[self._duration_col]
                standardized_df[self._event_col] = df[self._event_col]
            else:
                standardized_values = self.standardalizer.transform(df_x)
                standardized_df = pd.DataFrame(data=standardized_values).set_index(df_x.index)
                standardized_df.columns = df_x.columns
            return standardized_df
        else:
            return df

    def _train_pca(self, train_df):
        """
        Conduct PCA dimension reduction for train dataset if pca_flag is true.

        :param train_df: original train DataFrame
        :return: train DataFrame after dimension reduction
        """
        train_x = train_df.drop(
            columns=[self._duration_col, self._event_col]).values
        # fit and transform
        self.pca.fit(train_x)
        reduced_x = self.pca.transform(train_x)
        # convert back to DataFrame
        reduced_train_df = pd.DataFrame(reduced_x).set_index(train_df.index)
        # we don't care about the column name here
        columns = ["C" + str(i) for i in range(reduced_train_df.shape[1])]
        reduced_train_df.columns = columns
        # get back the y (LOS and OUT)
        reduced_train_df[self._duration_col] = train_df[self._duration_col]
        reduced_train_df[self._event_col] = train_df[self._event_col]
        return reduced_train_df

    def _test_pca(self, test_df):
        """
        Conduct PCA dimension reduction for test dataset if pca_flag is true.

        :param test_df: original test DataFrame
        :return: test DataFrame after dimension reduction
        """
        test_x = test_df.values
        # transform
        reduced_x = self.pca.transform(test_x)
        # convert back to DataFrame
        reduced_test_df = pd.DataFrame(reduced_x).set_index(test_df.index)
        columns = ["C" + str(i) for i in range(reduced_test_df.shape[1])]
        reduced_test_df.columns = columns
        return reduced_test_df

    def fit(self, train_x, train_y, feature_names, duration_col='LOS', event_col='OUT'):
        """
        As we only need to run a couple of lines in R, here we directly generate
        R command as python string, and call R using rpy2. There should be a
        more elegant way of using rpy2.

        :param train_df: DataFrame, with the duration and the event column
        :param duration_col: the column name for duration
        :param event_col: the column name for event
        """
        train_df = pd.DataFrame(data=train_x, columns=feature_names)
        train_df[duration_col] = train_y[:, 0]
        train_df[event_col] = train_y[:, 1]

        self._feature_names = feature_names
        self._duration_col = duration_col
        self._event_col = event_col

        train_df = self._standardize_df(train_df, flag='train')
        reduced_train_df = self._train_pca(train_df)

        self.column_ref = {}

        # Generate R command.
        # Basically we first generate a vector for each feature we have:
        # C.i <- c(1,0,2,1,...)
        # Then we combine those vectors into a dataframe in R
        # train.df <- data.frame(C.1, C.2, ..., OUT, LOS)
        # Then we fit the model
        # s <- Surv(train.df$LOS, train.df$OUT)
        # survregWeibull <- survreg(s ~ C1 + C2 + ..., train.df, dist="weibull")
        df_command = 'train.df <- data.frame('
        sr_command = 'survregWeibull <- survreg(s ~ '
        columns = \
            reduced_train_df.drop(columns=[duration_col, event_col]).columns
        for index, column in enumerate(columns):
            # we don't actually care about the column name in R
            r_var_name = "C." + str(index)
            self.column_ref[column] = r_var_name
            # generate a R vector for each feature
            robjects.globalenv[r_var_name] = \
                FloatVector(list(reduced_train_df[column]))
            df_command += r_var_name + ', '
            sr_command += r_var_name
            if index != len(columns) - 1:
                sr_command += ' + '
        # generate a R vector for LOS
        robjects.globalenv[duration_col] = \
            FloatVector(list(reduced_train_df[duration_col]))
        # generate a R vector for OUT
        robjects.globalenv[event_col] = \
            FloatVector(list(reduced_train_df[event_col]))
        df_command += duration_col + ', ' + event_col + ')'
        sr_command += ', train.df, dist = "weibull")'

        robjects.r(df_command)
        robjects.r('s <- Surv(train.df$' + duration_col + ', train.df$' +
                   event_col + ')')
        robjects.r(sr_command)

    def predict(self, test_x, time_list):
        """
        :param test_df: DataFrame
        :param time_list: checkpoint time to calculate probability on
        :return: the list of median survival time and the probability matrix
        """
        test_df = pd.DataFrame(data=test_x, columns=self._feature_names)
        test_df = self._standardize_df(test_df, flag='test')
        reduced_test_df = self._test_pca(test_df)

        if isinstance(time_list, int) or isinstance(time_list, float):
            timestamps = [time_list]
        else:
            timestamps = time_list

        # generate a sequence of survival probabilities, and predict the
        # corresponding survival time for each probability
        robjects.r('pct <- seq(.001,.999,by=.001)')
        proba_matrix = []
        pred_median = []
        # perform the predictionn for each test timepoint
        for _, row in reduced_test_df.iterrows():
            # generate R command:
            # result <- predict(survregWeibull, newdata=list(C.1=0,C.2=1,...),
            # type="quantile", p=pct)
            pd_command = 'result <- predict(survregWeibull, newdata=list('
            for index, column in enumerate(self.column_ref):
                r_var_name = self.column_ref[column]
                pd_command += r_var_name + '=' + str(row[column])
                if index != len(self.column_ref) - 1:
                    pd_command += ','
                else:
                    pd_command += '), type="quantile", p=pct)'
            # result is a list of time that has the survival probability from
            # 0.999 to 0.001. Notice that time in result list is in ascending
            # order
            result = np.array(list(robjects.r(pd_command)))
            curr_proba = compute_proba_from_quantiles(result, time_list)
            proba_matrix.append(curr_proba)
            # the 500th time is the time that has a survival probability of 0.5
            curr_median = result[499]
            if curr_median == np.inf:
                curr_median = np.max(time_list)
            pred_median.append(curr_median)

        return np.array(pred_median), pd.DataFrame(np.transpose(proba_matrix), index=np.array(time_list))
