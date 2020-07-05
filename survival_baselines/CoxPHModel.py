import glmnet_python
from glmnet import glmnet
from glmnetCoef import glmnetCoef

from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from collections import Counter
from multiprocessing import Pool

'''
coxph: Cox proportional hazard model using glmnet, params = {lambda(for lasso)}
coxph_pca: coxph with pca dimensionality reduction, params = {lambda, n_components(for pca)}

default:
    - entirely lasso regularization (no ridge), controlled by lambda
    - feature standardization

implemented by George Chen & Ren Zuo, last modified by Lexie Li @ 2020.02
'''
def predict_survival_function_par(args):
    subject_x, par_pred_dict_1 = args
    log_hazard = par_pred_dict_1["log_baseline_hazard"] + np.inner(par_pred_dict_1["beta"], subject_x)
    survival_proba = np.zeros(par_pred_dict_1["num_unique_times"])
    for time_idx in range(par_pred_dict_1["num_unique_times"]):
        # log cumulative hazard at this time point
        log_cumulative_hazard = logsumexp(log_hazard[:time_idx + 1])
        # the corresponding probability
        survival_proba[time_idx] = np.exp(-np.exp(log_cumulative_hazard))
    return survival_proba

def predict_median_survival_time_par(args):
    survival_proba, time_list = args
    median_time = max(time_list)
    # if the predicted proba never goes below 0.5, predict the largest seen value
    # the survival_proba is in descending order
    for col, proba in enumerate(survival_proba):
        if proba > 0.5:
            continue
        if proba == 0.5 or col == 0:
            median_time = time_list[col]
        else:
            median_time = (time_list[col - 1] + time_list[col]) / 2
        break
    
    return median_time

class CoxPHModel:

    def __init__(self, lambda_, alpha=1):
        """
        The Cox Proportional Hazard Model is implemented based on `glmnet`.
        We did not use the lifelines's implementation because it does not
        support regularization.

        :param lambda_: the parameter of lasso.
        :param alpha: elastic regularization parameter. 1 -> lasso, 0 -> ridge,
        0~1 -> combination of lasso and ridge. currently we use lasso.
        """
        self._alpha = alpha
        self._lambda = lambda_
        self.standardalizer = None
        self.standardize_data = True

    def set_standardize(self, standardize_bool):
        self.standardize_data = standardize_bool

    # def _standardize_df(self, df, flag):
    #     # if flag = test, the df passed in does not contain Y labels
    #     if self.standardize_data:
    #         df_x = df.drop(columns=[self._duration_col, self._event_col]) if flag == 'train' else df
    #         df_x_vals = df_x.values
    #         if flag == 'train':
    #             self.standardalizer.fit(df_x_vals)
    #         df_x_vals = self.standardalizer.transform(df_x_vals)
    #         standardized_df = pd.DataFrame(df_x_vals).set_index(df.index)
    #         standardized_df.columns = df_x.columns
    #         if flag == 'train':
    #             standardized_df[self._duration_col] = df[self._duration_col]
    #             standardized_df[self._event_col] = df[self._event_col]
    #         return standardized_df
    #     else:
    #         return df

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

    def _find_nearest_time_index(self, time):
        """
        Our implementation in fit determined that we can only calculate log
        hazard at timepoint from the train dataset. if we want to calculate
        probability on other time points, we first need to map that timepoint
        to a time that is already known to the model.
        This helper function simply maps the time_list we want use to the train
        dataset's unique time list.

        :param time: The time list we want to calculate survival probability on
        :return:
        """
        nearest_time_index = -1
        nearest_time = -np.inf
        for index, tmp_time in enumerate(self.sorted_unique_times):
            if tmp_time == time:
                return index
            elif tmp_time < time:
                nearest_time = tmp_time
                nearest_time_index = index
            else:
                if time - nearest_time > tmp_time - time:
                    nearest_time_index = index
                break
        return nearest_time_index

    def fit(self, train_x, train_y, feature_names, duration_col='LOS', event_col='OUT'):
        """
        Given the train dataset, we firstly use glmnet to find the beta (for
        regression). Then we calculate the log baseline hazard (implemented by
        George, modified by Ren).

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
        train_y = train_df[[duration_col, event_col]].values
        train_x = train_df.drop(columns=[duration_col, event_col]).values

        # Find beta. Modified from George's demo.
        fit = glmnet(x=train_x.copy(), y=train_y.copy(),
                     family='cox', alpha=self._alpha, standardize=False, # we performed our own standardization
                     intr=False)
        self.beta = glmnetCoef(fit, s=np.array([self._lambda])).flatten()

        # self.beta = cph_kera(x = train_x.copy(), y = train_y.copy(), \
        #     alpha = self._alpha, lmbda = self._lambda, standardize = True)
 
        observed_times = train_y[:, 0]
        event_indicators = train_y[:, 1]
        # For each observed time, how many times the event occurred
        event_counts = Counter()
        for t, r in zip(observed_times, event_indicators):
            event_counts[t] += int(r)
        # Sorted list of observed times
        self.sorted_unique_times = np.sort(list(event_counts.keys()))
        self.num_unique_times = len(self.sorted_unique_times)
        self.log_baseline_hazard = np.zeros(self.num_unique_times)
        # Calculate the log baseline hazard. Implemented by George.
        for time_idx, t in enumerate(self.sorted_unique_times):
            logsumexp_args = []
            for subj_idx, observed_time in enumerate(observed_times):
                if observed_time >= t:
                    logsumexp_args.append(
                        np.inner(self.beta, train_x[subj_idx]))
            if event_counts[t] > 0:
                self.log_baseline_hazard[time_idx] = \
                    np.log(event_counts[t]) - logsumexp(logsumexp_args)
            else:
                self.log_baseline_hazard[time_idx] = \
                    -np.inf - logsumexp(logsumexp_args)

    def pred_median_time(self, test_df, average_to_get_median=True):
        """
        Given the beta and baseline hazard, for each test datapoint, we can
        calculate the hazard function. Based on that, we find the time when
        the survival probability is around 0.5.

        :param test_df: DataFrame
        :return: the list of median survival time for each test datapoint.
        """
        test_df = self._standardize_df(test_df, flag='test')
        test_x = test_df.values

        num_test_subjects = len(test_x)
        median_survival_times = np.zeros(num_test_subjects)

        # log(-log(0.5)) -> instead of calculating the survival function based
        # on the hazard function and then finding the time point near the Pr of
        # 0.5, we actually change the question into: find the timepoint when
        # the log cumulative hazard is near log(-log(0.5)).
        log_minus_log_half = np.log(-np.log(0.5))

        for subj_idx in range(num_test_subjects):
            # log hazard of the given object
            log_hazard = \
                self.log_baseline_hazard + np.inner(self.beta, test_x[subj_idx])
            # simulate the integral to get the log cumulative hazard function
            log_cumulative_hazard = np.zeros(self.num_unique_times)
            for time_idx in range(self.num_unique_times):
                log_cumulative_hazard[time_idx] \
                    = logsumexp(log_hazard[:time_idx + 1])
            # find the time when the log cumulative hazard near log(-log(0.5))
            t_inf = np.inf
            t_sup = 0.
            # notice that we use the list of unique observed time from the
            # training dataset as the source to look for the time
            for time_idx, t in enumerate(self.sorted_unique_times):
                cur_chazard = log_cumulative_hazard[time_idx]
                if log_minus_log_half <= cur_chazard and t < t_inf:
                    t_inf = t
                if log_minus_log_half >= cur_chazard and t > t_sup:
                    t_sup = t
            if average_to_get_median:
                median_survival_times[subj_idx] = 0.5 * (t_inf + t_sup)
            else:
                median_survival_times[subj_idx] = t_inf
        return median_survival_times

    def pred_proba(self, test_df, time_list):
        """
        Given the beta and baseline hazard, for each test datapoint, we can
        calculate the hazard function, and then calculate the survival function.

        :param test_df: DataFrame
        :param time_list: checkpoint time to calculate probability on
        :return: the probability matrix. each row is the survival function for
        one test datapoint.
        """
        test_df = self._standardize_df(test_df, flag='test')
        test_x = test_df.values

        pred_proba_dict = dict()
        pred_proba_dict["log_baseline_hazard"] = self.log_baseline_hazard
        pred_proba_dict["beta"] = self.beta
        pred_proba_dict["num_unique_times"] = self.num_unique_times

        predict_survival_function_inputs = [(subject_x, pred_proba_dict) for subject_x in test_x]
        tmp_proba = np.array(self.predict_pool.map(predict_survival_function_par, predict_survival_function_inputs), dtype="float32")
        # the tmp_proba would be a matrix of n * k, where n is the number of
        # test datapoints, and k is the number of unique observed time from the
        # train dataset

        # # Using the mapping between our input time_list to the train dataset
        # # time_list.
        # time_indices = [self._find_nearest_time_index(cur_time)
        #                 for cur_time in time_list]

        # # the proba_matrix would be a matrix of n * m, where n is the number of
        # # test datapoints, and m is the number of unique time we want to
        # # estimate on (i.e. len(time))
        # proba_matrix = np.zeros((len(test_x), len(time_indices)))
        # for row in range(len(test_x)):
        #     for col, time_index in enumerate(time_indices):
        #         proba_matrix[row][col] = tmp_proba[row][time_index]
        proba_matrix = tmp_proba

        return proba_matrix

    def predict(self, test_x, time_list):
        """
        :param test_df: DataFrame
        :param time_list: checkpoint time to calculate probability on
        :return: the list of median survival time and the probability matrix
        """
        test_df = pd.DataFrame(data=test_x, columns=self._feature_names)

        self.predict_pool = Pool(processes=None)
        proba_matrix = self.pred_proba(test_df, time_list)
        predict_median_survival_time_inputs = [(survival_proba, time_list) for survival_proba in proba_matrix]
        pred_medians = self.predict_pool.map(predict_median_survival_time_par, predict_median_survival_time_inputs)
        self.predict_pool.close()
        self.predict_pool.join()

        return np.array(pred_medians), pd.DataFrame(np.transpose(proba_matrix), index=np.array(time_list))

class CoxPHModel_PCA:

    def __init__(self, lambda_, n_components, alpha=1):
        """
        The Cox Proportional Hazard Model is implemented based on `glmnet`.
        We did not use the lifelines's implementation because it does not
        support regularization.

        :param lambda_: the parameter of lasso.
        :param alpha: elastic regularization parameter. 1 -> lasso, 0 -> ridge,
        0~1 -> combination of lasso and ridge. currently we use lasso.
        :param pca_flag: use PCA before fit the model.
        :param n_components: number of principle components for PCA
        """
        self._alpha = alpha
        self._lambda = lambda_
        # bayesian opt does not guarantee to pass in int values
        self.pca = PCA(n_components=int(n_components),random_state=47) 
        self.standardalizer = None
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

    def _find_nearest_time_index(self, time):
        """
        Our implementation in fit determined that we can only calculate log
        hazard at timepoint from the train dataset. if we want to calculate
        probability on other time points, we first need to map that timepoint
        to a time that is already known to the model.
        This helper function simply maps the time_list we want use to the train
        dataset's unique time list.

        :param time: The time list we want to calculate survival probability on
        :return:
        """
        nearest_time_index = -1
        nearest_time = -np.inf
        for index, tmp_time in enumerate(self.sorted_unique_times):
            if tmp_time == time:
                return index
            elif tmp_time < time:
                nearest_time = tmp_time
                nearest_time_index = index
            else:
                if time - nearest_time > tmp_time - time:
                    nearest_time_index = index
                break
        return nearest_time_index

    def fit(self, train_x, train_y, feature_names, duration_col='LOS', event_col='OUT'):
        """
        Given the train dataset, we firstly use glmnet to find the beta (for
        regression). Then we calculate the log baseline hazard (implemented by
        George, modified by Ren).

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
        train_y = reduced_train_df[[duration_col, event_col]].values
        train_x = \
            reduced_train_df.drop(columns=[duration_col, event_col]).values

        # Find beta. Modified from George's demo.
        fit = glmnet(x=train_x.copy(), y=train_y.copy(),
                     family='cox', alpha=self._alpha, standardize=False,
                     intr=False)
        self.beta = glmnetCoef(fit, s=np.array([self._lambda])).flatten()

        # self.beta = cph_kera(x = train_x.copy(), y = train_y.copy(), \
        #     alpha = self._alpha, lmbda = self._lambda, standardize = True)
 
        observed_times = train_y[:, 0]
        event_indicators = train_y[:, 1]
        # For each observed time, how many times the event occurred
        event_counts = Counter()
        for t, r in zip(observed_times, event_indicators):
            event_counts[t] += int(r)
        # Sorted list of observed times
        self.sorted_unique_times = np.sort(list(event_counts.keys()))
        self.num_unique_times = len(self.sorted_unique_times)
        self.log_baseline_hazard = np.zeros(self.num_unique_times)
        # Calculate the log baseline hazard. Implemented by George.
        for time_idx, t in enumerate(self.sorted_unique_times):
            logsumexp_args = []
            for subj_idx, observed_time in enumerate(observed_times):
                if observed_time >= t:
                    logsumexp_args.append(
                        np.inner(self.beta, train_x[subj_idx]))
            if event_counts[t] > 0:
                self.log_baseline_hazard[time_idx] = \
                    np.log(event_counts[t]) - logsumexp(logsumexp_args)
            else:
                self.log_baseline_hazard[time_idx] = \
                    -np.inf - logsumexp(logsumexp_args)

    # def pred_median_time(self, test_df, average_to_get_median=True):
    #     """
    #     Given the beta and baseline hazard, for each test datapoint, we can
    #     calculate the hazard function. Based on that, we find the time when
    #     the survival probability is around 0.5.

    #     :param test_df: DataFrame
    #     :return: the list of median survival time for each test datapoint.
    #     """

    #     test_df = self._standardize_df(test_df, flag='test')
    #     # Dimension reduction first if applicable
    #     reduced_test_df = self._test_pca(test_df)
    #     test_x = reduced_test_df.values

    #     num_test_subjects = len(test_x)
    #     median_survival_times = np.zeros(num_test_subjects)

    #     # log(-log(0.5)) -> instead of calculating the survival function based
    #     # on the hazard function and then finding the time point near the Pr of
    #     # 0.5, we actually change the question into: find the timepoint when
    #     # the log cumulative hazard is near log(-log(0.5)).
    #     log_minus_log_half = np.log(-np.log(0.5))

    #     for subj_idx in range(num_test_subjects):
    #         # log hazard of the given object
    #         log_hazard = \
    #             self.log_baseline_hazard + np.inner(self.beta, test_x[subj_idx])
    #         # simulate the integral to get the log cumulative hazard function
    #         log_cumulative_hazard = np.zeros(self.num_unique_times)
    #         for time_idx in range(self.num_unique_times):
    #             log_cumulative_hazard[time_idx] \
    #                 = logsumexp(log_hazard[:time_idx + 1])
    #         # find the time when the log cumulative hazard near log(-log(0.5))
    #         t_inf = np.inf
    #         t_sup = 0.
    #         # notice that we use the list of unique observed time from the
    #         # training dataset as the source to look for the time
    #         for time_idx, t in enumerate(self.sorted_unique_times):
    #             cur_chazard = log_cumulative_hazard[time_idx]
    #             if log_minus_log_half <= cur_chazard and t < t_inf:
    #                 t_inf = t
    #             if log_minus_log_half >= cur_chazard and t > t_sup:
    #                 t_sup = t
    #         if average_to_get_median:
    #             median_survival_times[subj_idx] = 0.5 * (t_inf + t_sup)
    #         else:
    #             median_survival_times[subj_idx] = t_inf
    #     return median_survival_times

    def pred_proba(self, test_df, time_list):
        """
        Given the beta and baseline hazard, for each test datapoint, we can
        calculate the hazard function, and then calculate the survival function.

        :param test_df: DataFrame
        :param time_list: checkpoint time to calculate probability on
        :return: the probability matrix. each row is the survival function for
        one test datapoint.
        """
        test_df = self._standardize_df(test_df, flag='test')
        # Dimension reduction first if applicable
        reduced_test_df = self._test_pca(test_df)
        # Get the ndarray representation of the test dataset
        test_x = reduced_test_df.values

        pred_proba_dict = dict()
        pred_proba_dict["log_baseline_hazard"] = self.log_baseline_hazard
        pred_proba_dict["beta"] = self.beta
        pred_proba_dict["num_unique_times"] = self.num_unique_times

        predict_survival_function_inputs = [(subject_x, pred_proba_dict) for subject_x in test_x]
        tmp_proba = np.array(self.predict_pool.map(predict_survival_function_par, predict_survival_function_inputs), dtype="float32")
        # the tmp_proba would be a matrix of n * k, where n is the number of
        # test datapoints, and k is the number of unique observed time from the
        # train dataset

        # # Using the mapping between our input time_list to the train dataset
        # # time_list.
        # time_indices = [self._find_nearest_time_index(cur_time)
        #                 for cur_time in time_list]

        # # the proba_matrix would be a matrix of n * m, where n is the number of
        # # test datapoints, and m is the number of unique time we want to
        # # estimate on (i.e. len(time))
        # proba_matrix = np.zeros((len(test_x), len(time_indices)))
        # for row in range(len(test_x)):
        #     for col, time_index in enumerate(time_indices):
        #         proba_matrix[row][col] = tmp_proba[row][time_index]
        proba_matrix = tmp_proba

        return proba_matrix

    def predict(self, test_x, time_list):
        """
        :param test_df: DataFrame
        :param time_list: checkpoint time to calculate probability on
        :return: the list of median survival time and the probability matrix
        """
        test_df = pd.DataFrame(data=test_x, columns=self._feature_names)

        self.predict_pool = Pool(processes=None)
        proba_matrix = self.pred_proba(test_df, time_list)
        predict_median_survival_time_inputs = [(survival_proba, time_list) for survival_proba in proba_matrix]
        pred_medians = self.predict_pool.map(predict_median_survival_time_par, predict_median_survival_time_inputs)
        self.predict_pool.close()
        self.predict_pool.join()

        return np.array(pred_medians), pd.DataFrame(np.transpose(proba_matrix), index=np.array(time_list))

class CoxPHModel0:

    def __init__(self, lambda_=1e-100, alpha=1):
        """
        The Cox Proportional Hazard Model is implemented based on `glmnet`.
        We did not use the lifelines's implementation because it does not
        support regularization.

        :param lambda_: the parameter of lasso.
        :param alpha: elastic regularization parameter. 1 -> lasso, 0 -> ridge,
        0~1 -> combination of lasso and ridge. currently we use lasso.
        """
        self._alpha = alpha
        self._lambda = lambda_
        self.standardalizer = None
        self.standardize_data = True

    def set_standardize(self, standardize_bool):
        self.standardize_data = standardize_bool

    # def _standardize_df(self, df, flag):
    #     # if flag = test, the df passed in does not contain Y labels
    #     if self.standardize_data:
    #         df_x = df.drop(columns=[self._duration_col, self._event_col]) if flag == 'train' else df
    #         df_x_vals = df_x.values
    #         if flag == 'train':
    #             self.standardalizer.fit(df_x_vals)
    #         df_x_vals = self.standardalizer.transform(df_x_vals)
    #         standardized_df = pd.DataFrame(df_x_vals).set_index(df.index)
    #         standardized_df.columns = df_x.columns
    #         if flag == 'train':
    #             standardized_df[self._duration_col] = df[self._duration_col]
    #             standardized_df[self._event_col] = df[self._event_col]
    #         return standardized_df
    #     else:
    #         return df

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

    def _find_nearest_time_index(self, time):
        """
        Our implementation in fit determined that we can only calculate log
        hazard at timepoint from the train dataset. if we want to calculate
        probability on other time points, we first need to map that timepoint
        to a time that is already known to the model.
        This helper function simply maps the time_list we want use to the train
        dataset's unique time list.

        :param time: The time list we want to calculate survival probability on
        :return:
        """
        nearest_time_index = -1
        nearest_time = -np.inf
        for index, tmp_time in enumerate(self.sorted_unique_times):
            if tmp_time == time:
                return index
            elif tmp_time < time:
                nearest_time = tmp_time
                nearest_time_index = index
            else:
                if time - nearest_time > tmp_time - time:
                    nearest_time_index = index
                break
        return nearest_time_index

    def fit(self, train_x, train_y, feature_names, duration_col='LOS', event_col='OUT'):
        """
        Given the train dataset, we firstly use glmnet to find the beta (for
        regression). Then we calculate the log baseline hazard (implemented by
        George, modified by Ren).

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
        train_y = train_df[[duration_col, event_col]].values
        train_x = train_df.drop(columns=[duration_col, event_col]).values

        # Find beta. Modified from George's demo.
        fit = glmnet(x=train_x.copy(), y=train_y.copy(),
                     family='cox', alpha=self._alpha, standardize=False, # we performed our own standardization
                     intr=False)
        self.beta = glmnetCoef(fit, s=np.array([self._lambda])).flatten()

        # self.beta = cph_kera(x = train_x.copy(), y = train_y.copy(), \
        #     alpha = self._alpha, lmbda = self._lambda, standardize = True)
 
        observed_times = train_y[:, 0]
        event_indicators = train_y[:, 1]
        # For each observed time, how many times the event occurred
        event_counts = Counter()
        for t, r in zip(observed_times, event_indicators):
            event_counts[t] += int(r)
        # Sorted list of observed times
        self.sorted_unique_times = np.sort(list(event_counts.keys()))
        self.num_unique_times = len(self.sorted_unique_times)
        self.log_baseline_hazard = np.zeros(self.num_unique_times)
        # Calculate the log baseline hazard. Implemented by George.
        for time_idx, t in enumerate(self.sorted_unique_times):
            logsumexp_args = []
            for subj_idx, observed_time in enumerate(observed_times):
                if observed_time >= t:
                    logsumexp_args.append(
                        np.inner(self.beta, train_x[subj_idx]))
            if event_counts[t] > 0:
                self.log_baseline_hazard[time_idx] = \
                    np.log(event_counts[t]) - logsumexp(logsumexp_args)
            else:
                self.log_baseline_hazard[time_idx] = \
                    -np.inf - logsumexp(logsumexp_args)

    def pred_median_time(self, test_df, average_to_get_median=True):
        """
        Given the beta and baseline hazard, for each test datapoint, we can
        calculate the hazard function. Based on that, we find the time when
        the survival probability is around 0.5.

        :param test_df: DataFrame
        :return: the list of median survival time for each test datapoint.
        """
        test_df = self._standardize_df(test_df, flag='test')
        test_x = test_df.values

        num_test_subjects = len(test_x)
        median_survival_times = np.zeros(num_test_subjects)

        # log(-log(0.5)) -> instead of calculating the survival function based
        # on the hazard function and then finding the time point near the Pr of
        # 0.5, we actually change the question into: find the timepoint when
        # the log cumulative hazard is near log(-log(0.5)).
        log_minus_log_half = np.log(-np.log(0.5))

        for subj_idx in range(num_test_subjects):
            # log hazard of the given object
            log_hazard = \
                self.log_baseline_hazard + np.inner(self.beta, test_x[subj_idx])
            # simulate the integral to get the log cumulative hazard function
            log_cumulative_hazard = np.zeros(self.num_unique_times)
            for time_idx in range(self.num_unique_times):
                log_cumulative_hazard[time_idx] \
                    = logsumexp(log_hazard[:time_idx + 1])
            # find the time when the log cumulative hazard near log(-log(0.5))
            t_inf = np.inf
            t_sup = 0.
            # notice that we use the list of unique observed time from the
            # training dataset as the source to look for the time
            for time_idx, t in enumerate(self.sorted_unique_times):
                cur_chazard = log_cumulative_hazard[time_idx]
                if log_minus_log_half <= cur_chazard and t < t_inf:
                    t_inf = t
                if log_minus_log_half >= cur_chazard and t > t_sup:
                    t_sup = t
            if average_to_get_median:
                median_survival_times[subj_idx] = 0.5 * (t_inf + t_sup)
            else:
                median_survival_times[subj_idx] = t_inf
        return median_survival_times

    def pred_proba(self, test_df, time_list):
        """
        Given the beta and baseline hazard, for each test datapoint, we can
        calculate the hazard function, and then calculate the survival function.

        :param test_df: DataFrame
        :param time_list: checkpoint time to calculate probability on
        :return: the probability matrix. each row is the survival function for
        one test datapoint.
        """
        test_df = self._standardize_df(test_df, flag='test')
        test_x = test_df.values

        pred_proba_dict = dict()
        pred_proba_dict["log_baseline_hazard"] = self.log_baseline_hazard
        pred_proba_dict["beta"] = self.beta
        pred_proba_dict["num_unique_times"] = self.num_unique_times

        predict_survival_function_inputs = [(subject_x, pred_proba_dict) for subject_x in test_x]
        tmp_proba = np.array(self.predict_pool.map(predict_survival_function_par, predict_survival_function_inputs), dtype="float32")
        # the tmp_proba would be a matrix of n * k, where n is the number of
        # test datapoints, and k is the number of unique observed time from the
        # train dataset

        # # Using the mapping between our input time_list to the train dataset
        # # time_list.
        # time_indices = [self._find_nearest_time_index(cur_time)
        #                 for cur_time in time_list]

        # # the proba_matrix would be a matrix of n * m, where n is the number of
        # # test datapoints, and m is the number of unique time we want to
        # # estimate on (i.e. len(time))
        # proba_matrix = np.zeros((len(test_x), len(time_indices)))
        # for row in range(len(test_x)):
        #     for col, time_index in enumerate(time_indices):
        #         proba_matrix[row][col] = tmp_proba[row][time_index]
        proba_matrix = tmp_proba

        return proba_matrix

    def predict(self, test_x, time_list):
        """
        :param test_df: DataFrame
        :param time_list: checkpoint time to calculate probability on
        :return: the list of median survival time and the probability matrix
        """
        test_df = pd.DataFrame(data=test_x, columns=self._feature_names)

        self.predict_pool = Pool(processes=None)
        proba_matrix = self.pred_proba(test_df, time_list)
        predict_median_survival_time_inputs = [(survival_proba, time_list) for survival_proba in proba_matrix]
        pred_medians = self.predict_pool.map(predict_median_survival_time_par, predict_median_survival_time_inputs)
        self.predict_pool.close()
        self.predict_pool.join()

        return np.array(pred_medians), pd.DataFrame(np.transpose(proba_matrix), index=np.array(time_list))
