
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 
from lifelines import KaplanMeierFitter
import numpy as np
import pandas as pd

'''
knnkm: K-Nearest Neighbor model using sklearn + KaplanMeier using lifelines, params = {n_neighbors}
knnkm_pca: knnkm with pca dimensionality reduction, params = {n_neighbors, n_components}

default:
    - feature standardization

implemented by George Chen & Ren Zuo, last modified by Lexie Li @ 2019.12
'''

class KNNKaplanMeier():

    def __init__(self, n_neighbors):
        """
        The K-Nearest Neighbor model is implemented using sikit-learn's
        KNeighborsClassifier along with lifelines's KaplanMeierFitter.
        The `pred_proba` and `pred_median_time` are based on KaplanMeierFitter's
        predict methods.

        :param n_neighbors:
        """
        self.n_neighbors = n_neighbors
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

        self.train_df  = self._standardize_df(train_df, flag='train')
        train_x = self.train_df.drop(columns=[duration_col, event_col]).values
        # For kNN, we simply initiate the KNeighborsClassifier
        # the n_neighbors will be used later when calculating nearest neighbor
        self.neighbors = KNeighborsClassifier()
        self.neighbors.fit(train_x, np.zeros(len(train_x)))
        self.train_points = len(train_x)

    def predict(self, test_x, time_list):
        """
        for each test datapoint, find the k nearest neighbors, and use them to
        fit a Kaplan-Meier Model to get the survival function, and then use
        the survival function the calculate the median survival time

        :param test_df: DataFrame
        :param time_list: checkpoint time to calculate probability on
        :return: the list of median survival time and the probability matrix
        """
        test_df = pd.DataFrame(data=test_x, columns=self._feature_names)
        test_df = self._standardize_df(test_df, flag='test')
        test_x = test_df.values
        # calculate distance matrix to find the nearest neighbors
        distance_matrix, neighbor_matrix = \
            self.neighbors.kneighbors(
                X=test_x,
                n_neighbors=int(np.min([self.n_neighbors, self.train_points]))
            )

        proba_matrix = []
        test_time_median_pred = []
        for test_idx, test_point in enumerate(test_x):
            # find the k nearest neighbors
            neighbor_train_y = \
                self.train_df.iloc[neighbor_matrix[test_idx]][
                    [self._duration_col, self._event_col]
                ]
            kmf = KaplanMeierFitter()
            kmf.fit(neighbor_train_y[self._duration_col],
                    neighbor_train_y[self._event_col])
            survival_proba = kmf.predict(time_list)
            # calculate the median survival time.
            # the median survival time is the time at which the survival proba.
            # equals to 0.5. Here the survival_proba is descending sorted from
            # 1 to 0, so we only need to find the first probability that <= 0.5
            median_time = np.max(time_list)
            for col, proba in enumerate(survival_proba):
                if proba > 0.5:
                    continue

                if proba == 0.5:
                    median_time = time_list[col]
                else:
                    # here we take the average of the time before and after
                    # Pr = 0.5
                    median_time = (time_list[col - 1] + time_list[col]) / 2
                break

            test_time_median_pred.append(median_time)
            proba_matrix.append(survival_proba)

        return np.array(test_time_median_pred), \
               pd.DataFrame(np.transpose(np.array(proba_matrix)), index=np.array(time_list))

class KNNKaplanMeier_PCA():

    def __init__(self, n_neighbors, n_components):
        """
        The K-Nearest Neighbor model is implemented using sikit-learn's
        KNeighborsClassifier along with lifelines's KaplanMeierFitter.
        The `pred_proba` and `pred_median_time` are based on KaplanMeierFitter's
        predict methods.

        :param n_neighbors:
        :param n_components: number of principle components for PCA
        """
        self.n_neighbors = n_neighbors
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
        self.train_df = self._train_pca(train_df)
        train_x = self.train_df.drop(columns=[duration_col, event_col]).values
        # For kNN, we simply initiate the KNeighborsClassifier
        # the n_neighbors will be used later when calculating nearest neighbor
        self.neighbors = KNeighborsClassifier()
        self.neighbors.fit(train_x, np.zeros(len(train_x)))
        self.train_points = len(train_x)

    def predict(self, test_x, time_list):
        """
        for each test datapoint, find the k nearest neighbors, and use them to
        fit a Kaplan-Meier Model to get the survival function, and then use
        the survival function the calculate the median survival time

        :param test_df: DataFrame
        :param time_list: checkpoint time to calculate probability on
        :return: the list of median survival time and the probability matrix
        """
        test_df = pd.DataFrame(data=test_x, columns=self._feature_names)
        test_df = self._standardize_df(test_df, flag='test')
        reduced_test_df = self._test_pca(test_df)
        test_x = reduced_test_df.values
        # calculate distance matrix to find the nearest neighbors
        distance_matrix, neighbor_matrix = \
            self.neighbors.kneighbors(
                X=test_x,
                n_neighbors=int(np.min([self.n_neighbors, self.train_points]))
            )

        proba_matrix = []
        test_time_median_pred = []
        for test_idx, test_point in enumerate(test_x):
            # find the k nearest neighbors
            neighbor_train_y = \
                self.train_df.iloc[neighbor_matrix[test_idx]][
                    [self._duration_col, self._event_col]
                ]
            kmf = KaplanMeierFitter()
            kmf.fit(neighbor_train_y[self._duration_col],
                    neighbor_train_y[self._event_col])
            survival_proba = kmf.predict(time_list)
            # calculate the median survival time.
            # the median survival time is the time at which the survival proba.
            # equals to 0.5. Here the survival_proba is descending sorted from
            # 1 to 0, so we only need to find the first probability that <= 0.5
            median_time = np.max(time_list)
            for col, proba in enumerate(survival_proba):
                if proba > 0.5:
                    continue

                if proba == 0.5:
                    median_time = time_list[col]
                else:
                    # here we take the average of the time before and after
                    # Pr = 0.5
                    median_time = (time_list[col - 1] + time_list[col]) / 2
                break

            test_time_median_pred.append(median_time)
            proba_matrix.append(survival_proba)

        return np.array(test_time_median_pred), \
               pd.DataFrame(np.transpose(np.array(proba_matrix)), index=np.array(time_list))
