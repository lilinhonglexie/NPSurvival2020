
import pandas as pd
import numpy as np
import math
# from collections import defaultdict, Counter
import functools

from joblib import Parallel, delayed
from lifelines.utils import concordance_index
# from sklearn.neighbors import NearestNeighbors

from survival_baselines.random_survival_forest_cython import logrank
from survival_baselines.NPSurvivalModels import \
     _build_tree, _find_best_feature_split, _fit_leaf, _fit_leaf_weighted, \
     _build_tree_ANN, _predict_leaf, _predict_row, _predict_tree, \
     _predict_row_vimp, _predict_tree_vimp, _compute_tree_ANN, _compute_ANN_row

'''
rsf: random survival forest, params = {n_estimators, max_features, max_depth, }

rsf implemented by George Chen, interface designed by Ren Zuo,
merged and last modified by Lexie Li @ 2020.01
'''

class RandomSurvivalForest():
    def __init__(self, max_features, min_samples_leaf, min_samples_split=2, 
                 max_depth=None, n_estimators=100, split='logrank',
                 split_threshold_mode='exhaustive', random_state=47,
                 n_jobs=9, oob_score=True, feature_importance=True):

        """
        A random survival forest survival probability estimator. This is very
        similar to the usual random forest that is used for regression and
        classification. However, in a random survival forest, the prediction
        task is to estimate the survival probability function for a test
        feature vector. Training data can have right-censoring. For details,
        see any introductory text on survival analysis.
        
        Parameters
        ----------
        n_estimators: int, optional (default=10)
            Number of trees.

        max_features : int, string, optional (default='sqrt')
            Number of features chosen per tree. Allowable string choices are
            'sqrt' (max_features=ceil(sqrt(n_features))) and 'log2'
            (max_features=ceil(log2(n_features))).

        max_depth : int, optional (default=None)
            Maximum depth of each tree. If None, then each tree is grown
            until other termination criteria are met (see `min_samples_split`
            and `min_samples_leaf` parameters).

        min_samples_split : int, optional (default=2)
            A node must have at least this many samples to be split.

        min_samples_leaf : int, float, optional (default=1)
            Both sides of a split must have at least this many samples
            (or in the case of a fraction, at least a fraction of samples)
            for the split to happen. Otherwise, the node is turned into a
            leaf node.

        split : string, optional (default='logrank')
            Currently only the log-rank splitting criterion is supported.

        split_threshold_mode : string, optional (default='exhaustive')
            If 'exhaustive', then we compute the split score for every observed
            feature value as a possible threshold (this can be very expensive).
            If 'median', then for any feature, we always split on the median
            value observed for that feature (this is the only supported option
            in Wrymm's original random survival analysis code).
            If 'random', then for any feature, we randomly choose a split
            threshold among the observed feature values (this is recommended by
            the random survival forest authors if fast computation is desired).

        random_state : int, numpy RandomState instance, None, optional
            (default=None)
            If an integer, then a new numpy RandomState is created with the
            integer as the random seed. If a numpy RandomState instance is
            provided, then it is used as the pseudorandom number generator. If
            None is specified, then a new numpy RandomState is created without
            providing a seed.

        n_jobs : int, None, optional (default=None)
            Number of cores to use with joblib's Parallel. This is the same
            `n_jobs` parameter as for Parallel. Setting `n_jobs` to -1 uses all
            the cores.

        oob_score : boolean, optional (default=False)
            Whether to compute an out-of-bag (OOB) accuracy estimate (as with
            the original random survival forest paper, this is done using
            c-index with cumulative hazard estimates). The OOB estimate is
            computed during model fitting (via fit()), and the resulting
            c-index estimate is stored in the attribute `oob_score_`.

        feature_importance : boolean, optional (default=False)
            Whether to compute feature importances (requires `oob_score` to
            be set to True). Feature importances are computed during the
            model fitting (via fit()), and the resulting feature importances is
            stored in the attribute `feature_importances_`.

        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = int(max_features)
        self.split_threshold_mode = split_threshold_mode
        self.n_jobs = n_jobs
        self.oob_score = oob_score
        self.feature_importance = feature_importance
        self.column_names = None
        self.oob_score_ = None
        self.feature_importances_ = None

        if random_state is None:
            self.random_state = np.random.RandomState()
        elif type(random_state) == int:
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = random_state

        if split == 'logrank':
            self.split_score_function = logrank
        else:
            raise NotImplementedError('Unsupported split criterion '
                                      + '"{0}"'.format(split))

        # # name of the time column used within the class. this can be anything
        # self._time_column = "time"
        # # name of the event column used within the class. this can be anything
        # self._event_column = "event"

    # def fit__tbd(self, train_x, train_y, feature_names, duration_col='LOS', event_col='OUT'):
    #     """
    #     Build the forest.

    #     :param train_df: DataFrame, with the duration and the event column
    #     :param duration_col: the column name for duration
    #     :param event_col: the column name for event
    #     :param num_workers:
    #     """
    #     train_df = pd.DataFrame(data=train_x, columns=feature_names)
    #     train_df[duration_col] = train_y[:, 0]
    #     train_df[event_col] = train_y[:, 1]

    #     self._feature_names = feature_names
    #     self._duration_col = duration_col
    #     self._event_col = event_col

    #     x_train = train_df.drop(columns=[duration_col, event_col])
    #     featureNames = list(x_train.columns)

    #     y_train = train_df[[duration_col, event_col]]
    #     y_train.columns = [self._time_column, self._event_column]
    #     self._times = np.sort(list(y_train[self._time_column].unique()))
    #     self.fitMain(X=x_train.values, y=y_train.values)

    def fit(self, X, y, column_names):
        """
        Fits the random survival forest to training data.

        Parameters
        ----------
        X : 2D numpy array, shape = [n_samples, n_features]
            Feature vectors.

        y : 2D numpy array, shape = [n_samples, 2]
            Survival labels (first column is for observed times, second column
            is for event indicators). The i-th row corresponds to the i-th row
            in `X`.

        column_names : list, None, optional (default=None)
            Names for features can be specified. This is only for display
            purposes when using the `draw` method. If set to None, then
            `column_names` is just set to be a range of integers indexing the
            columns from 0.

        Returns
        -------
        None
        """
        if column_names is None:
            self.column_names = list(range(X.shape[1]))
        else:
            self.column_names = column_names
            assert len(column_names) == X.shape[1]

        if type(self.max_features) == str:
            if self.max_features == 'sqrt':
                max_features = int(np.ceil(np.sqrt(X.shape[1])))
            elif self.max_features == 'log2':
                max_features = int(np.ceil(np.log2(X.shape[1])))
            else:
                raise NotImplementedError('Unsupported max features choice '
                                          + '"{0}"'.format(self.max_features))
        else:
            max_features = self.max_features

        self.tree_bootstrap_indices = []
        sort_indices = np.argsort(y[:, 0])
        X = X[sort_indices].astype(np.float)
        y = y[sort_indices].astype(np.float)
        random_state = self.random_state
        for tree_idx in range(self.n_estimators):
            bootstrap_indices = np.sort(random_state.choice(X.shape[0],
                                                            X.shape[0],
                                                            replace=True))
            self.tree_bootstrap_indices.append(bootstrap_indices)

        with Parallel(n_jobs=self.n_jobs) as parallel:
            self.trees = \
                parallel(
                  delayed(_build_tree)(
                      X[self.tree_bootstrap_indices[tree_idx]],
                      y[self.tree_bootstrap_indices[tree_idx]],
                      0, self.max_depth, max_features,
                      self.split_score_function, self.min_samples_split,
                      self.min_samples_leaf, self.split_threshold_mode,
                      np.random.RandomState(random_state.randint(4294967296)))
                  for tree_idx in range(self.n_estimators))

            if self.oob_score:
                parallel_args = []
                oob_masks = []
                for tree_idx, bootstrap_indices \
                        in enumerate(self.tree_bootstrap_indices):
                    oob_mask = np.ones(X.shape[0], dtype=np.bool)
                    for idx in bootstrap_indices:
                        oob_mask[idx] = 0
                    if oob_mask.sum() > 0:
                        X_oob = X[oob_mask]
                        if len(X_oob.shape) == 1:
                            X_oob = X_oob.reshape(1, -1)
                        parallel_args.append((tree_idx, X_oob))
                        oob_masks.append(
                            (oob_mask,
                             {original_idx: new_idx
                              for new_idx, original_idx
                              in enumerate(np.where(oob_mask)[0])}))

                sorted_unique_times = np.unique(y[:, 0])
                results = parallel(
                    delayed(_predict_tree)(
                        self.trees[tree_idx], 'cum_haz', X_oob,
                        sorted_unique_times, True)
                    for (tree_idx, X_oob) in parallel_args)

                num_unique_times = len(sorted_unique_times)
                cum_hazard_scores = []
                oob_y = []
                for idx in range(X.shape[0]):
                    num = 0.
                    den = 0.
                    for tree_idx2, (oob_mask, forward_map) \
                            in enumerate(oob_masks):
                        if oob_mask[idx]:
                            num += results[tree_idx2][forward_map[idx]].sum()
                            den += 1
                    if den > 0:
                        cum_hazard_scores.append(num / den)
                        oob_y.append(y[idx])

                cum_hazard_scores = np.array(cum_hazard_scores)
                oob_y = np.array(oob_y)

                self.oob_score_ = concordance_index(oob_y[:, 0],
                                                    -cum_hazard_scores,
                                                    oob_y[:, 1])

                if self.feature_importance:
                    self.feature_importances_ = []
                    for col_idx in range(X.shape[1]):
                        vimp_results = \
                            parallel(
                                delayed(_predict_tree_vimp)(
                                    self.trees[tree_idx], 'cum_haz',
                                    X_oob, sorted_unique_times, True,
                                    col_idx,
                                    np.random.RandomState(
                                        random_state.randint(4294967296)))
                                for (tree_idx, X_oob)
                                in parallel_args)

                        cum_hazard_scores = []
                        oob_y = []
                        for idx in range(X.shape[0]):
                            num = 0.
                            den = 0.
                            for tree_idx2, (oob_mask, forward_map) \
                                    in enumerate(oob_masks):
                                if oob_mask[idx]:
                                    num += vimp_results[tree_idx2][
                                        forward_map[idx]].sum()
                                    den += 1
                            if den > 0:
                                cum_hazard_scores.append(num / den)
                                oob_y.append(y[idx])

                        if len(cum_hazard_scores) > 0:
                            cum_hazard_scores = np.array(cum_hazard_scores)
                            oob_y = np.array(oob_y)

                            vimp = self.oob_score_ - \
                                concordance_index(oob_y[:, 0],
                                                  -cum_hazard_scores,
                                                  oob_y[:, 1])
                        else:
                            vimp = np.nan
                        self.feature_importances_.append(vimp)
                    self.feature_importances_ \
                        = np.array(self.feature_importances_)

    # def pred_proba__tbd(self, test_df, time_list):
    #     """
    #     :param test_df: DataFrame
    #     :param time: checkpoint time to calculate probability on
    #     :return: the probability matrix. each row is the survival function for
    #     one test data point.
    #     """
    #     reduced_test_df = self._test_pca(test_df)

    #     if isinstance(time_list, int) or isinstance(time_list, float):
    #         time_list = [time_list]
    #     else:
    #         time_list = time_list

    #     return self.pred_proba_main(X=reduced_test_df.values,\
    #                                 times=np.array(time_list),\
    #                                 presorted_times=False)

    def pred_proba(self, X, times, presorted_times=False):
        """
        Computes the forest's survival probability function estimate for each
        feature vector evaluated at user-specified times.

        Parameters
        ----------
        X : 2D numpy array, shape = [n_samples, n_features]
            Feature vectors.

        times : 1D numpy array
            Times to compute the survival probability function at.

        presorted_times : boolean, optional (default=False)
            Flag for whether `times` is already sorted.

        Returns
        -------
        output : 2D numpy array
            Survival probability function evaluated at each of the times
            specified in `times` for each feature vector. The i-th row
            corresponds to the i-th feature vector.
        """
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_predict_tree)(self.trees[tree_idx], 'surv', X, times,
                                   presorted_times)
            for tree_idx in range(self.n_estimators))
        return functools.reduce(lambda x, y: x + y, results) \
            / self.n_estimators

    def predict_cum_haz(self, X, times, presorted_times=False):
        """
        Computes the forest's cumulative hazard function estimate for each
        feature vector evaluated at user-specified times.

        Parameters
        ----------
        X : 2D numpy array, shape = [n_samples, n_features]
            Feature vectors.

        times : 1D numpy array
            Times to compute the survival probability function at.

        presorted_times : boolean, optional (default=False)
            Flag for whether `times` is already sorted.

        Returns
        -------
        output : 2D numpy array
            Cumulative hazard function evaluated at each of the times
            specified in `times` for each feature vector. The i-th row
            corresponds to the i-th feature vector.
        """
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_predict_tree)(self.trees[tree_idx], 'cum_haz', X, times,
                                   presorted_times)
            for tree_idx in range(self.n_estimators))
        return functools.reduce(lambda x, y: x + y, results) \
            / self.n_estimators

    def predict(self, test_x, time_list):
        """
        :param test_df: DataFrame
        :param time_list: checkpoint time to calculate probability on
        :return: the list of median survival time and the probability matrix
        """
        proba_matrix = self.pred_proba(X=test_x,times=np.array(time_list),presorted_times=True)
        pred_medians = []
        median_time = np.max(time_list)
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

        return np.array(pred_medians), pd.DataFrame(np.transpose(proba_matrix), index=np.array(time_list))

    def _print_with_depth(self, string, depth):
        """
        Auxiliary function to print a string with indentation dependent on
        depth.
        """
        print("{0}{1}".format("    " * depth, string))

    def _print_tree(self, tree, current_depth=0):
        """
        Auxiliary function to print a survival tree.
        """
        if 'surv' in tree:
            self._print_with_depth(tree['times'], current_depth)
            return
        self._print_with_depth(
            "{0} > {1}".format(self.column_names[tree['feature']],
                               tree['threshold']),
            current_depth)
        self._print_tree(tree['left'], current_depth + 1)
        self._print_tree(tree['right'], current_depth + 1)

    def draw(self):
        """
        Prints out each tree of the random survival forest.
        """
        for tree_idx, tree in enumerate(self.trees):
            print("==========================================\nTree",
                  tree_idx)
            self._print_tree(tree)

