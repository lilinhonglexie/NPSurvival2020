import keras.backend as K
from keras import models
from keras import layers
from keras import regularizers
from collections import Counter
from scipy.special import logsumexp
import numpy as np
from sklearn.preprocessing import StandardScaler


class NeuralNetworkCox:

    def __init__(self, first_layer_size, epochs=1500, lmbda=0., alpha=1.,
                 verbose=0):
        """
        The Neural Network Cox Proportional Model's core codes (generate betas)
        are written by George Chen in
        Elastic-net regularized Cox proportional hazards with Keras.ipynb.
        This class provides a wrapper with standard interfaces. The hazard
        calculation is identical with that of the Cox Proportional Model.

        :param first_layer_size: The size of the first dense layer. This layer
        is used for dimensional reduction.
        :param epochs: Number of epochs
        :param lmbda: The parameter for regularization for the second layer
        :param alpha: The parameter for the weight of Lasso and Ridge. by
        default alpha = 1, which means we only use Lasso regularization
        :param verbose:
        """
        self.lmbda = lmbda
        self.alpha = alpha
        self.epochs = epochs
        self.first_layer_size = first_layer_size
        self.verbose = verbose
        self.standard_scaler = StandardScaler()

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

    def fit(self, train_df, duration_col='LOS', event_col='OUT'):
        """
        :param train_df: DataFrame, with the duration and the event column
        :param duration_col: the column name for duration
        :param event_col: the column name for event
        """
        self._duration_col = duration_col
        self._event_col = event_col
        train_x = train_df.drop(columns=[duration_col, event_col]).values
        train_y = train_df[[duration_col, event_col]].values
        x_standardized = self.standard_scaler.fit_transform(train_x)

        def coxph_partial_log_likelihood_batch(y_true, y_pred, batch_size):
            # y_pred in this context consists of each feature vector dotted with
            #  beta, with a 1 padded
            y_observed_times = y_true[:, 0]
            y_event_indicators = y_true[:, 1]

            R_batch = K.cast(K.greater_equal(
                K.repeat_elements(
                    K.expand_dims(y_observed_times, axis=0),
                    batch_size, 0
                ),
                K.repeat_elements(
                    K.expand_dims(y_observed_times, axis=-1),
                    batch_size, -1
                )
            ), 'float32')

            x_transpose_beta = y_pred[:, 0]
            return -K.mean(
                (
                    x_transpose_beta -
                    K.log(K.flatten(K.dot(
                        R_batch, K.expand_dims(K.exp(x_transpose_beta), axis=-1)
                    )))
                ) * y_event_indicators)

        # yes, the code works even when batch size is not the full dataset
        batch_size = len(train_x)
        coxph_partial_log_likelihood = lambda y_true, y_pred: \
                coxph_partial_log_likelihood_batch(y_true, y_pred, batch_size)

        l1_weight = self.lmbda * self.alpha
        l2_weight = self.lmbda * (1 - self.alpha) / 2.
        coxph_neural = models.Sequential()
        coxph_neural.add(
            layers.Dense(self.first_layer_size, activation=None,
                         input_shape=(x_standardized.shape[1],),
                         use_bias=True)
        )
        coxph_neural.add(
            layers.Dense(1, activation=None,
                         input_shape=(self.first_layer_size,),
                         kernel_regularizer=regularizers.L1L2(
                             l1_weight,
                             l2_weight
                         ),
                         use_bias=False)
        )
        coxph_neural.add(
            layers.Lambda(lambda x: K.concatenate([x, K.ones_like(x)], axis=-1))
        )

        if self.verbose:
            coxph_neural.summary()
        coxph_neural.compile(optimizer='Adam',
                             loss=coxph_partial_log_likelihood)
        coxph_neural.fit(x_standardized, train_y,
                         epochs=self.epochs, batch_size=batch_size,
                         verbose=self.verbose)

        self.dense_params = coxph_neural.get_weights()[0]
        self.bias = coxph_neural.get_weights()[1].flatten()
        self.betas = coxph_neural.get_weights()[2].flatten()

        observed_times = train_y[:, 0]
        event_indicators = train_y[:, 1]
        transformed_train_x = x_standardized.dot(self.dense_params) + self.bias

        # For each observed time, how many times the event occurred
        event_counts = Counter()
        for t, r in zip(observed_times, event_indicators):
            event_counts[t] += int(r)

        # Sorted list of observed times
        self.sorted_unique_times = np.sort(list(event_counts.keys()))
        self.num_unique_times = len(self.sorted_unique_times)
        self.log_baseline_hazard = np.zeros(self.num_unique_times)

        # Calculate the log baseline hazard
        for time_idx, t in enumerate(self.sorted_unique_times):
            logsumexp_args = []
            for subj_idx, observed_time in enumerate(observed_times):
                if observed_time >= t:
                    logsumexp_args.append(
                        np.inner(self.betas, transformed_train_x[subj_idx]))
            if event_counts[t] > 0:
                self.log_baseline_hazard[time_idx] \
                    = np.log(event_counts[t]) - logsumexp(logsumexp_args)
            else:
                self.log_baseline_hazard[time_idx] \
                    = -np.inf - logsumexp(logsumexp_args)

    def pred_median_time(self, test_df):
        """
        Given the beta and baseline hazard, for each test datapoint, we can
        calculate the hazard function. Based on that, we find the time when
        the survival probability is around 0.5.

        :param test_df: DataFrame
        :return: the list of median survival time for each test datapoint.
        """

        test_x = \
            test_df.drop(columns=[self._duration_col, self._event_col]).values
        # test_x = self.mean_remover.transform(test_x)
        test_x = self.standard_scaler.transform(test_x)
        # manually perform the first layer's dimension reduction
        test_x = test_x.dot(self.dense_params) + self.bias

        num_test_subjects = len(test_x)
        median_survival_times = np.zeros(num_test_subjects)

        # log(-log(0.5)) -> instead of calculating the survival function based
        # on the hazard function and then finding the time point near the Pr of
        # 0.5, we actually change the question into: find the timepoint when
        # the log cumulative hazard is near log(-log(0.5)).
        log_minus_log_half = np.log(-np.log(0.5))

        for subj_idx in range(num_test_subjects):
            # log hazard of the given object
            log_hazard = self.log_baseline_hazard + \
                         np.inner(self.betas, test_x[subj_idx])
            # simulate the integral to get the log cumulative hazard function
            log_cumulative_hazard = np.zeros(self.num_unique_times)
            for time_idx in range(self.num_unique_times):
                log_cumulative_hazard[time_idx] = \
                    logsumexp(log_hazard[:time_idx + 1])
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
            median_survival_times[subj_idx] = 0.5 * (t_inf + t_sup)

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
        test_x = \
            test_df.drop(columns=[self._duration_col, self._event_col]).values
        test_x = self.standard_scaler.transform(test_x)
        # manually perform the first layer's dimension reduction
        test_x = test_x.dot(self.dense_params) + self.bias

        # the pred_matrix would be a matrix of n * k, where n is the number of
        # test datapoints, and k is the number of unique observed time from the
        # train dataset
        pred_matrix = []
        for subject_x in test_x:
            # log hazard of the given object
            log_hazard = \
                self.log_baseline_hazard + np.inner(self.betas, subject_x)
            survival_proba = np.zeros(self.num_unique_times)
            for time_idx in range(self.num_unique_times):
                # log cumulative hazard at this time point
                log_cumulative_hazard = logsumexp(log_hazard[:time_idx + 1])
                # the corresponding probability
                survival_proba[time_idx] = \
                    np.exp(-np.exp(log_cumulative_hazard))
            pred_matrix.append(survival_proba)

        # Using the mapping between our input time_list to the train dataset
        # time_list.
        time_indice = [self._find_nearest_time_index(cur_time)
                       for cur_time in time_list]

        # the proba_matrix would be a matrix of n * m, where n is the number of
        # test datapoints, and m is the number of unique time we want to
        # estimate on (i.e. len(time))
        proba_matrix = np.zeros((len(test_x), len(time_indice)))
        for row in range(len(test_x)):
            for col, time_index in enumerate(time_indice):
                proba_matrix[row][col] = pred_matrix[row][time_index]

        return proba_matrix

    def predict(self, test_df, time_list):
        """
        Given the beta and baseline hazard, for each test datapoint, we can
        calculate the hazard function, and then calculate the survival function.
        Then it should be easy to find the median survival time.

        :param test_df: DataFrame
        :param time_list: checkpoint time to calculate probability on
        :return: the list of median survival time and the probability matrix
        """
        proba_matrix = self.pred_proba(test_df, time_list)
        pred_medians = []
        median_time = 0
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

        return np.array(pred_medians), proba_matrix

