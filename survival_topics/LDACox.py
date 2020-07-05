
import pickle
import numpy as np
import pandas as pd

from sklearn.decomposition import LatentDirichletAllocation
import glmnet_python
from glmnet import glmnet
from glmnetCoef import glmnetCoef

from scipy.special import logsumexp, softmax
from collections import Counter

import time
from progressbar import ProgressBar
# from multiprocessing import Pool

# def predict_survival_function_par(args):
#     subject_x, par_pred_dict_1 = args
#     log_hazard = par_pred_dict_1["log_baseline_hazard"] + np.inner(par_pred_dict_1["beta"], subject_x)
#     survival_proba = np.zeros(par_pred_dict_1["num_unique_times"])
#     for time_idx in range(par_pred_dict_1["num_unique_times"]):
#         # log cumulative hazard at this time point
#         log_cumulative_hazard = logsumexp(log_hazard[:time_idx + 1])
#         # the corresponding probability
#         survival_proba[time_idx] = np.exp(-np.exp(log_cumulative_hazard))
#     return survival_proba

# def predict_median_survival_time_par(args):
#     survival_proba, time_list = args
#     median_time = max(time_list)
#     # if the predicted proba never goes below 0.5, predict the largest seen value
#     # the survival_proba is in descending order
#     for col, proba in enumerate(survival_proba):
#         if proba > 0.5:
#             continue
#         if proba == 0.5 or col == 0:
#             median_time = time_list[col]
#         else:
#             median_time = (time_list[col - 1] + time_list[col]) / 2
#         break
    
#     return median_time

class LDACox(object):
    '''
    Topic based model baseline
    Default: no regularization for the cox model

    '''
    def __init__(self, n_topics, cox_alpha=1, cox_lambda=1e-5, seed=47):
        self.n_topics = int(n_topics)
        self.seed = seed
        self._alpha = cox_alpha
        self._lambda = cox_lambda

    def fit(self, train_x, train_y, feature_names):
        self._feature_names = feature_names

        print("Start fitting LDA...")
        tic = time.time()
        self.lda = LatentDirichletAllocation(n_components=self.n_topics, learning_method='online', \
                                             random_state=self.seed, n_jobs=8)
        thetas = self.lda.fit_transform(train_x) # train_x: n_samples * n_features --> thetas: n_samples * n_topics
        toc = time.time()
        print("Finish fitting LDA... time spent {} seconds.".format(toc-tic))

        # Find beta. Modified from George's demo.
        print("Start fitting CoxPH...")
        tic = time.time()
        fit = glmnet(x=thetas.copy(), y=train_y.copy(),
                     family='cox', alpha=self._alpha, standardize=False, # we performed our own standardization
                     intr=False)
        self.beta = glmnetCoef(fit, s=np.array([self._lambda])).flatten()
        toc = time.time()
        print("Finish fitting CoxPH... time spent {} seconds.".format(toc-tic))
 
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

        # In the lazy version of predict, there's no need to fit the baseline hazard
        # # Calculate the log baseline hazard. Implemented by George.
        # for time_idx, t in enumerate(self.sorted_unique_times):
        #     logsumexp_args = []
        #     for subj_idx, observed_time in enumerate(observed_times):
        #         if observed_time >= t:
        #             logsumexp_args.append(
        #                 np.inner(self.beta, thetas[subj_idx]))
        #     if event_counts[t] > 0:
        #         self.log_baseline_hazard[time_idx] = \
        #             np.log(event_counts[t]) - logsumexp(logsumexp_args)
        #     else:
        #         self.log_baseline_hazard[time_idx] = \
        #             -np.inf - logsumexp(logsumexp_args)

    def predict(self, test_x, time_list=None, parallel="prediction"):
        """
        Given the beta and baseline hazard, for each test datapoint, we can
        calculate the hazard function, and then calculate the survival function.

        :param test_df: DataFrame
        :param time_list: checkpoint time to calculate probability on
        :return: the probability matrix. each row is the survival function for
        one test datapoint.
        """
        test_thetas = self.lda.transform(test_x)

        if parallel == "prediction":

            par_pred_dict_1 = dict()
            par_pred_dict_1["log_baseline_hazard"] = self.log_baseline_hazard
            par_pred_dict_1["beta"] = self.beta
            par_pred_dict_1["num_unique_times"] = self.num_unique_times

            tic = time.time()
            print(">>>> In progress: predicting median survival time...")
            par_pred_input_1 = [(subject_x, par_pred_dict_1) for subject_x in test_thetas]
            predict_pool = Pool(processes=None)
            tmp_proba = predict_pool.map(predict_survival_function_par, par_pred_input_1)
            tmp_proba = np.array(tmp_proba, dtype="float32")

            # If you'd like to allow estimated time points to be different from the training set's
            # uncomment the below chunk, otherwise, we save this step for now
            proba_matrix = tmp_proba

            # # Using the mapping between our input time_list to the train dataset
            # # time_list.
            # time_indices = [self._find_nearest_time_index(cur_time)
            #                 for cur_time in time_list]

            # # the proba_matrix would be a matrix of n * m, where n is the number of
            # # test datapoints, and m is the number of unique time we want to
            # # estimate on (i.e. len(time))
            # proba_matrix = np.zeros((len(test_thetas), len(time_indices)))
            # for row in range(len(test_thetas)):
            #     for col, time_index in enumerate(time_indices):
            #         proba_matrix[row][col] = tmp_proba[row][time_index]

            par_pred_input_2 = [(survival_proba, time_list) for survival_proba in proba_matrix]
            pred_medians = predict_pool.map(predict_median_survival_time_par, par_pred_input_2)
            pred_medians = np.array(pred_medians, dtype="float32")

            predict_pool.close()
            predict_pool.join()
            toc = time.time()
            print(">>>> Time spent with parallelism: {} seconds".format(toc-tic))

            return pred_medians, pd.DataFrame(np.transpose(proba_matrix), index=np.array(time_list))

        else:
            tic = time.time()
            print(">>>> In progress: predicting median survival time...")

            # the tmp_proba would be a matrix of n * k, where n is the number of
            # test datapoints, and k is the number of unique observed time from the
            # train dataset
            tmp_proba = []
            for subject_x in test_thetas:
                # log hazard of the given object
                log_hazard = \
                    self.log_baseline_hazard + np.inner(self.beta, subject_x)
                survival_proba = np.zeros(self.num_unique_times)
                for time_idx in range(self.num_unique_times):
                    # log cumulative hazard at this time point
                    log_cumulative_hazard = logsumexp(log_hazard[:time_idx + 1])
                    # the corresponding probability
                    survival_proba[time_idx] = \
                        np.exp(-np.exp(log_cumulative_hazard))
                tmp_proba.append(survival_proba)

            # If you'd like to allow estimated time points to be different from the training set's
            # uncomment the below chunk, otherwise, we save this step for now
            proba_matrix = tmp_proba

            # # Using the mapping between our input time_list to the train dataset
            # # time_list.
            # time_indices = [self._find_nearest_time_index(cur_time)
            #                 for cur_time in time_list]

            # # the proba_matrix would be a matrix of n * m, where n is the number of
            # # test datapoints, and m is the number of unique time we want to
            # # estimate on (i.e. len(time))
            # proba_matrix = np.zeros((len(test_thetas), len(time_indices)))
            # for row in range(len(test_thetas)):
            #     for col, time_index in enumerate(time_indices):
            #         proba_matrix[row][col] = tmp_proba[row][time_index]

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

            toc = time.time()
            print(">>>> Time spent without parallelism: {} seconds".format(toc-tic))

            return np.array(pred_medians), pd.DataFrame(np.transpose(proba_matrix), index=np.array(time_list))

    def predict_lazy(self, test_x, time_list=None, parallel="none"):
        test_thetas = self.lda.transform(test_x)
        neg_hazards = []
        print("Entering prediction...")
        tic = time.time()
        for subject_x in test_thetas:
            # log hazard of the given object
            curr_hazard = np.inner(self.beta, subject_x)
            neg_hazards.append(-1 * curr_hazard)
        toc = time.time()
        print("Exiting prediction..., time spent: {} seconds.".format(toc-tic))

        return np.array(neg_hazards), None, None

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

    def beta_explain(self, feature_names, save_path):
        '''
        Output: 
        N * P topic model + feature names 

        '''
        survival_topic_model = dict()
        survival_topic_model['topic_distributions'] = np.array([row / row.sum() for row in self.lda.components_])
        survival_topic_model['beta'] = self.beta
        survival_topic_model['vocabulary'] = np.array(feature_names)

        with open(save_path, 'wb') as pkg_write:
            pickle.dump(survival_topic_model, pkg_write)

        print(" >>> Survival topic model saved to " + save_path)

        # feature_names = np.array(feature_names)
        
        # sorted_vocab_dists_by_topics = []
        # for i in range(self.n_topics):
        #     curr_vocab_dist = topic_distributions[i]
        #     curr_sorted_indices_by_dist = np.argsort(-curr_vocab_dist).astype(np.int)
        #     curr_word_list_with_probs = list(zip(feature_names[curr_sorted_indices_by_dist], curr_vocab_dist[curr_sorted_indices_by_dist]))
        #     sorted_vocab_dists_by_topics.append(curr_word_list_with_probs)

        # if save_path is not None:
        #     shap_test_pkg = dict()
        #     shap_test_pkg['n_topics'] = self.n_topics
        #     shap_test_pkg['beta'] = self.beta
        #     shap_test_pkg['vocab_dists_sorted'] = sorted_vocab_dists_by_topics

            














