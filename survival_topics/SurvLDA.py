
from time import time
import sys, os, multiprocessing, traceback, pickle
import numpy as np
import pandas as pd
from numpy.random import RandomState
import scipy.io
import scipy.sparse as sparse
from scipy.special import logsumexp
from scipy.optimize import minimize
from scipy.special import digamma

from topic_models.survlda.survlda_helper import compute_updated_phi_i
from scipy.spatial.distance import euclidean, squareform, cdist

# from lifelines import CoxPHFitter
# import glmnet_python
# from glmnet import glmnet
# from glmnetCoef import glmnetCoef

from pycox.evaluation import EvalSurv

class SurvLDA():
    def __init__(self, n_topics=10, alpha0=1, max_iter=30, predict_max_iter=30, \
                 random_init_tau=False, seed=47, verbose=True):
        '''
        Uniform initialization of tau: topic-word distribution

        '''
        np.random.seed(seed)
        self.n_topics = int(n_topics)
        self.alpha0 = alpha0
        # alpha: hyperparameter for Dirichlet prior over topic distribution vectors
        self.alpha = np.full(self.n_topics, self.alpha0/self.n_topics, dtype=np.float64)
        # Cox regression coefficient beta (there are n_topics coefficients, the g-th
        # coefficient corresponds to how much we weight a subject's probability of belonging to the g-th topic)
        self.beta = np.zeros(self.n_topics, dtype=np.float64)
        self.random_init_tau = random_init_tau
        self.max_iter = max_iter # maximum number of iterations for EM
        self.predict_max_iter = predict_max_iter
        self.predict_convergence_threshold = 1e-9
        self.verbose = verbose

    def fit(self, train_x, train_y, feature_names):

        print(train_x)

        n_train = train_x.shape[0]
        val_row_ids = np.random.choice(n_train, int(n_train*0.2), replace=False)
        val_x = train_x[val_row_ids]
        val_y = train_y[val_row_ids]
        train_x = np.delete(train_x, val_row_ids, axis=0)
        train_y = np.delete(train_y, val_row_ids, axis=0)

        self._survLDA_train_variational_EM(word_count_matrix=train_x, 
            survival_or_censoring_times=train_y[:, 0], censoring_indicator_variables=train_y[:, 1],
            val_word_count_matrix=val_x, val_survival_or_censoring_times=val_y[:, 0],
            val_censoring_indicator_variables=val_y[:, 1])
 
    def predict(self, test_x, time_list):
        test_W, _ = self._word_count_matrix_to_word_vectors_phi(test_x)
        surv_estimates = []
        median_estimates = []
        for i in range(len(test_W)):
            preds = self._predict_survival(test_W[i])
            surv_estimates.append(preds[1])
            median_estimates.append(preds[0])
        surv_estimates = np.array(surv_estimates)
        median_estimates = np.array(median_estimates)

        return surv_estimates, pd.DataFrame(np.transpose(surv_estimates), index=self.sorted_unique_times)        

    def _predict_survival(self, w):
        '''
        Input:
        - w: document representation of a single subject
        - sorted_unique_times: used to predict survival function
        - log_baseline_hazard: used to compute cox cumulative hazard

        '''
        sorted_unique_times = self.sorted_unique_times
        log_baseline_hazard = self.log_h0_discretized

        N_i = w.shape[0]
        phi = np.full((N_i, self.n_topics), 1./self.n_topics, dtype=np.float64)
        prev_z = phi.sum(axis=0).copy()

        for it in range(self.predict_max_iter):
            gamma = self.alpha + phi.sum(axis=0)
            psi = digamma(gamma) - digamma(gamma.sum())
            xi = np.zeros((N_i, self.n_topics))
            for j in range(0, N_i):
                word = int(w[j])
                for k in range(0, self.n_topics):
                    if self.tau[k][word] > 0:
                        xi[j][k] = np.log(self.tau[k][word])
                    else:
                        xi[j][k] = -np.inf
            phi = np.exp(psi + xi)
            phi /= phi.sum(axis=1)[:, np.newaxis]
            z = 1./N_i * phi.sum(axis=0)
            log_hazard = log_baseline_hazard + np.inner(self.beta, z)
            z_diff = np.linalg.norm(prev_z - z)
            if z_diff < self.predict_convergence_threshold:
                break
            prev_z = z.copy()

        # compute cumulative hazard --copied code from survival_evaluate
        # should this be done in each iteration or just once at end?
        survival_probas = []
        log_cumulatives = []
        assert(len(log_hazard) == len(sorted_unique_times))
        for idx2 in range(len(log_hazard)):
            log_cumulative = logsumexp(log_hazard[:idx2 + 1])
            survival_probas.append(np.exp(-np.exp(log_cumulative)))
            log_cumulatives.append(log_cumulative)

        # estimate median survival time
        t_inf = np.inf
        t_sup = 0   
        log_minus_log_half = np.log(-np.log(0.5))
        for sorted_time_idx, t in enumerate(sorted_unique_times):
            if log_minus_log_half <= log_cumulatives[sorted_time_idx]:
                if t < t_inf:
                    t_inf = t
            if log_minus_log_half >= log_cumulatives[sorted_time_idx]:
                if t > t_sup:
                    t_sup = t

        if t_inf == np.inf:
            median_LoS_estimate = t_sup
        else:
            median_LoS_estimate = 0.5 * (t_inf + t_sup)
        # print("done")
        return median_LoS_estimate, survival_probas

    def _survLDA_train_variational_EM(self, word_count_matrix,
                                     survival_or_censoring_times,
                                     censoring_indicator_variables,
                                     val_word_count_matrix,
                                     val_survival_or_censoring_times,
                                     val_censoring_indicator_variables):
        """
        Input:
        - K: number of topics
        - alpha0: hyperparameter for Dirichlet prior

        Output:
        - tau: 2D numpy array; g-th row is for g-th topic and consists of word
            distribution for that topic
        - beta: 1D numpy array with length equal to the number of topics
            (Cox regression coefficients)
        - h0_reformatted: 2D numpy array; first row is sorted unique times and
            second row is the discretized version of log(h0)
        - gamma: 2D numpy array: i-th row is variational parameter for Dirichlet
            distribution for i-th subject (length is number of topics)
        - phi: list of length given by number of subjects; i-th element is a
            2D numpy array with number of rows given by the number of words in
            i-th subject's document and number of columns given by number of
            topics (variational parameter for how much each word of each subject
            belongs to different topics)
        - rmse: estimated median survival time RMSE computed using validation data
        - mae: estimated median survival time MAE computed using validation data
        - cindex: concordance index computed using validation data
        - stop_iter: interation number after stopping
        """
        # *********************************************************************
        # 1. SET UP TOPIC MODEL PARAMETERS BASED ON TRAINING DATA

        self.word_count_matrix = word_count_matrix
        num_subjects, num_words = self.word_count_matrix.shape # train size and vocabulary size
        # the length of each patient's document (i.e. sum of all words including duplicates)
        doc_length = self.word_count_matrix.sum(axis=1).astype(np.int64)

        # tau tells us what the word distribution is per topic
        # - each row corresponds to a topic
        # - each row is a probability distribution, which we initialize to be a
        #   uniform distribution across the words (so each entry is 1 divided by `num_words`
        if self.random_init_tau:
            self.tau = np.random.rand(self.n_topics, num_words)
            self.tau /= self.tau.sum(axis=1)[:, np.newaxis] # normalize tau
        else:
            self.tau = np.full((self.n_topics, num_words), 1. / num_words, dtype=np.float64)

        # variational distribution parameter gamma tells us the Dirichlet
        # distribution parameter for the topic distribution specific to each subject; 
        # we can initialize this to be all ones corresponding to a uniform distribution prior
        self.gamma = np.ones((num_subjects, self.n_topics), dtype=np.float64)

        # variational distribution parameter phi tells us the probabilities of
        # each subject's words coming from each of the K different topics; we can
        # initialize these to be uniform over topics (1/K)
        self.W, self.phi = self._word_count_matrix_to_word_vectors_phi(word_count_matrix)
        val_W, _ = self._word_count_matrix_to_word_vectors_phi(val_word_count_matrix)

        # *********************************************************************
        # 2. SET UP COX BASELINE HAZARD FUNCTION

        # Cox baseline hazard function h0 can be represented as a finite 1D vector
        death_counter = {}
        for t in survival_or_censoring_times:
            if t not in death_counter:
                death_counter[t] = 1
            else:
                death_counter[t] += 1
        sorted_unique_times = np.sort(list(death_counter.keys()))
        self.sorted_unique_times = sorted_unique_times

        num_unique_times = len(sorted_unique_times)
        self.log_h0_discretized = np.zeros(num_unique_times)
        for r, t in enumerate(sorted_unique_times):
            self.log_h0_discretized[r] = np.log(death_counter[t])

        log_H0 = []
        for r in range(num_unique_times):
            log_H0.append(logsumexp(self.log_h0_discretized[:(r + 1)]))
        log_H0 = np.array(log_H0)

        time_map = {t: r for r, t in enumerate(sorted_unique_times)}
        time_order = np.array([time_map[t] for t in survival_or_censoring_times], dtype=np.int64)

        # *********************************************************************
        # 3. EM MAIN LOOP

        pool = multiprocessing.Pool(4)
        stop_iter = 0
        val_censoring_indicator_variables = val_censoring_indicator_variables.astype(np.bool)
        for EM_iter_idx in range(self.max_iter):
            # ------------------------------------------------------------------
            # E-step (update gamma, phi; uses helper variables psi, xi)
            if self.verbose:
                print('[Variational EM iteration %d]' % (EM_iter_idx + 1))
                print('  Running E-step...', end='', flush=True)
            tic = time()

            # update gamma (dimensions: `num_subjects` by `K`)
            for i, phi_i in enumerate(self.phi):
                self.gamma[i] = self.alpha + phi_i.sum(axis=0)

            # compute psi (dimensions: `num_subjects` by `K`)
            psi = digamma(self.gamma) - digamma(self.gamma.sum(axis=1))[:, np.newaxis]
            # update phi, this normalizes already
            self.phi = pool.map(SurvLDA._compute_updated_phi_i_pmap_helper,
                                [(i, self.phi[i], psi[i], self.W[i], self.tau, self.beta,
                                  np.exp(log_H0[time_order[i]]), censoring_indicator_variables[i],
                                  doc_length[i], self.n_topics) for i in range(num_subjects)])

            toc = time()
            if self.verbose:
                print(' Done. Time elapsed: %f second(s).' % (toc - tic), flush=True)

            # --------------------------------------------------------------------
            # M-step (update tau, beta, h0, H0; uses helper variable phi_bar)
            if self.verbose:
                print('  Running M-step...', end='', flush=True)
            tic = time()

            # update tau
            tau = np.zeros((self.n_topics, num_words), dtype=np.float64)
            for i in range(num_subjects):
                for j in range(doc_length[i]):
                    word = self.W[i][j]
                    for k in range(self.n_topics):
                        tau[k][word] += self.phi[i][j, k]
            # normalize tau
            self.tau /= tau.sum(axis=1)[:, np.newaxis]

            phi_bar = np.zeros((num_subjects, self.n_topics), dtype=np.float64)
            for i in range(num_subjects):
                phi_bar[i] = self.phi[i].sum(axis=0) / doc_length[i]

            y_r = np.vstack((survival_or_censoring_times, censoring_indicator_variables)).T
            beta_phi_bar = np.dot(phi_bar, self.beta)
            log_h0_discretized_ = np.zeros(num_unique_times)
            log_H0_ = []
            for r, t in enumerate(sorted_unique_times):
                R_t = np.where(survival_or_censoring_times >= t, True, False)
                log_h0_discretized_[r] = np.log(death_counter[t]) - logsumexp(beta_phi_bar[R_t])
                log_H0_.append(logsumexp(log_h0_discretized_[:(r + 1)]))

            # update beta
            def obj_fun(beta_):
                beta_phi_bar_ = np.dot(phi_bar, beta_)
                fun_val = np.dot(censoring_indicator_variables, beta_phi_bar_)
                # finally, we add in the third term
                for i in range(num_subjects):
                    product_terms = np.dot(self.phi[i], np.exp(beta_ / doc_length[i]))
                    fun_val -= np.exp(log_H0_[time_order[i]]
                                      + np.sum(np.log(product_terms)))
                return -fun_val

            self.beta = minimize(obj_fun, self.beta).x

            # compute dot product <beta, phi_bar[i]> for each patient i
            # (resulting in a 1D array of length `num_subjects`)
            beta_phi_bar = np.dot(phi_bar, self.beta)
            # print("phi bar is \n", phi_bar)

            # update h0
            for r, t in enumerate(sorted_unique_times):
                R_t = np.where(survival_or_censoring_times >= t, True, False)
                self.log_h0_discretized[r] = np.log(death_counter[t]) - logsumexp(beta_phi_bar[R_t])
            log_H0 = []
            for r in range(num_unique_times):
                log_H0.append(logsumexp(self.log_h0_discretized[:(r + 1)]))
            log_H0 = np.array(log_H0)

            toc = time()
            if self.verbose:
                print(' Done. Time elapsed: %f second(s).' % (toc - tic), flush=True)
                print('  Beta:', self.beta)

            # convergence criteria to decide whether to break out of the for loop early
            surv_estimates = []
            median_estimates = []
            for i in range(len(val_W)):
                preds = self._predict_survival(val_W[i])
                surv_estimates.append(preds[1])
                median_estimates.append(preds[0])
            surv_estimates = np.array(surv_estimates)
            median_estimates = np.array(median_estimates)

            val_ev = EvalSurv(pd.DataFrame(np.transpose(surv_estimates), index=sorted_unique_times), 
                              val_survival_or_censoring_times, 
                              val_censoring_indicator_variables, 
                              censor_surv='km')
            val_cindex = val_ev.concordance_td('antolini')

            val_rmse = np.sqrt(np.mean((median_estimates[val_censoring_indicator_variables] - \
                            val_survival_or_censoring_times[val_censoring_indicator_variables])**2))
            val_mae = np.mean(np.abs(median_estimates[val_censoring_indicator_variables] - \
                            val_survival_or_censoring_times[val_censoring_indicator_variables]))

            if self.verbose:
                print('  Validation set survival time c-index: %f' % val_cindex)
                print('  Validation set survival time rmse: %f' % val_rmse)
                print('  Validation set survival time mae: %f' % val_mae)

            stop_iter += 1

        pool.close()
        if self.verbose:
            print('  EM algorithm completes training at iteration ', stop_iter)

        return

    def _compute_updated_phi_i_pmap_helper(args):
        i, phi_i_old, psi_i, W_i, tau, beta, H0_at_doc_i_observed_time, \
                doc_i_recorded_or_censored, N_i, K \
            = args
        # print('\n    Updating phi for subject %d (number of words: %d)...'
        #       % (i, N_i), end='', flush=True)
        log_phi_i = np.log(compute_updated_phi_i(phi_i_old, psi_i, W_i, tau, beta,
                                                  H0_at_doc_i_observed_time,
                                                  doc_i_recorded_or_censored, N_i, K))
        phi_i = np.zeros((N_i, K))
        for j in range(N_i):
            phi_i[j] = np.exp(log_phi_i[j] - logsumexp(log_phi_i[j]))
        return phi_i

    def _word_count_matrix_to_word_vectors_phi(self, word_count_matrix):
        # create the document representation used by Dawson and Kendziorski. 
        # W is indexed by patient id and contains a list indexed by position
        # with value = word
        W = []
        phi = []
        num_words = word_count_matrix.shape[1]
        doc_length = word_count_matrix.sum(axis=1).astype(np.int64)
        for i, row in enumerate(word_count_matrix):
            N_i = doc_length[i]
            words = np.zeros(N_i, dtype=np.int64)
            pos = 0
            for column in range(num_words):
                val = int(row[column])
                for a in range(val):
                    words[pos] = column
                    pos += 1
            W.append(words)
            phi.append(np.full((N_i, self.n_topics), 1./self.n_topics, dtype=np.float64))
        return W, phi











