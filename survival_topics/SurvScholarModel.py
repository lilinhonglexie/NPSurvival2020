
from collections import Counter
# import gensim
import lifelines, pickle, sys
import numpy as np
import pandas as pd
from scipy.special import logsumexp, softmax

import warnings  
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf
    from tensorflow.python.ops import array_ops
    slim = tf.contrib.slim

print('future warnings ignored') 

import shap

# for development
# from lifelines.utils import concordance_index
# from pycox.evaluation import EvalSurv
from progressbar import ProgressBar
import time
from multiprocessing import Pool
from tqdm import tqdm
# import matplotlib.pyplot as plt

def load_data_mimic(feature_vectors, labels, feature_names, col_sel=None):
    # feature_vectors: 2D numpy array where rows index different data points, columns index features
    # labels: 2D numpy array where the first column is observed times and the second column is indicators (0 = censored, 1 = event of interested happened)

    X = feature_vectors.astype(dtype='float32')
    n, d = X.shape

    if col_sel is None:
        col_sel = np.ones(d, dtype=np.bool)

    X = X[:, col_sel]
    new_feature_names = [feature_names[w] for w in range(d) if col_sel[w]]

    y = labels.astype(dtype='float32') if labels is not None else None
    label_names = ['observed time', 'event indicator']
    label_type = 'survival'

    covariates = None
    covariate_names = None
    covariates_type = None

    return X, new_feature_names, y, label_names, label_type, \
           covariates, covariate_names, covariates_type, col_sel

def get_init_bg(data):
    """
    Compute the log background frequency of all words
    """
    sums = np.sum(data, axis=0)+1.0
    # print("Computing background frequencies")
    # print("Min/max word counts in training data: %d %d" % (int(np.min(sums)), int(np.max(sums))))
    bg = np.array(np.log(sums) - np.log(float(np.sum(sums))), dtype=np.float32)
    return bg

def make_network(dv, n_topics, survival_layers, survival_loss_weight, encoder_layers=2, \
                embedding_dim=300, encoder_shortcut=False, label_type=None, n_labels=2, \
                label_emb_dim=0, covariate_type=None, n_covariates=0, covar_emb_dim=0, \
                use_covar_interactions=False, covars_in_survival_regression=True,\
                survival_layers_num_nodes=None, survival_layers_keep_prob=1.0):
    """
    Combine the network configuration parameters into a dictionary
    """
    tf.reset_default_graph()
    network_architecture = \
        dict(encoder_layers=encoder_layers,
             encoder_shortcut=encoder_shortcut,
             embedding_dim=embedding_dim,
             n_topics=n_topics,
             dv=dv,
             label_type=label_type,
             n_labels=n_labels,
             label_emb_dim=label_emb_dim,
             covariate_type=covariate_type,
             n_covariates=n_covariates,
             covar_emb_dim=covar_emb_dim,
             use_covar_interactions=use_covar_interactions,
             survival_layers=survival_layers,
             covars_in_survival_regression=covars_in_survival_regression,
             survival_loss_weight=survival_loss_weight,
             survival_layers_num_nodes=survival_layers_num_nodes,
             survival_layers_keep_prob=survival_layers_keep_prob
             )
    return network_architecture

def train(model, network_architecture, X, Y, C, batch_size, training_epochs=50, display_step=50, min_weights_sq=1e-7, regularize=False, X_dev=None, Y_dev=None, C_dev=None, bn_anneal=True, init_eta_bn_prop=1.0, rng=None, verbose=False, early_stopping=False):

    n_train, dv = X.shape
    mb_gen = create_minibatch(X, Y, C, batch_size=batch_size, rng=rng)

    # This machine only runs non linear experiments
    # These number are hard-coded with respect to the specific datasets we have
    # We are using 200 epochs for UNOS (>60K total samples), and 500 epochs for all other datasets (<10K total samples)
    if n_train > 10000:
        training_epochs = 25
    elif n_train < 1000:
        training_epochs = 100
    else:
        training_epochs = 50

    if early_stopping:
        assert(X_dev is not None)
        dev_mb_gen = create_minibatch(X_dev, Y_dev, C_dev, batch_size=batch_size, rng=rng)
        time_list = list(np.sort(np.unique(Y[:, 0].astype('float32'))))

        # train_losses_for_plot = []
        # train_cls_losses_for_plot = []
        # dev_losses_for_plot = []
        # dev_cls_losses_for_plot = []
    else:
        assert(X_dev is None)
        dev_mb_gen = None
        time_list = None

    dv = network_architecture['dv']
    n_topics = network_architecture['n_topics']

    total_batch = int(n_train / batch_size)

    # create np arrays to store regularization strengths, which we'll update outside of the tensorflow model
    if regularize:
        l2_strengths = 0.5 * np.ones([n_topics, dv]) / float(n_train)
        l2_strengths_c = 0.5 * np.ones([model.beta_c_length, dv]) / float(n_train)
        l2_strengths_ci = 0.5 * np.ones([model.beta_ci_length, dv]) / float(n_train)
    else:
        l2_strengths = np.zeros([n_topics, dv])
        l2_strengths_c = np.zeros([model.beta_c_length, dv])
        l2_strengths_ci = np.zeros([model.beta_ci_length, dv])

    batches = 0

    eta_bn_prop = init_eta_bn_prop  # interpolation between batch norm and no batch norm in final layer of recon
    kld_weight = 1.0  # could use this to anneal KLD, but not currently doing so

    # Training cycle
    for epoch in range(training_epochs):
        avg_loss = 0.
        avg_cls_loss = 0.
        accuracy = 0.
        # Loop over all batches
        for i in range(total_batch):
            # get a minibatch
            batch_xs, batch_ys, batch_cs = next(mb_gen)
            # do one update, passing in the data, regularization strengths, and bn
            loss, cls_loss = model.fit(batch_xs, batch_ys, batch_cs, l2_strengths=l2_strengths, l2_strengths_c=l2_strengths_c, l2_strengths_ci=l2_strengths_ci, eta_bn_prop=eta_bn_prop, kld_weight=kld_weight)
            # # compute accuracy on minibatch
            # if network_architecture['n_labels'] > 0:
            #     accuracy += np.sum(pred == np.argmax(batch_ys, axis=1)) / float(n_train)
            # Compute average loss
            avg_loss += loss / n_train * batch_size
            avg_cls_loss += cls_loss / n_train * batch_size
            batches += 1
            if np.isnan(avg_loss):
                print(epoch, i, np.sum(batch_xs, 1).astype(np.int), batch_xs.shape)
                print('Encountered NaN, stopping training. Please check the learning_rate settings and the momentum.')
                # return vae,emb
                sys.exit()

        # update weight prior variances using current weight values
        if regularize:
            weights = model.get_weights()
            weights_sq = weights ** 2
            # avoid infinite regularization
            weights_sq[weights_sq < min_weights_sq] = min_weights_sq
            l2_strengths = 0.5 / weights_sq / float(n_train)

            if network_architecture['n_covariates'] > 0:
                weights = model.get_covar_weights()
                weights_sq = weights ** 2
                # avoid infinite regularization
                weights_sq[weights_sq < min_weights_sq] = min_weights_sq
                l2_strengths_c = 0.5 / weights_sq / float(n_train)
                if network_architecture['use_covar_interactions']:
                    weights = model.get_covar_inter_weights()
                    weights_sq = weights ** 2
                    # avoid infinite regularization
                    weights_sq[weights_sq < min_weights_sq] = min_weights_sq
                    l2_strengths_ci = 0.5 / weights_sq / float(n_train)

        # Display logs per epoch step
        if epoch % display_step == 0 and epoch > 0 and verbose:
            if network_architecture['n_labels'] > 0:
                print("Epoch:", '%d' % epoch, "; loss =", "{:.9f}".format(avg_loss), "; survival loss =", "{:.9f}".format(avg_cls_loss))
            else:
                print("Epoch:", '%d' % epoch, "loss=", "{:.9f}".format(avg_loss))

            # if X_dev is not None:
            #     dev_perplexity = evaluate_perplexity(model, X_dev, Y_dev, C_dev, eta_bn_prop=eta_bn_prop)
            #     n_dev, _ = X_dev.shape
            #     if network_architecture['n_labels'] > 0:
            #         raise Exception("This part hasn't been properly implemented yet.")
            #         # dev_predictions = predict_labels(model, X_dev, C_dev, eta_bn_prop=eta_bn_prop)
            #         # dev_accuracy = float(np.sum(dev_predictions == np.argmax(Y_dev, axis=1))) / float(n_dev)
            #         # print("Epoch: %d; Dev perplexity = %0.4f; Dev accuracy = %0.4f" % (epoch, dev_perplexity, dev_accuracy))
            #     else:
            #         print("Epoch: %d; Dev perplexity = %0.4f" % (epoch, dev_perplexity))

        # anneal eta_bn_prop from 1 to 0 over the course of training
        if bn_anneal:
            if eta_bn_prop > 0:
                eta_bn_prop -= 1.0 / float(training_epochs*0.75)
                if eta_bn_prop < 0:
                    eta_bn_prop = 0.0

        if early_stopping and epoch > 0 and epoch % 100 == 0:

            raise Exception("Early stopping is not supported with a feasible run-time, use fixed number of epochs for now.")

            # # Uncomment the following block to get validation losses
            # n_dev, _ = X_dev.shape
            # # evaluate loss on held-out development set, in batches
            # avg_dev_loss = 0.
            # avg_dev_cls_loss = 0.
            # dev_total_batch = int(n_dev / batch_size)

            # for i in range(dev_total_batch):
            #     # get a minibatch
            #     dev_batch_xs, dev_batch_ys, dev_batch_cs = next(dev_mb_gen)
            #     # do one update, passing in the data, regularization strengths, and bn
            #     dev_theta_input = np.zeros([batch_size, n_topics]).astype('float32')
            #     dev_loss, dev_cls_loss = model.fit(dev_batch_xs, dev_batch_ys, dev_batch_cs, l2_strengths=l2_strengths, l2_strengths_c=l2_strengths_c, l2_strengths_ci=l2_strengths_ci, eta_bn_prop=eta_bn_prop, kld_weight=kld_weight, is_training=False, keep_prob=1)

            #     # # compute accuracy on minibatch
            #     # if network_architecture['n_labels'] > 0:
            #     #     accuracy += np.sum(pred == np.argmax(batch_ys, axis=1)) / float(n_train)
            #     # Compute average loss

            #     avg_dev_loss += dev_loss / n_dev * batch_size
            #     avg_dev_cls_loss += dev_cls_loss / n_dev * batch_size
            #     if np.isnan(avg_dev_loss):
            #         print('Encountered NaN, stopping training. Please check the learning_rate settings and the momentum.')
            #         # return vae,emb
            #         sys.exit()

            # # # Uncomment the following block to get validation cindex : note: it's time-consuming

            # model.fit_baseline_hazard(X, Y)

            # print("Finished fitting baseline hazards...")

            # _, _, dev_pred_times, dev_pred_surv_functions = model.predict(X_dev, None)
            # print("Finished making predictions on dev set...")

            # dev_pred_surv_functions = pd.DataFrame(np.transpose(dev_pred_surv_functions), index=np.array(time_list))
            # dev_ev = EvalSurv(dev_pred_surv_functions, Y_dev[:, 0], Y_dev[:, 1].astype(np.bool), censor_surv='km')
            # dev_cindex_new = dev_ev.concordance_td('antolini')
            # dev_cindex_old = concordance_index(Y_dev[:, 0], np.array(dev_pred_times), Y_dev[:, 1].astype(np.bool)) 
            # print("Finished dev set evaluation...")

            # # print("avg_dev_loss", avg_dev_loss)
            # # print("avg_dev_cls_loss", avg_dev_cls_loss)
            # print("dev_cindex_new", dev_cindex_new)
            # print("dev_cindex_old", dev_cindex_old)

            # train_losses_for_plot.append(avg_loss)
            # train_cls_losses_for_plot.append(avg_cls_loss)
            # dev_losses_for_plot.append(dev_cindex_new)
            # dev_cls_losses_for_plot.append(dev_cindex_old)
    
    # plt.plot(train_losses_for_plot, label="Training Loss")
    # plt.plot(train_cls_losses_for_plot, label="Training Survival Loss")
    # plt.plot(dev_losses_for_plot, label="Dev cindex new")
    # plt.plot(dev_cls_losses_for_plot, label="Dev cindex old")
    # plt.title("Scholar Cindex Plot")
    # plt.legend()
    # plt.savefig("UNOS_Cindex_Plot_small_lr.png")
    # sys.exit(0)

    return model

def create_minibatch(X, Y, C, batch_size, rng=None):
    """
    Split data into minibatches
    """
    while True:
        # Return random data samples of a size 'minibatch_size' at each iteration
        if rng is not None:
            # ixs = np.arange(X.shape[0])
            # rng.shuffle(ixs)
            # assert(len(ixs) == batch_size)
            ixs = rng.randint(X.shape[0], size=batch_size)
        else:
            ixs = np.random.randint(X.shape[0], size=batch_size)
        if Y is not None and C is not None:
            yield X[ixs, :].astype('float32'), Y[ixs, :].astype('float32'), C[ixs, :].astype('float32')
        elif Y is not None:
            yield X[ixs, :].astype('float32'), Y[ixs, :].astype('float32'), None
        elif C is not None:
            yield X[ixs, :].astype('float32'), None, C[ixs, :].astype('float32')
        else:
            yield X[ixs, :].astype('float32'), None, None

# def predict_labels_and_evaluate_new(model, train_X, train_labels, test_X, test_labels, test_covariates, vocab, n_top_word=20):
#     '''
#     This new version of model evaludation computers: RMSE, MAE, C-index, Unique top words, coherence, lift
#     The input variables are also modified to contain the minimum set of variables needed.

#     '''

#     # RMSE, MAE, concordance ###################################################

#     n_items, vocab_size = test_X.shape
#     predicted_median_survival_times, predicted_survival_functions = predict_labels(model, test_X, test_covariates)
#     true_observed_times = test_labels[:, 0]
#     true_event_indicators = test_labels[:, 1].astype(np.bool)
#     if true_event_indicators.sum() > 0:
#         difference_over_events = predicted_median_survival_times[true_event_indicators]\
#                                              - true_observed_times[true_event_indicators]
#         rmse = np.sqrt(np.mean((difference_over_events)**2))
#         mae = np.mean(np.abs(difference_over_events))
#     else:
#         rmse = 0.
#         mae = 0.
#     cindex = lifelines.utils.concordance_index(true_observed_times,
#                                                predicted_median_survival_times,
#                                                true_event_indicators)

#     # Average unique top words, coherence, lift ################################

#     corpus = np.vstack((train_X, test_X))
#     top_words = []
#     coherence_scores = []
#     lift_scores = []

#     # compute topic specific vocabulary distributions
#     background_log_freq = model.get_bg()
#     topic_deviations = model.get_weights()
#     topic_distributions = topic_deviations + background_log_freq
#     topic_distributions = softmax(topic_distributions, axis=1)
#     cox_beta = model.sess.run(model.survival_weights).flatten()

#     n_topics = topic_deviations.shape[0]
#     assert(n_topics == cox_beta.shape[0])
#     for i in range(n_topics):
#         if cox_beta[i] != 0:
#             curr_vocab_deviation = topic_deviations[i]
#             curr_vocab_dist = topic_distributions[i]
#             # top words sorted by deviation, probabilities using distributions
#             curr_sorted_indices = np.argsort(-curr_vocab_deviation)
#             curr_top_words_list = curr_sorted_indices[:n_top_word]
#             # print("word distribution", curr_vocab_dist, curr_top_words_list)
#             coherence_scores.append(compute_coherence(corpus, curr_top_words_list))
#             lift_scores.append(compute_lift(corpus, curr_top_words_list, curr_vocab_dist))
#             curr_top_words_set = set(curr_top_words_list)
#             top_words.append(curr_top_words_set)

#     n_nonzero_topics = len(top_words)
#     assert(n_nonzero_topics == len(coherence_scores) == len(lift_scores))

#     n_uniques = []
#     for i, topic_i in enumerate(top_words):
#         other_tops = set()
#         for j, topic_j in enumerate(top_words):
#             if i != j: other_tops.union(topic_j)
#         n_uniques.append(len(topic_i - other_tops))

#     assert(len(n_uniques) == n_nonzero_topics)

#     avg_n_unique = sum(n_uniques)/n_nonzero_topics
#     avg_coherence = sum(coherence_scores)/n_nonzero_topics
#     avg_lift = sum(lift_scores)/n_nonzero_topics

#     # print(rmse, mae, cindex, n_uniques, coherence_scores, lift_scores)

#     return {'rmse': rmse, 'mae': mae, 'concordance': cindex,
#             'unique_top':avg_n_unique, 'coherence':avg_coherence, 'lift':avg_lift}

# def predict_labels_and_evaluate(model, X, Y, C, train_Y, train_unique_times_sorted, output_dir=None, subset='train'):
#     """
#     Predict labels for all instances using the classifier network and evaluate the accuracy

#     This old version is discarded now. It does not compute metrics specific to 
#     topic based models.
#     """

#     # RMSE, MAE, concordance ###################################################

#     n_items, vocab_size = X.shape
#     predicted_median_survival_times, predicted_survival_functions = predict_labels(model, X, C)
#     true_observed_times = Y[:, 0]
#     true_event_indicators = Y[:, 1].astype(np.bool)
#     if true_event_indicators.sum() > 0:
#         difference_over_events = predicted_median_survival_times[true_event_indicators]\
#                                              - true_observed_times[true_event_indicators]
#         rmse = np.sqrt(np.mean((difference_over_events)**2))
#         mae = np.mean(np.abs(difference_over_events))
#     else:
#         rmse = 0.
#         mae = 0.
#     cindex = lifelines.utils.concordance_index(true_observed_times,
#                                                predicted_median_survival_times,
#                                                true_event_indicators)

#     # IPEC - removed ###########################################################

#     # train_labels_df = pd.DataFrame(data=train_Y, columns=['LOS', 'OUT'] )
#     # ipec_score_list = calc_ipec_list(predicted_survival_functions, Y[:,0], Y[:,1],\
#     #                                  train_unique_times_sorted, train_labels_df)
#     # time_horizon_i = int(len(train_unique_times_sorted) * 0.8)  
#     # assert(time_horizon_i == int(len(ipec_score_list) * 0.8))       
#     # ipec_score = ipec_score_list[time_horizon_i]/train_unique_times_sorted[time_horizon_i]
#     # # ipec_score = 0

#     # # save the results to file
#     # if output_dir is not None:
#     #     fh.write_list_to_text([str(rmse), str(mae), str(cindex), str(ipec_score)], \
#     #                         os.path.join(output_dir, 'accuracy.' + subset + '.txt'))

#     return {'rmse': rmse, 'mae': mae, 'concordance': cindex}

# def predict_labels(model, X, C, eta_bn_prop=0.0):
#     """
#     Predict a label for each instance using the classifier part of the network
#     """
#     theta, survival_inner_prod, predicted_median_survival_times, predicted_survival_functions = \
#             model.predict(X, C, eta_bn_prop=eta_bn_prop, \
#                 predict_median_survival_times=True, predict_survival_function=True)
#     return predicted_median_survival_times, predicted_survival_functions


# def calc_scores(model, train_feature_vectors, test_feature_vectors, test_labels, n_top_word=20):
#     '''
#     Metrics calculated: RMSE, MAE, concordance, 
#                         average unique top words (n=20), coherence, lift

#     '''
#     # RMSE, MAE, concordance ###################################################

#     test_archetype_weights = predict_weights(test_feature_vectors, model["archetypes"])
#     test_predicted_survival_times, test_predicted_proba_matrix = \
#         predict_median_survival_times_and_proba(test_archetype_weights, \
#                                       model["train_archetype_weights"], model["train_labels"], \
#                                       model["cox_beta"], average_to_get_median=True)

#     pred_obs_differences = (test_predicted_survival_times - test_labels[:, 0])[test_labels[:,1].astype(np.bool)]
#     assert(len(pred_obs_differences) == sum(test_labels[:, 1]))
#     rmse = np.sqrt(np.mean(pred_obs_differences**2))
#     mae = np.mean(np.abs(pred_obs_differences))

#     concordance = concordance_index(test_labels[:,0], test_predicted_survival_times,
#                                     test_labels[:,1])

#     # IPEC - removed ###########################################################

#     # train_df = pd.DataFrame(data = model["train_feature_vectors"], columns = model["feature_names"])
#     # train_df["LOS"] = model["train_labels"][:, 0]
#     # train_df["OUT"] = model["train_labels"][:, 1]
#     # ipec_score_list = calc_ipec_list(test_predicted_proba_matrix, \
#     #                                  labels[:, 0], labels[:, 1], \
#     #                                  model["unique_times"], train_df)

def compute_coherence(corpus, top_words):
    n = len(top_words)
    total = 0
    for i, w_i in enumerate(top_words):
        n_i = sum(corpus[:, w_i] != 0)
        assert(n_i != 0)
        for j in range(i+1, n):
            n_ij = sum(np.logical_and(corpus[:, w_i] != 0, corpus[:, top_words[j]] != 0))
            assert(n_ij <= n_i)
            total += np.log((n_ij + 1)/n_i)
    return total

def compute_lift(corpus, top_words, beta):
    base = np.sum(corpus)
    total = 0
    n = len(top_words)
    for i, word in enumerate(top_words):
        emp_prob = np.sum(corpus[:, word])/base
        total += np.log(beta[word]/emp_prob)
        #print(emp_prob, beta[word])
    return total/n

class SurvivalScholarTrainer_Linear(object):
    '''
    Training wrapper for Survival Scholar, Tensorflow implementation

    '''
    def __init__(self, survival_loss_weight, batch_size, n_topics=5, survival_layers=0, seed=47, saved_model=None):
        '''
        :param n_topics: number of topics in the topic model
        :param survival_layers: 0: linear survival regression, 1: with one softplus nonlinearity before survival regression
        :param survival_loss_weight: weight of the survival loss in training the network

        '''
        self.n_topics = int(n_topics)
        self.survival_layers = survival_layers
        self.survival_loss_weight = 10**survival_loss_weight
        self.batch_size = int(batch_size)
        # Other parameters are not tuned

        if seed is not None:
            self.rng = np.random.RandomState(seed)
            np.random.seed(seed)
            self.seed = seed
        else:
            raise Exception("Did not set seed.")
            self.rng = np.random.RandomState(np.random.randint(0, 100000))

        self.saved_model = saved_model

    def fit(self, train_x, train_y, feature_names):
        self.feature_names = feature_names
        # format data
        train_X, _, train_labels, label_names, label_type, train_covariates, _, _, _ = \
                                            load_data_mimic(train_x, train_y, self.feature_names)
        n_train, dv = train_X.shape # dv: vocab size
        _, n_labels = train_labels.shape

        # initialize the background using overall word frequencies
        self.init_bg = get_init_bg(train_X)
        # combine the network configuration parameters into a dictionary
        self.network_architecture = make_network(dv=dv, n_topics=self.n_topics, 
                                                 survival_layers=self.survival_layers, 
                                                 survival_loss_weight=self.survival_loss_weight,
                                                 label_type=label_type, n_labels=n_labels)

        # create the model
        tf.reset_default_graph()
        self.model = SurvivalScholar(network_architecture=self.network_architecture, batch_size=self.batch_size, 
                                     init_bg=self.init_bg, seed=self.seed,
                                     load_model_filename_prefix=self.saved_model)

        if self.saved_model is None:
            # train the model
            self.model = train(self.model, self.network_architecture, 
                               train_X, train_labels, train_covariates, 
                               batch_size=self.batch_size, rng=self.rng, verbose=False)

            self.model.fit_baseline_hazard(train_X, train_labels)
        else:
            self.model.load_baseline_hazard(self.saved_model)

    def predict(self, test_x, time_list=None, parallel="prediction"):

        if time_list is None:
            time_list = self.time_list
        else:
            self.time_list = time_list

        test_X, _, test_labels, label_names, label_type, test_covariates, _, _, _ = \
                                            load_data_mimic(test_x, None, self.feature_names)
        theta, survival_inner_prod, predicted_median_survival_times, predicted_survival_functions = \
            self.model.predict(test_X, test_covariates, parallel=parallel)
        return np.array(predicted_median_survival_times), pd.DataFrame(np.transpose(predicted_survival_functions), index=np.array(time_list))

    def predict_lazy(self, test_x, time_list, parallel="none"):

        test_X, _, test_labels, label_names, label_type, test_covariates, _, _, _ = \
                                            load_data_mimic(test_x, None, self.feature_names)
        theta, survival_inner_prod, predicted_median_survival_times, predicted_survival_functions = \
            self.model.predict(test_X, test_covariates, predict_median_survival_times=False, predict_survival_function=False, parallel=parallel)
        self.theta = theta
        return -1 * survival_inner_prod, None, None

    def save_to_disk(self, output_filename_prefix):
        self.model.save_to_disk(output_filename_prefix)

    def close_sess(self):
        self.model.close_session()

    def kernel_explain(self, train_x, test_x, feature_names, summary_k=100, save_path=None):
        '''
        Uses the kernel explainer from the shap package to explain each topic's influence on prediction

        '''
        # theta dim: np.zeros((n, self.network_architecture['n_topics'])

        train_theta, _, _, _, = self.model.predict(X=train_x, C=None, \
                predict_median_survival_times=False, predict_survival_function=False)

        test_theta, _, _, _ = self.model.predict(X=test_x, C=None, \
                predict_median_survival_times=False, predict_survival_function=False)

        def f(theta):
            '''
            model : function or iml.Model
            User supplied function that takes a matrix of samples (# samples x # features) and
            computes a the output of the model for those samples. The output can be a vector
            (# samples) or a matrix (# samples x # model outputs).

            '''
            return self.model.get_survival_inner_prod(theta) 

        train_theta_summary = shap.kmeans(train_theta, summary_k)
        print("Entering kernel explain...")

        explainer = shap.KernelExplainer(f, train_theta_summary)
        print("Finished fitting shap kernel explainer...")

        shap_values = explainer.shap_values(test_theta, nsamples="auto", l1_reg="aic") # nsamples="auto"

        if save_path is not None:

            shap_test_pkg = dict() # info for plotting locally
            shap_test_pkg["explainer_expected_values"] = explainer.expected_value
            shap_test_pkg["shap_values"] = shap_values
            shap_test_pkg["feature_names"] = feature_names
            shap_test_pkg["readme"] = "Using 10 means training data summary as background, auto perterbation samples for kernel explainer"
            shap_test_pkg["test_theta"] = test_theta
            shap_test_pkg["train_theta_summary"] = train_theta_summary

            with open(save_path, 'wb') as pkg_write:
                pickle.dump(shap_test_pkg, pkg_write)

    def beta_explain(self, feature_names, save_path):
        '''
        Use the beta coefficients to explain topic influence on median survival time prediction.

        '''
        # compute topic specific vocabulary distributions
        background_log_freq = self.model.get_bg()
        topic_deviations = self.model.get_weights()
        topic_distributions = topic_deviations + background_log_freq
        topic_distributions = softmax(topic_distributions, axis=1)
        cox_beta = self.model.sess.run(self.model.survival_weights).flatten()

        survival_topic_model = dict()
        survival_topic_model['topic_distributions'] = topic_distributions
        survival_topic_model['beta'] = cox_beta
        survival_topic_model['vocabulary'] = np.array(feature_names)

        survival_topic_model['topic_deviations'] = topic_deviations
        survival_topic_model['background_log_freq'] = background_log_freq

        with open(save_path, 'wb') as pkg_write:
            pickle.dump(survival_topic_model, pkg_write)

        print(" >>> Survival topic model saved to " + save_path)

        # feature_names = np.array(feature_names)
        # corpus = np.vstack((train_X, test_X))
        # top_words = []
        # coherence_scores = []
        # lift_scores = []

        # # compute topic specific vocabulary distributions
        # background_log_freq = self.model.get_bg()
        # topic_deviations = self.model.get_weights()
        # topic_distributions = topic_deviations + background_log_freq
        # topic_distributions = softmax(topic_distributions, axis=1)
        # cox_beta = self.model.sess.run(self.model.survival_weights).flatten()

        # n_topics = topic_deviations.shape[0]
        # assert(n_topics == cox_beta.shape[0])

        # sorted_vocab_dists_by_topics = []
        # sorted_vocab_deviations_by_topics = []

        # for i in range(n_topics):
        #     # if cox_beta[i] != 0:
        #     curr_vocab_deviation = topic_deviations[i]
        #     curr_vocab_dist = topic_distributions[i]
        #     # # top words sorted by deviation, probabilities using distributions
        #     # curr_sorted_indices = np.argsort(-abs(curr_vocab_deviation))
        #     curr_sorted_indices = np.argsort(-1*(curr_vocab_deviation))
        #     curr_top_words_list = curr_sorted_indices[:n_top_word].astype(np.int)
        #     sorted_vocab_deviations_by_topics.append(list(zip(feature_names[curr_sorted_indices], curr_vocab_deviation[curr_sorted_indices])))
 
        #     curr_sorted_indices_by_dist = np.argsort(-curr_vocab_dist).astype(np.int)
        #     curr_word_list_with_probs = list(zip(feature_names[curr_sorted_indices_by_dist], curr_vocab_dist[curr_sorted_indices_by_dist]))
        #     sorted_vocab_dists_by_topics.append(curr_word_list_with_probs)
        #     # print("Sorted Vocabs by Topic-specific Probabilities:", curr_word_list_with_probs[:n_top_word])

        #     coherence_scores.append(compute_coherence(corpus, curr_top_words_list))
        #     lift_scores.append(compute_lift(corpus, curr_top_words_list, curr_vocab_dist))
        #     curr_top_words_set = set(curr_top_words_list)
        #     top_words.append(curr_top_words_set)

        # n_nonzero_topics = len(top_words) # let's look at all topics now
        # assert(n_nonzero_topics == len(coherence_scores) == len(lift_scores))

        # n_uniques = []
        # for i, topic_i in enumerate(top_words):
        #     other_tops = set()
        #     for j, topic_j in enumerate(top_words):
        #         if i != j: other_tops.union(topic_j)
        #     n_uniques.append(len(topic_i - other_tops))

        # # print(n_uniques, coherence_scores, lift_scores)

        # if save_path is not None:

        #     shap_test_pkg = dict()
        #     shap_test_pkg['n_topics'] = n_topics
        #     shap_test_pkg['beta'] = cox_beta
        #     shap_test_pkg['coherence_scores'] = coherence_scores
        #     shap_test_pkg['lift_scores'] = lift_scores
        #     shap_test_pkg['vocab_dists_sorted'] = sorted_vocab_dists_by_topics
        #     shap_test_pkg['vocab_deviations_sorted'] = sorted_vocab_deviations_by_topics

        #     with open(save_path, 'wb') as pkg_write:
        #         pickle.dump(shap_test_pkg, pkg_write)

class SurvivalScholarTrainer_NonLinear(object):
    '''
    Training wrapper for Survival Scholar, Tensorflow implementation

    '''
    def __init__(self, n_topics, survival_loss_weight, batch_size, survival_layers, survival_layers_size, survival_layers_dropout, seed=47, saved_model=None):
        '''
        :param n_topics: number of topics in the topic model
        :param survival_layers: 0: linear survival regression, 1: with one softplus nonlinearity before survival regression
        :param survival_loss_weight: weight of the survival loss in training the network

        '''
        self.n_topics = int(n_topics)
        self.survival_layers = int(survival_layers)
        self.survival_loss_weight = 10**survival_loss_weight
        self.batch_size = int(batch_size)

        self.survival_layers_dropout = survival_layers_dropout
        self.survival_layers_num_nodes = [] # survival layers input size = n_topics
        for layer_i in range(self.survival_layers):
            self.survival_layers_num_nodes.append(int(survival_layers_size))
        # self.survival_layers_num_nodes.append(self.n_topics)

        # Other parameters are not tuned

        if seed is not None:
            self.rng = np.random.RandomState(seed)
            np.random.seed(seed)
            self.seed = seed
        else:
            raise Exception("Did not set seed.")
            self.rng = np.random.RandomState(np.random.randint(0, 100000))

        self.saved_model = saved_model

    def fit(self, train_x, train_y, feature_names):
        self.feature_names = feature_names
        # format data
        train_X, _, train_labels, label_names, label_type, train_covariates, _, _, _ = \
                                            load_data_mimic(train_x, train_y, self.feature_names)

        # # Split the train into train and dev, 1000 is an arbitrary number 
        # dev_row_ids = np.random.choice(train_X.shape[0], 1000, replace=False)
        # dev_X = train_X[dev_row_ids]
        # dev_labels = train_labels[dev_row_ids]
        # dev_covariates = train_covariates # None
        # train_X = np.delete(train_X, dev_row_ids, axis=0)
        # train_labels = np.delete(train_labels, dev_row_ids, axis=0)

        n_train, dv = train_X.shape # dv: vocab size
        _, n_labels = train_labels.shape
        # initialize the background using overall word frequencies
        self.init_bg = get_init_bg(train_X)
        # combine the network configuration parameters into a dictionary
        self.network_architecture = make_network(dv=dv, n_topics=self.n_topics, 
                                                 survival_layers=self.survival_layers, 
                                                 survival_loss_weight=self.survival_loss_weight,
                                                 label_type=label_type, n_labels=n_labels,
                                                 survival_layers_num_nodes=self.survival_layers_num_nodes,
                                                 survival_layers_keep_prob=1-self.survival_layers_dropout)

        # create the model
        tf.reset_default_graph()
        self.model = SurvivalScholar(network_architecture=self.network_architecture, batch_size=self.batch_size, 
                                     init_bg=self.init_bg, seed=self.seed,
                                     load_model_filename_prefix=self.saved_model)

        if self.saved_model is None:
            # train the model
            self.model = train(self.model, self.network_architecture, 
                               X=train_X, Y=train_labels, C=train_covariates, batch_size=self.batch_size, 
                               rng=self.rng, verbose=True)

            self.model.fit_baseline_hazard(train_X, train_labels)
        else:
            self.model.load_baseline_hazard(self.saved_model)

    def predict(self, test_x, time_list=None, parallel="prediction"):

        if time_list is None:
            time_list = self.time_list
        else:
            self.time_list = time_list

        test_X, _, test_labels, label_names, label_type, test_covariates, _, _, _ = \
                                            load_data_mimic(test_x, None, self.feature_names)
        theta, survival_inner_prod, predicted_median_survival_times, predicted_survival_functions = \
            self.model.predict(test_X, test_covariates, parallel=parallel)
        return np.array(predicted_median_survival_times), pd.DataFrame(np.transpose(predicted_survival_functions), index=np.array(time_list))

    def save_to_disk(self, output_filename_prefix):
        self.model.save_to_disk(output_filename_prefix)

    def close_sess(self):
        self.model.close_session()

    def kernel_explain(self, train_x, test_x, feature_names, summary_k=10, save_path=None):
        '''
        Uses the kernel explainer from the shap package to explain each topic's influence on prediction

        '''
        # theta dim: np.zeros((n, self.network_architecture['n_topics'])

        train_theta, _, _, _, = self.model.predict(X=train_x, C=None, \
                predict_median_survival_times=False, predict_survival_function=False)

        test_theta, _, _, _ = self.model.predict(X=test_x, C=None, \
                predict_median_survival_times=False, predict_survival_function=False)

        def f(theta):
            '''
            model : function or iml.Model
            User supplied function that takes a matrix of samples (# samples x # features) and
            computes a the output of the model for those samples. The output can be a vector
            (# samples) or a matrix (# samples x # model outputs).

            '''
            return self.model.get_survival_inner_prod(theta) 

        train_theta_summary = shap.kmeans(train_theta, summary_k)
        print("Entering kernel explain...")

        explainer = shap.KernelExplainer(f, train_theta_summary)
        print("Finished fitting shap kernel explainer...")

        shap_values = explainer.shap_values(test_theta, nsamples="auto", l1_reg="aic") # nsamples="auto"

        if save_path is not None:

            shap_test_pkg = dict() # info for plotting locally
            shap_test_pkg["explainer_expected_values"] = explainer.expected_value
            shap_test_pkg["shap_values"] = shap_values
            shap_test_pkg["feature_names"] = feature_names
            shap_test_pkg["readme"] = "Using 10 means training data summary as background, auto perterbation samples for kernel explainer"
            shap_test_pkg["test_theta"] = test_theta
            shap_test_pkg["train_theta_summary"] = train_theta_summary

            with open(save_path, 'wb') as pkg_write:
                pickle.dump(shap_test_pkg, pkg_write)

    def beta_explain(self, train_X, test_X, feature_names, n_top_word=20, save_path=None):
        '''
        Use the beta coefficients to explain topic influence on median survival time prediction.

        '''
        feature_names = np.array(feature_names)
        corpus = np.vstack((train_X, test_X))
        top_words = []
        coherence_scores = []
        lift_scores = []

        # compute topic specific vocabulary distributions
        background_log_freq = self.model.get_bg()
        topic_deviations = self.model.get_weights()
        topic_distributions = topic_deviations + background_log_freq
        topic_distributions = softmax(topic_distributions, axis=1)
        cox_beta = self.model.sess.run(self.model.survival_weights).flatten()

        n_topics = topic_deviations.shape[0]
        # assert(n_topics == cox_beta.shape[0])

        sorted_vocab_dists_by_topics = []
        sorted_vocab_deviations_by_topics = []

        for i in range(n_topics):
            # if cox_beta[i] != 0:
            curr_vocab_deviation = topic_deviations[i]
            curr_vocab_dist = topic_distributions[i]
            # # top words sorted by deviation, probabilities using distributions
            curr_sorted_indices = np.argsort(abs(curr_vocab_deviation))
            curr_top_words_list = curr_sorted_indices[:n_top_word].astype(np.int)
            sorted_vocab_deviations_by_topics.append(list(zip(feature_names[curr_sorted_indices], curr_vocab_deviation[curr_sorted_indices])))
 
            curr_sorted_indices_by_dist = np.argsort(-curr_vocab_dist).astype(np.int)
            curr_word_list = feature_names[curr_sorted_indices_by_dist]
            curr_word_list_with_probs = list(zip(curr_word_list, -1.*np.sort(-curr_vocab_dist)))
            sorted_vocab_dists_by_topics.append(curr_word_list_with_probs)
            # print("Sorted Vocabs by Topic-specific Probabilities:", curr_word_list_with_probs[:n_top_word])

            coherence_scores.append(compute_coherence(corpus, curr_top_words_list))
            lift_scores.append(compute_lift(corpus, curr_top_words_list, curr_vocab_dist))
            curr_top_words_set = set(curr_top_words_list)
            top_words.append(curr_top_words_set)

        n_nonzero_topics = len(top_words) # let's look at all topics now
        assert(n_nonzero_topics == len(coherence_scores) == len(lift_scores))

        n_uniques = []
        for i, topic_i in enumerate(top_words):
            other_tops = set()
            for j, topic_j in enumerate(top_words):
                if i != j: other_tops.union(topic_j)
            n_uniques.append(len(topic_i - other_tops))

        # print(n_uniques, coherence_scores, lift_scores)

        if save_path is not None:

            shap_test_pkg = dict()
            shap_test_pkg['n_topics'] = n_topics
            shap_test_pkg['beta'] = cox_beta
            shap_test_pkg['coherence_scores'] = coherence_scores
            shap_test_pkg['lift_scores'] = lift_scores
            shap_test_pkg['vocab_dists_sorted'] = sorted_vocab_dists_by_topics
            shap_test_pkg['vocab_deviations_sorted'] = sorted_vocab_deviations_by_topics

            with open(save_path, 'wb') as pkg_write:
                pickle.dump(shap_test_pkg, pkg_write)

def predict_median_survival_times_par(args):

    i, others_dict = args
    log_minus_log_half = -0.366512920581664347619010868584155105054378509521484375

    log_hazard = others_dict['log_baseline_hazard'] + others_dict['survival_inner_prod'][i] + others_dict['bias']
    log_cumulative_hazard = np.zeros(others_dict['num_unique_times'])
    for time_idx in range(others_dict['num_unique_times']):
        log_cumulative_hazard[time_idx] \
            = logsumexp(log_hazard[:time_idx + 1])

    t_inf = np.inf
    t_sup = 0.
    for time_idx, t in enumerate(others_dict['hazard_sorted_unique_times']):
        if log_minus_log_half <= log_cumulative_hazard[time_idx]:
            if t < t_inf:
                t_inf = t
        if log_minus_log_half >= log_cumulative_hazard[time_idx]:
            if t > t_sup:
                t_sup = t

    if t_inf == np.inf:
        return t_sup
    else:
        return 0.5 * (t_inf + t_sup)

def predicted_survival_functions_par(args):

    i, others_dict = args
    log_hazard = others_dict['log_baseline_hazard'] + others_dict['survival_inner_prod'][i] + others_dict['bias']
    log_cumulative_hazard = np.zeros(others_dict['num_unique_times'])
    for time_idx in range(others_dict['num_unique_times']):
        log_cumulative_hazard[time_idx] \
            = logsumexp(log_hazard[:time_idx + 1])
    curr_survival_probs = np.exp(-np.exp(log_cumulative_hazard))
    return curr_survival_probs

def fit_baseline_hazard_par(args):
    t, others_dict = args

    logsumexp_args = []
    for subj_idx, observed_time in enumerate(others_dict['observed_times']):
        if observed_time >= t:
            logsumexp_args.append(others_dict['survival_inner_prod'][subj_idx] + others_dict['bias'])
    if others_dict['event_counts'][t] > 0:
        return np.log(others_dict['event_counts'][t]) - logsumexp(logsumexp_args)
    else:
        return -np.inf - logsumexp(logsumexp_args)

class SurvivalScholar(object):
    """
    Scholar: a neural model for documents with metadata

    WARNING: how this is coded up, the batch size critically must stay the same

    """
    def __init__(self, network_architecture, batch_size, alpha=1.0,
                 learning_rate=0.002, init_embeddings=None, update_embeddings=True,
                 init_bg=None, update_background=True, init_beta=None, update_beta=True,
                 threads=8, regularize=False, optimizer='adam',
                 adam_beta1=0.99, seed=None, cox_elastic_net_lmbda=0., cox_elastic_net_alpha=1.,
                 cox_use_bias=False, load_model_filename_prefix=None):
        """
        :param network_architecture: a dictionary of model configuration parameters (see run_scholar_tf.py)
        :param alpha: hyperparameter for Dirichlet prior on documents (scalar or np.array)
        :param learning_rate:
        :param batch_size: default batch size
        :param init_embeddings: np.array of word vectors to be used in the encoder (optional)
        :param update_embeddings: if False, do not update the word embeddings used in the encoder
        :param init_bg: vector of weights to iniatialize the background term (optional)
        :param update_background: if False, do not update the weights of the background term
        :param init_beta: initial topic-word weights (optional)
        :param update_beta: if False, do not update topic-word weights
        :param threads: limit computation to this many threads (seems to be doubled in practice)
        :param regularize: if True, apply adaptive L2 regularizatoin
        :param optimizer: optimizer to use [adam|sgd|adagrad]
        :param adam_beta1: beta1 parameter for Adam optimizer
        :param seed: random seed (optional)
        """

        if seed is not None:
            tf.set_random_seed(seed)

        self.network_architecture = network_architecture
        self.survival_loss_weight = network_architecture['survival_loss_weight']
        self.learning_rate = learning_rate
        self.adam_beta1 = adam_beta1

        self.cox_elastic_net_lmbda = cox_elastic_net_lmbda
        self.cox_elastic_net_alpha = cox_elastic_net_alpha
        self.cox_use_bias = cox_use_bias

        self.l1_weight = cox_elastic_net_lmbda * cox_elastic_net_alpha
        self.l2_weight = cox_elastic_net_lmbda * (1.-cox_elastic_net_alpha) / 2.

        n_topics = network_architecture['n_topics']
        n_labels = network_architecture['n_labels']
        n_covariates = network_architecture['n_covariates']
        covar_emb_dim = network_architecture['covar_emb_dim']
        use_covar_interactions = network_architecture['use_covar_interactions']
        dv = network_architecture['dv']

        assert n_labels == 2

        self.regularize = regularize

        # create placeholders for covariates l2 penalties
        self.beta_c_length = 0      # size of embedded covariates
        self.beta_ci_length = 0     # size of embedded covariates * topics
        if n_covariates > 0:
            if covar_emb_dim > 0:
                self.beta_c_length = covar_emb_dim
            else:
                self.beta_c_length = n_covariates
        if use_covar_interactions:
            self.beta_ci_length = self.beta_c_length * n_topics

        self.l2_strengths = tf.placeholder(tf.float32, [n_topics, dv], name="l2_strengths")
        self.l2_strengths_c = tf.placeholder(tf.float32, [self.beta_c_length, dv], name="l2_strengths_c")
        self.l2_strengths_ci = tf.placeholder(tf.float32, [self.beta_ci_length, dv], name="l2_strengths_ci")

        # create placeholders for runtime options
        self.batch_size = tf.constant(batch_size, dtype='int32', shape=[], name='batch_size')  # a constant! (unlike regular Scholar)
        self.var_scale = tf.placeholder_with_default(1.0, [], name='var_scale')        # set to 0 to use posterior mean
        self.bg_scale = tf.placeholder_with_default(1.0, [], name='bg_scale')          # set to 0 to not use background
        self.is_training = tf.placeholder_with_default(True, [], name='is_training')   # placeholder for batchnorm
        self.eta_bn_prop = tf.placeholder_with_default(1.0, [], name='eta_bn_prop')    # used to anneal away from bn
        self.kld_weight = tf.placeholder_with_default(1.0, [], name='kld_weight')      # optional KLD weight param

        self.update_embeddings = update_embeddings
        self.update_background = update_background
        self.update_beta = update_beta
        self.optimizer_type = optimizer

        # create a placeholder for train / test inputs
        self.x = tf.placeholder(tf.float32, [None, dv], name='input')  # batch size x vocab matrix of word counts
        if n_labels > 0:
            self.y = tf.placeholder(tf.float32, [None, n_labels], name='input_y')
        else:
            self.y = tf.placeholder(tf.float32, [], name='input_y')
        if n_covariates > 0:
            self.c = tf.placeholder(tf.float32, [None, n_covariates], name='input_c')
        else:
            self.c = tf.placeholder(tf.float32, [], name='input_c')

        # create a placeholder for dropout strength
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # create placeholders to allow injecting a specific value of hidden variables
        self.theta_input = tf.placeholder(tf.float32, [None, n_topics], name='theta_input')
        # set self.use_theta_input to 1 to override sampled theta and generate from self.theta_input
        self.use_theta_input = tf.placeholder_with_default(0.0, [], name='use_theta_input')

        # create priors on the hidden state
        self.h_dim = (network_architecture["n_topics"])

        # interpret alpha as either a (symmetric) scalar prior or a vector prior
        if np.array(alpha).size == 1:
            self.alpha = alpha * np.ones((1, self.h_dim)).astype(np.float32)
        else:
            self.alpha = np.array(alpha).astype(np.float32)
            assert len(self.alpha) == self.h_dim

        # compute prior mean and variance of Laplace approximation to Dirichlet
        self.prior_mean = tf.constant((np.log(self.alpha).T - np.mean(np.log(self.alpha), 1)).T)
        if self.h_dim > 1:
            self.prior_var = tf.constant((((1.0/self.alpha) * (1 - (2.0/self.h_dim))).T + (1.0/(self.h_dim*self.h_dim)) * np.sum(1.0/self.alpha, 1)).T)
        else:
            self.prior_var = tf.constant(1.0/self.alpha)
        self.prior_logvar = tf.log(self.prior_var)

        # create the network
        self._create_network()
        with tf.name_scope('loss'):
            self._create_loss_optimizer(batch_size)  # the loss has a fixed batch size now!

        init = tf.global_variables_initializer()

        self.saver = tf.train.Saver()

        # create a session
        config = tf.ConfigProto(intra_op_parallelism_threads=threads,
                                inter_op_parallelism_threads=threads)
        self.sess = tf.InteractiveSession(config=config)
        self.writer = tf.summary.FileWriter('./graphs', self.sess.graph)

        if load_model_filename_prefix is None:
            self.sess.run(init)

            # initialize background
            if init_bg is not None:
                self.sess.run(self.network_weights['background'].assign(init_bg))

            # initialize topic-word weights
            if init_beta is not None:
                self.sess.run(self.network_weights['beta'].assign(init_beta))

            # initialize word embeddings
            if init_embeddings is not None:
                self.sess.run(self.network_weights['embeddings'].assign(init_embeddings))
        else:
            self.saver.restore(self.sess, load_model_filename_prefix)

    def save_to_disk(self, output_filename_prefix):
        self.saver.save(self.sess, output_filename_prefix)

        baseline_hazard_fit = {"hazard_sorted_unique_times": self.hazard_sorted_unique_times,
                               "log_baseline_hazard": self.log_baseline_hazard}

        with open(output_filename_prefix+".pickle", 'wb') as model_write:
            pickle.dump(baseline_hazard_fit, model_write)

    def load_baseline_hazard(self, output_filename_prefix):
        with open(output_filename_prefix+".pickle", 'rb') as model_read:
            baseline_hazard_fit = pickle.load(model_read)

        self.hazard_sorted_unique_times = baseline_hazard_fit["hazard_sorted_unique_times"]
        self.log_baseline_hazard = baseline_hazard_fit["log_baseline_hazard"]


    def _create_network(self):
        encoder_layers = self.network_architecture['encoder_layers']
        dh = self.network_architecture['n_topics']
        n_labels = self.network_architecture['n_labels']
        n_covariates = self.network_architecture['n_covariates']
        words_emb_dim = self.network_architecture['embedding_dim']
        label_emb_dim = self.network_architecture['label_emb_dim']
        covar_emb_dim = self.network_architecture['covar_emb_dim']
        emb_size = words_emb_dim
        use_covar_interactions = self.network_architecture['use_covar_interactions']
        survival_layers = self.network_architecture['survival_layers']
        survival_layers_num_nodes = self.network_architecture['survival_layers_num_nodes']
        survival_layers_keep_prob = self.network_architecture['survival_layers_keep_prob']

        self.network_weights = self._initialize_weights()

        # create the first layer of the encoder
        encoder_parts = []
        # convert word indices to embeddings
        en0_x = tf.matmul(self.x, self.network_weights['embeddings'])
        encoder_parts.append(en0_x)

        # add label if we have them
        if n_labels > 0:
            if label_emb_dim > 0:
                # use the label embedding if we're projecting them down
                y_emb = tf.matmul(self.y, self.network_weights['label_embeddings'])
                en0_y = y_emb
                emb_size += int(label_emb_dim)
                encoder_parts.append(en0_y)
            elif label_emb_dim < 0:
                # if label_emb_dim < 0 (default), just feed in label vectors as is
                emb_size += n_labels
                encoder_parts.append(self.y)

        # do the same for covariates
        if n_covariates > 0:
            if covar_emb_dim > 0:
                c_emb = tf.matmul(self.c, self.network_weights['covariate_embeddings'])
                en0_c = c_emb
                emb_size += covar_emb_dim
                encoder_parts.append(en0_c)
            elif covar_emb_dim < 0:
                # if covar_emb_dim < 0 (default), just feed in covariate vectors as is
                c_emb = self.c
                emb_size += n_covariates
                encoder_parts.append(c_emb)
            else:
                # if covar_emb_dim == 0, do not give the covariate vectors to the encoder
                c_emb = self.c

        # combine everything to produce the output of layer 0
        if len(encoder_parts) > 1:
            en0 = tf.concat(encoder_parts, axis=1)
        else:
            en0 = en0_x

        # optionally add more encoder layers
        if encoder_layers == 0:
            # technically this will involve two layers, but they're both linear, so it's basically the same as one
            encoder_output = en0
        elif encoder_layers == 1:
            encoder_output = tf.nn.softplus(en0, name='softplus0')
        else:
            en0_softmax = tf.nn.softplus(en0, name='softplus0')
            en1 = slim.layers.linear(en0_softmax, emb_size, scope='en1')
            encoder_output = tf.nn.softplus(en1, name='softplus1')

        # optionally add an encoder shortcut
        if self.network_architecture['encoder_shortcut']:
            encoder_output = tf.add(encoder_output, slim.layers.linear(self.x, emb_size))

        # apply dropout to encoder output
        encoder_output_do = slim.layers.dropout(encoder_output, self.keep_prob, scope='en_dropped')

        # apply linear transformations to encoder output for mean and log of diagonal of covariance matrix
        self.posterior_mean = slim.layers.linear(encoder_output_do, dh, scope='FC_mean')
        self.posterior_logvar = slim.layers.linear(encoder_output_do, dh, scope='FC_logvar')

        # apply batchnorm to these vectors
        self.posterior_mean_bn = slim.layers.batch_norm(self.posterior_mean, scope='BN_mean', is_training=self.is_training)
        self.posterior_logvar_bn = slim.layers.batch_norm(self.posterior_logvar, scope='BN_logvar', is_training=self.is_training)

        with tf.name_scope('h_scope'):
            # sample from symmetric Gaussian noise
            eps = tf.random_normal((self.batch_size, dh), 0, 1, dtype=tf.float32)
            # use the reparameterization trick to get a sample from the latent variable posterior
            self.z = tf.add(self.posterior_mean_bn, tf.multiply(self.var_scale, tf.multiply(tf.sqrt(tf.exp(self.posterior_logvar_bn)), eps)))
            self.posterior_var = tf.exp(self.posterior_logvar_bn)

        # apply dropout to the (unnormalized) latent representation
        z_do = slim.layers.dropout(self.z, self.keep_prob, scope='p_dropped')

        # transform z to the simplex using a softmax
        theta_sample = slim.layers.softmax(z_do)

        # use manually-set generator output for generation; during training use_theta_input should equal 0
        self.theta = tf.add(tf.multiply((1.0 - self.use_theta_input), theta_sample), tf.multiply(self.use_theta_input, self.theta_input))

        # combine latent representation with topics and background
        eta = tf.add(tf.matmul(self.theta, self.network_weights['beta']), tf.multiply(self.bg_scale, self.network_weights['background']))

        # add deviations for covariates (and interactions)
        if n_covariates > 0:
            eta = tf.add(eta, tf.matmul(c_emb, self.network_weights['beta_c']))
            if use_covar_interactions:
                gen_output_rsh = tf.reshape(self.theta, [self.batch_size, dh, 1])
                c_emb_rsh = array_ops.reshape(c_emb, [self.batch_size, 1, self.beta_c_length])
                covar_interactions = tf.reshape(gen_output_rsh * c_emb_rsh, [self.batch_size, self.beta_ci_length])
                eta = tf.add(eta, tf.matmul(covar_interactions, self.network_weights['beta_ci']))

        # add batchnorm to eta
        eta_bn = slim.layers.batch_norm(eta, scope='BN_decoder', is_training=self.is_training)

        # reconstruct both with and without batchnorm on eta
        self.x_recon = tf.nn.softmax(eta_bn)
        self.x_recon_no_bn = tf.nn.softmax(eta)

        # predict labels using theta and (optionally) covariates
        if n_labels > 0:
            if n_covariates > 0 and self.network_architecture['covars_in_survival_regression']:
                self.survival_input = tf.concat([self.theta, c_emb], axis=1)
            else:
                self.survival_input = self.theta

            for hidden_layer_idx in range(survival_layers):

                self.survival_input = slim.layers.linear(self.survival_input, survival_layers_num_nodes[hidden_layer_idx],
                                                         scope='cls%d' % hidden_layer_idx)
                self.survival_input = tf.nn.relu(self.survival_input, name='cls%d_relu' % hidden_layer_idx)
                # this is to mimic what pycox does for DeepHit and DeepSurv: activation (relu) --> batch norm --> dropout 
                self.survival_input = slim.layers.batch_norm(self.survival_input, scope='cls%d_bn' % hidden_layer_idx, is_training=self.is_training)
                self.survival_input = slim.layers.dropout(self.survival_input, survival_layers_keep_prob, scope='cls%d_dropped' % hidden_layer_idx)
                
                # self.survival_input = slim.layers.linear(self.survival_input, dh,
                #                                     scope='cls%d' % hidden_layer_idx)
                # self.survival_input = tf.nn.softplus(self.survival_input,
                #                                 name='cls%d_softplus' % hidden_layer_idx)

            if self.cox_elastic_net_lmbda > 0:  # apply elastic-net regularization on Cox regression coefficients
                if self.cox_use_bias:
                    self.survival_inner_prod = slim.layers.linear(self.survival_input, 1, scope='survival_inner_prod',
                                                                  weights_regularizer=slim.l1_l2_regularizer(self.l1_weight, self.l2_weight))
                else:
                    self.survival_inner_prod = slim.layers.linear(self.survival_input, 1, scope='survival_inner_prod',
                                                                  weights_regularizer=slim.l1_l2_regularizer(self.l1_weight, self.l2_weight),
                                                                  biases_initializer=None)
            else:
                if self.cox_use_bias:
                    self.survival_inner_prod = slim.layers.linear(self.survival_input, 1, scope='survival_inner_prod')
                else:
                    self.survival_inner_prod = slim.layers.linear(self.survival_input, 1, scope='survival_inner_prod', biases_initializer=None)

            for v in tf.trainable_variables():
                if v.name.startswith('survival_inner_prod/weights'):
                    self.survival_weights = v
                    break
            else:
                self.survival_weights = None

            for v in tf.trainable_variables():
                if v.name.startswith('survival_inner_prod/biases'):
                    self.survival_bias = v
                    break
            else:
                self.survival_bias = None

    def _initialize_weights(self):
        all_weights = dict()

        dh = self.network_architecture['n_topics']
        dv = self.network_architecture['dv']
        embedding_dim = self.network_architecture['embedding_dim']
        n_labels = self.network_architecture['n_labels']
        label_emb_dim = self.network_architecture['label_emb_dim']
        n_covariates = self.network_architecture['n_covariates']
        covar_emb_dim = self.network_architecture['covar_emb_dim']

        # background log-frequency of terms (overwrite with pre-specified initialization later))
        # all_weights['background'] = tf.Variable(tf.zeros(dv, dtype=tf.float32), trainable=self.update_background)
        all_weights['background'] = tf.get_variable('background',
                                                    shape=[dv],
                                                    dtype=tf.float32,
                                                    initializer=tf.zeros_initializer(),
                                                    trainable=self.update_background)

        # initial layer of word embeddings (overwrite with pre-specified initialization later))
        # all_weights['embeddings'] = tf.Variable(xavier_init(dv, embedding_dim), trainable=self.update_embeddings)
        all_weights['embeddings'] = tf.get_variable('embedding',
                                                    shape=[dv, embedding_dim],
                                                    dtype=tf.float32,
                                                    initializer=tf.contrib.layers.xavier_initializer(),
                                                    trainable=self.update_embeddings)

        # topic deviations (overwrite with pre-specified initialization later))
        # all_weights['beta'] = tf.Variable(xavier_init(dh, dv), trainable=self.update_beta)
        all_weights['beta'] = tf.get_variable('beta',
                                              shape=[dh, dv],
                                              dtype=tf.float32,
                                              initializer=tf.contrib.layers.xavier_initializer(),
                                              trainable=self.update_beta)

        # create embeddings for labels
        if n_labels > 0:
            if label_emb_dim > 0:
                # all_weights['label_embeddings'] = tf.Variable(xavier_init(n_labels, label_emb_dim), trainable=True)
                all_weights['label_embeddings'] = tf.get_variable('label_embeddings',
                                                                  shape=[n_labels, label_emb_dim],
                                                                  dtype=tf.float32,
                                                                  initializer=tf.contrib.layers.xavier_initializer(),
                                                                  trainable=True)

        if n_covariates > 0:
            if covar_emb_dim > 0:
                # all_weights['covariate_embeddings'] = tf.Variable(xavier_init(n_covariates, covar_emb_dim), trainable=True)
                all_weights['covariate_embeddings'] = tf.get_variable('covariate_embeddings',
                                                                      shape=[n_covariates, covar_emb_dim],
                                                                      dtype=tf.float32,
                                                                      initializer=tf.contrib.layers.xavier_initializer(),
                                                                      trainable=True)

        # all_weights['beta_c'] = tf.Variable(xavier_init(self.beta_c_length, dv))
        # all_weights['beta_ci'] = tf.Variable(xavier_init(self.beta_ci_length, dv))
        all_weights['beta_c'] = tf.get_variable('beta_c',
                                                shape=[self.beta_c_length, dv],
                                                dtype=tf.float32,
                                                initializer=tf.contrib.layers.xavier_initializer())
        all_weights['beta_ci'] = tf.get_variable('beta_ci',
                                                 shape=[self.beta_ci_length, dv],
                                                 dtype=tf.float32,
                                                 initializer=tf.contrib.layers.xavier_initializer())

        return all_weights

    def _create_loss_optimizer(self, batch_size):

        # Compute an interpolation between reconstruction with and without batchnorm on eta.
        # This is done to allow annealing away from using batchnorm on eta over the course of training
        x_recon = tf.add(tf.add(tf.multiply(self.eta_bn_prop, self.x_recon), tf.multiply((1.0 - self.eta_bn_prop), self.x_recon_no_bn)), 1e-10)

        # compute the negative log loss
        # https://stackoverflow.com/questions/33712178/tensorflow-nan-bug : this solves the potential NaN error during reduce_sum
        # self.x = tf.Print(self.x, [self.x])
        self.NL_x = -tf.reduce_sum(self.x * tf.log(x_recon), 1)

        if self.network_architecture['n_labels'] > 0:
            # # loss for categorical labels
            # # TODO: add losses for other types of labels
            # NL_y = -tf.reduce_sum(self.y * tf.log(self.y_recon+1e-10), 1)  # test

            y_observed_times = self.y[:, 0]
            y_event_indicators = self.y[:, 1]

            R_batch = tf.cast(tf.greater_equal(tf.tile(tf.expand_dims(y_observed_times, axis=0), [batch_size, 1]),
                                               tf.tile(tf.expand_dims(y_observed_times, axis=-1), [1, batch_size])),
                              tf.float32)

            NL_y = tf.squeeze(-tf.multiply(self.survival_inner_prod - tf.log(tf.matmul(R_batch, tf.exp(self.survival_inner_prod))),
                                           tf.expand_dims(y_event_indicators, axis=-1))) * self.survival_loss_weight

            self.survival_loss = tf.reduce_mean(NL_y)

            self.NL = tf.add(self.NL_x, NL_y)

            survival_regularization_losses = tf.losses.get_regularization_losses(scope='survival_inner_prod')
            for regularization_loss in survival_regularization_losses:
                self.survival_loss = tf.add(self.survival_loss, regularization_loss)
        else:
            self.NL = self.NL_x
            survival_regularization_losses = []

        # compute terms for the KL divergence between prior and variational posterior
        var_division = self.posterior_var / self.prior_var
        diff = self.posterior_mean_bn - self.prior_mean
        diff_term = diff * diff / self.prior_var
        logvar_division = self.prior_logvar - self.posterior_logvar_bn

        self.KLD = 0.5 * (tf.reduce_sum(var_division + diff_term + logvar_division, 1) - self.h_dim)

        self.losses = tf.add(self.NL, tf.multiply(self.kld_weight, self.KLD))
        self.loss = tf.reduce_mean(self.losses)

        for regularization_loss in survival_regularization_losses:
            self.loss = tf.add(self.loss, regularization_loss)

        # add in regularization terms
        if self.regularize:
            self.loss = tf.add(self.loss, tf.reduce_sum(tf.multiply(self.l2_strengths, tf.square(self.network_weights['beta']))))
            if self.network_architecture['n_covariates']:
                self.loss = tf.add(self.loss, tf.reduce_sum(tf.multiply(self.l2_strengths_c, tf.square(self.network_weights['beta_c']))))
                if self.network_architecture['use_covar_interactions']:
                    self.loss = tf.add(self.loss, tf.reduce_sum(tf.multiply(self.l2_strengths_ci, tf.square(self.network_weights['beta_ci']))))

        # explicitly add batchnorm terms to parameters to be updated so as to save the global means
        update_ops = []
        update_ops.extend(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='BN_mean'))
        update_ops.extend(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='BN_logvar'))
        update_ops.extend(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='BN_decoder'))

        # choose an optimizer
        with tf.control_dependencies(update_ops):
            if self.optimizer_type == 'adam':
                # print("Using Adam")
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.adam_beta1).minimize(self.loss)
            elif self.optimizer_type == 'adagrad':
                # print("Using adagrad")
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            else:
                # print("Using SGD")
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def fit(self, X, Y, C, l2_strengths, l2_strengths_c, l2_strengths_ci, is_training=True, eta_bn_prop=1.0, kld_weight=1.0, keep_prob=0.8):
        """
        Fit the model to data
        :param X: np.array of document word counts [batch size x vocab size]
        :param Y: np.array of labels [batch size x n_labels]
        :param C: np.array of covariates [batch size x n_covariates]
        :param l2_strengths: np.array of l2 weights on beta (updated in run_scholar_tf.py)
        :param l2_strengths_c: np.array of l2 weights on beta_c (updated in run_scholar_tf.py)
        :param l2_strengths_ci: np.array of l2 weights on beta_ci (updated in run_scholar_tf.py)
        :param eta_bn_prop: in [0, 1] controlling the interpolation between using batch norm on the final layer and not
        :param kld_weight: weighting factor for KLD term (default=1.0)
        :param keep_prob: probability of not zeroing a weight in dropout
        :return: overall loss for minibatch; loss from the survival regression; per-instance predictions
        """
        # Standardize:
        # train_mean = X.mean(axis=0)
        # train_std = X.std(axis=0)
        # X = (X - train_mean)/train_std
        # if np.sum(np.isnan(X)) > 0:
        
        batch_size = self.sess.run(self.batch_size)
        n = self.get_batch_size(X)
        assert n == batch_size
        theta_input = np.zeros([batch_size, self.network_architecture['n_topics']]).astype('float32')

        if Y is not None:
            opt, loss, survival_loss = self.sess.run((self.optimizer, self.loss, self.survival_loss), feed_dict={self.x: X, self.y: Y, self.c: C, self.keep_prob: keep_prob, self.l2_strengths: l2_strengths, self.l2_strengths_c: l2_strengths_c, self.l2_strengths_ci: l2_strengths_ci, self.eta_bn_prop: eta_bn_prop, self.kld_weight: kld_weight, self.theta_input: theta_input, self.is_training: is_training})
        else:
            opt, loss = self.sess.run((self.optimizer, self.loss), feed_dict={self.x: X, self.y: Y, self.c: C, self.keep_prob: keep_prob, self.l2_strengths: l2_strengths, self.l2_strengths_c: l2_strengths_c, self.l2_strengths_ci: l2_strengths_ci, self.eta_bn_prop: eta_bn_prop, self.kld_weight: kld_weight, self.theta_input: theta_input, self.is_training: is_training})
            survival_loss = 0
        return loss, survival_loss

    def fit_baseline_hazard(self, X, Y, C=None, eta_bn_prop=0.0, parallel="none"):
        observed_times = Y[:, 0]
        event_indicators = Y[:, 1]

        _, survival_inner_prod, _, _ \
            = self.predict(X, C, eta_bn_prop,
                           predict_median_survival_times=False,
                           predict_survival_function=False)

        if self.survival_bias is not None:
            bias = self.sess.run(self.survival_bias)[0]
        else:
            bias = 0.

        event_counts = Counter()
        for t, r in zip(observed_times, event_indicators):
            event_counts[t] += int(r)

        sorted_unique_times = np.sort(list(event_counts.keys()))
        num_unique_times = len(sorted_unique_times)
        log_baseline_hazard = np.zeros(num_unique_times)

        if parallel == "prediction":
            
            tic = time.time()
            print(">>>> In progress: fitting baseline hazards...")
            others_dict = dict()
            others_dict['observed_times'] = observed_times
            others_dict['survival_inner_prod'] = survival_inner_prod
            others_dict['bias'] = bias
            others_dict['event_counts'] = event_counts

            fit_baseline_hazard_input = [(t, others_dict) for t in sorted_unique_times]

            fit_baseline_hazard_input_pool = Pool(processes=9)
            log_baseline_hazard = fit_baseline_hazard_input_pool.map(\
                                fit_baseline_hazard_par, fit_baseline_hazard_input)
            log_baseline_hazard = np.array(log_baseline_hazard, dtype="float32")
            fit_baseline_hazard_input_pool.close()
            fit_baseline_hazard_input_pool.join()

            toc = time.time()
            print(">>>> Time spent: {} seconds".format(toc-tic))

        else:
            # tic = time.time()
            # print(">>>> In progress: fitting baseline hazards...")
            # pbar = ProgressBar()
            # for time_idx, t in pbar(list(enumerate(sorted_unique_times))):
            for time_idx, t in enumerate(sorted_unique_times):
                logsumexp_args = []
                for subj_idx, observed_time in enumerate(observed_times):
                    if observed_time >= t:
                        logsumexp_args.append(survival_inner_prod[subj_idx] + bias)
                if event_counts[t] > 0:
                    log_baseline_hazard[time_idx] \
                        = np.log(event_counts[t]) - logsumexp(logsumexp_args)
                else:
                    log_baseline_hazard[time_idx] \
                        = -np.inf - logsumexp(logsumexp_args)

            # toc = time.time()
            # print(">>>> Time spent: {} seconds".format(toc-tic))

        self.hazard_sorted_unique_times = sorted_unique_times
        self.log_baseline_hazard = log_baseline_hazard

    def predict(self, X, C, eta_bn_prop=0.0, predict_median_survival_times=True, predict_survival_function=True, parallel="prediction"):
        """
        Predict document representations (theta) and labels (Y) given input (X) and covariates (C)
        """
        l2_strengths = np.zeros(self.network_weights['beta'].shape)
        l2_strengths_c = np.zeros(self.network_weights['beta_c'].shape)
        l2_strengths_ci = np.zeros(self.network_weights['beta_ci'].shape)

        n = self.get_batch_size(X)
        if n == 1 and len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)
        if C is not None and n == 1 and len(C.shape) == 1:
            C = np.expand_dims(C, axis=0)

        batch_size = self.sess.run(self.batch_size)
        num_batches = n / batch_size
        if num_batches != int(num_batches):
            num_batches = np.ceil(num_batches)
        num_batches = int(num_batches)

        minibatch_Y = np.zeros((batch_size, self.network_architecture['n_labels'])).astype('float32')

        if C is None:
            minibatch_C = None

        minibatch_theta_input = np.zeros([batch_size, self.network_architecture['n_topics']]).astype('float32')

        theta = np.zeros((n, self.network_architecture['n_topics']), dtype='float32')
        survival_inner_prod = np.zeros(n, dtype='float32')

        for batch_idx in list(range(num_batches)):
            start_idx = batch_idx*batch_size
            if batch_idx == num_batches - 1:
                end_idx = n
                minibatch_X = np.zeros((batch_size, X.shape[1]), dtype='float32')
                num_actual = end_idx - start_idx
                minibatch_X[:num_actual] = X[start_idx:n]
                if C is not None:
                    minibatch_C = np.zeros((batch_size, C.shape[1]), dtype='float32')
                    minibatch_C[:num_actual] = C[start_idx:n]
                minibatch_theta, minibatch_survival_inner_prod = self.sess.run((self.theta, self.survival_inner_prod), feed_dict={self.x: minibatch_X, self.y: minibatch_Y, self.c: minibatch_C, self.keep_prob: 1.0, self.l2_strengths: l2_strengths, self.l2_strengths_c: l2_strengths_c, self.l2_strengths_ci: l2_strengths_ci, self.var_scale: 0.0, self.is_training: False, self.theta_input: minibatch_theta_input, self.eta_bn_prop: eta_bn_prop})
                theta[start_idx:end_idx] = minibatch_theta[:num_actual]
                survival_inner_prod[start_idx:end_idx] = minibatch_survival_inner_prod.flatten()[:num_actual]
            else:
                end_idx = start_idx + batch_size
                minibatch_X = X[start_idx:end_idx].astype('float32')
                if C is not None:
                    minibatch_C = C[start_idx:end_idx].astype('float32')
                minibatch_theta, minibatch_survival_inner_prod = self.sess.run((self.theta, self.survival_inner_prod), feed_dict={self.x: minibatch_X, self.y: minibatch_Y, self.c: minibatch_C, self.keep_prob: 1.0, self.l2_strengths: l2_strengths, self.l2_strengths_c: l2_strengths_c, self.l2_strengths_ci: l2_strengths_ci, self.var_scale: 0.0, self.is_training: False, self.theta_input: minibatch_theta_input, self.eta_bn_prop: eta_bn_prop})
                theta[start_idx:end_idx] = minibatch_theta
                survival_inner_prod[start_idx:end_idx] = minibatch_survival_inner_prod.flatten()

        if predict_median_survival_times:
            if self.survival_bias is not None:
                bias = self.sess.run(self.survival_bias)[0]
            else:
                bias = 0.

            num_unique_times = len(self.hazard_sorted_unique_times)
            predicted_median_survival_times = np.zeros(n, dtype='float32')
            log_minus_log_half = -0.366512920581664347619010868584155105054378509521484375

            if parallel == "prediction":
                others_dict = dict()
                others_dict['log_baseline_hazard'] = self.log_baseline_hazard
                others_dict['survival_inner_prod'] = survival_inner_prod
                others_dict['bias'] = bias
                others_dict['num_unique_times'] = num_unique_times
                others_dict['hazard_sorted_unique_times'] = self.hazard_sorted_unique_times

                tic = time.time()
                print(">>>> In progress: predicting median survival time...")

                predicted_median_survival_times_input = [(i, others_dict) for i in range(n)]
                predicted_median_survival_times_pool = Pool(processes=9)
                predicted_median_survival_times = predicted_median_survival_times_pool.map(\
                                            predict_median_survival_times_par, predicted_median_survival_times_input)
                predicted_median_survival_times = np.array(predicted_median_survival_times, dtype="float32")
                predicted_median_survival_times_pool.close()
                predicted_median_survival_times_pool.join()

                toc = time.time()
                print(">>>> Time spent: {} seconds".format(toc-tic))

            else:

                tic = time.time()
                # print(">>>> In progress: predicting median survival time...")
                # pbar = ProgressBar()
                # for i in pbar(list(range(n))):
                for i in range(n):
                    log_hazard = self.log_baseline_hazard + survival_inner_prod[i] + bias
                    log_cumulative_hazard = np.zeros(num_unique_times)
                    for time_idx in range(num_unique_times):
                        log_cumulative_hazard[time_idx] \
                            = logsumexp(log_hazard[:time_idx + 1])

                    t_inf = np.inf
                    t_sup = 0.
                    for time_idx, t in enumerate(self.hazard_sorted_unique_times):
                        if log_minus_log_half <= log_cumulative_hazard[time_idx]:
                            if t < t_inf:
                                t_inf = t
                        if log_minus_log_half >= log_cumulative_hazard[time_idx]:
                            if t > t_sup:
                                t_sup = t

                    if t_inf == np.inf:
                        predicted_median_survival_times[i] = t_sup
                    else:
                        predicted_median_survival_times[i] = 0.5 * (t_inf + t_sup)
                
                # toc = time.time()
                # print(">>>> Time spent: {} seconds".format(toc-tic))

        else:
            predicted_median_survival_times = None

        if predict_survival_function:
            if self.survival_bias is not None:
                bias = self.sess.run(self.survival_bias)[0]
            else:
                bias = 0.

            num_unique_times = len(self.hazard_sorted_unique_times)
            predicted_survival_functions = np.zeros((n, num_unique_times), dtype='float32')

            if parallel == "prediction":
                if predict_median_survival_times:
                    predicted_survival_functions_input = predicted_median_survival_times_input
                else:
                    others_dict = dict()
                    others_dict['log_baseline_hazard'] = self.log_baseline_hazard
                    others_dict['survival_inner_prod'] = survival_inner_prod
                    others_dict['bias'] = bias
                    others_dict['num_unique_times'] = num_unique_times
                    others_dict['hazard_sorted_unique_times'] = self.hazard_sorted_unique_times
                    predicted_survival_functions_input = [(i, others_dict) for i in range(n)]

                tic = time.time()
                print(">>>> In progress: predicting survival functions...")
                predicted_survival_functions_pool = Pool(processes=9)
                predicted_survival_functions = predicted_survival_functions_pool.map(\
                                             predicted_survival_functions_par, predicted_survival_functions_input)
                predicted_survival_functions = np.array(predicted_survival_functions, dtype="float32")
                predicted_survival_functions_pool.close()
                predicted_survival_functions_pool.join()
                toc = time.time()
                print(">>>> Time spent: {} seconds".format(toc-tic))

            else:
                tic = time.time()
                # print(">>>> In progress: predicting survival functions...")
                # pbar = ProgressBar()
                # for i in pbar(list(range(n))):
                for i in range(n):
                    log_hazard = self.log_baseline_hazard + survival_inner_prod[i] + bias
                    log_cumulative_hazard = np.zeros(num_unique_times)
                    for time_idx in range(num_unique_times):
                        log_cumulative_hazard[time_idx] \
                            = logsumexp(log_hazard[:time_idx + 1])
                    curr_survival_probs = np.exp(-np.exp(log_cumulative_hazard))
                    predicted_survival_functions[i, :] = curr_survival_probs
                # toc = time.time()
                # print(">>>> Time spent: {} seconds".format(toc-tic))

        else:
            predicted_survival_functions = None

        return theta, survival_inner_prod, predicted_median_survival_times, predicted_survival_functions

    def get_losses(self, X, Y, C, eta_bn_prop=0.0, n_samples=0):
        """
        Compute and return the loss values for all instances in X, Y, C
        """
        l2_strengths = np.zeros(self.network_weights['beta'].shape)
        l2_strengths_c = np.zeros(self.network_weights['beta_c'].shape)
        l2_strengths_ci = np.zeros(self.network_weights['beta_ci'].shape)

        n = self.get_batch_size(X)
        if n == 1 and len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)
        if Y is not None and n == 1 and len(Y.shape) == 1:
            Y = np.expand_dims(Y, axis=0)
        if C is not None and n == 1 and len(C.shape) == 1:
            C = np.expand_dims(C, axis=0)

        batch_size = self.sess.run(self.batch_size)
        num_batches = n / batch_size
        if num_batches != int(num_batches):
            num_batches = np.ceil(num_batches)
        num_batches = int(num_batches)

        if Y is None:
            minibatch_Y = np.zeros((batch_size, self.network_architecture['n_labels'])).astype('float32')
        if C is None:
            minibatch_C = None

        minibatch_theta_input = np.zeros([batch_size, self.network_architecture['n_topics']]).astype('float32')

        losses = np.zeros(n, dtype='float32')

        for batch_idx in range(num_batches):
            start_idx = batch_idx*batch_size
            if batch_idx == num_batches - 1:
                end_idx = n
                minibatch_X = np.zeros((batch_size, X.shape[1]), dtype='float32')
                num_actual = end_idx - start_idx
                minibatch_X[:num_actual] = X[start_idx:n]
                if Y is not None:
                    minibatch_Y = np.zeros((batch_size, Y.shape[1]), dtype='float32')
                    minibatch_Y[:num_actual] = Y[start_idx:n]
                if C is not None:
                    minibatch_C = np.zeros((batch_size, C.shape[1]), dtype='float32')
                    minibatch_C[:num_actual] = C[start_idx:n]
                if n_samples == 0:
                    minibatch_losses = self.sess.run(self.losses, feed_dict={self.x: minibatch_X, self.y: minibatch_Y, self.c: minibatch_C, self.keep_prob: 1.0, self.l2_strengths: l2_strengths, self.l2_strengths_c: l2_strengths_c, self.l2_strengths_ci: l2_strengths_ci, self.var_scale: 0.0, self.is_training: False, self.theta_input: minibatch_theta_input, self.eta_bn_prop: eta_bn_prop})
                else:
                    minibatch_losses = self.sess.run(self.losses, feed_dict={self.x: minibatch_X, self.y: minibatch_Y, self.c: minibatch_C, self.keep_prob: 1.0, self.l2_strengths: l2_strengths, self.l2_strengths_c: l2_strengths_c, self.l2_strengths_ci: l2_strengths_ci, self.var_scale: 1.0, self.is_training: False, self.theta_input: minibatch_theta_input, self.eta_bn_prop: eta_bn_prop})
                    for s in range(1, n_samples):
                        minibatch_losses += self.sess.run(self.losses, feed_dict={self.x: minibatch_X, self.y: minibatch_Y, self.c: minibatch_C, self.keep_prob: 1.0, self.l2_strengths: l2_strengths, self.l2_strengths_c: l2_strengths_c, self.l2_strengths_ci: l2_strengths_ci, self.var_scale: 1.0, self.is_training: False, self.theta_input: minibatch_theta_input, self.eta_bn_prop: eta_bn_prop})
                    minibatch_losses /= float(n_samples)
                losses[start_idx:end_idx] = minibatch_losses[:num_actual]
            else:
                end_idx = start_idx + batch_size
                minibatch_X = X[start_idx:end_idx].astype('float32')
                if Y is not None:
                    minibatch_Y = Y[start_idx:end_idx].astype('float32')
                if C is not None:
                    minibatch_C = C[start_idx:end_idx].astype('float32')
                if n_samples == 0:
                    minibatch_losses = self.sess.run(self.losses, feed_dict={self.x: minibatch_X, self.y: minibatch_Y, self.c: minibatch_C, self.keep_prob: 1.0, self.l2_strengths: l2_strengths, self.l2_strengths_c: l2_strengths_c, self.l2_strengths_ci: l2_strengths_ci, self.var_scale: 0.0, self.is_training: False, self.theta_input: minibatch_theta_input, self.eta_bn_prop: eta_bn_prop})
                else:
                    minibatch_losses = self.sess.run(self.losses, feed_dict={self.x: minibatch_X, self.y: minibatch_Y, self.c: minibatch_C, self.keep_prob: 1.0, self.l2_strengths: l2_strengths, self.l2_strengths_c: l2_strengths_c, self.l2_strengths_ci: l2_strengths_ci, self.var_scale: 1.0, self.is_training: False, self.theta_input: minibatch_theta_input, self.eta_bn_prop: eta_bn_prop})
                    for s in range(1, n_samples):
                        minibatch_losses += self.sess.run(self.losses, feed_dict={self.x: minibatch_X, self.y: minibatch_Y, self.c: minibatch_C, self.keep_prob: 1.0, self.l2_strengths: l2_strengths, self.l2_strengths_c: l2_strengths_c, self.l2_strengths_ci: l2_strengths_ci, self.var_scale: 1.0, self.is_training: False, self.theta_input: minibatch_theta_input, self.eta_bn_prop: eta_bn_prop})
                    minibatch_losses /= float(n_samples)
                losses[start_idx:end_idx] = minibatch_losses

        return losses

    def compute_theta(self, X, Y, C):
        """
        Return the latent document representation (mean of posterior of theta) for a given batch of X, Y, C
        """
        l2_strengths = np.zeros(self.network_weights['beta'].shape)
        l2_strengths_c = np.zeros(self.network_weights['beta_c'].shape)
        l2_strengths_ci = np.zeros(self.network_weights['beta_ci'].shape)

        n = self.get_batch_size(X)
        if n == 1 and len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)
        if Y is not None and n == 1 and len(Y.shape) == 1:
            Y = np.expand_dims(Y, axis=0)
        if C is not None and n == 1 and len(C.shape) == 1:
            C = np.expand_dims(C, axis=0)

        batch_size = self.sess.run(self.batch_size)
        num_batches = n / batch_size
        if num_batches != int(num_batches):
            num_batches = np.ceil(num_batches)
        num_batches = int(num_batches)

        if Y is None:
            minibatch_Y = np.zeros((batch_size, self.network_architecture['n_labels'])).astype('float32')
        if C is None:
            minibatch_C = None

        minibatch_theta_input = np.zeros([batch_size, self.network_architecture['n_topics']]).astype('float32')

        theta = np.zeros((n, self.network_architecture['n_topics']), dtype='float32')

        for batch_idx in range(num_batches):
            start_idx = batch_idx*batch_size
            if batch_idx == num_batches - 1:
                end_idx = n
                minibatch_X = np.zeros((batch_size, X.shape[1]), dtype='float32')
                num_actual = end_idx - start_idx
                minibatch_X[:num_actual] = X[start_idx:n]
                if Y is not None:
                    minibatch_Y = np.zeros((batch_size, Y.shape[1]), dtype='float32')
                    minibatch_Y[:num_actual] = Y[start_idx:n]
                if C is not None:
                    minibatch_C = np.zeros((batch_size, C.shape[1]), dtype='float32')
                    minibatch_C[:num_actual] = C[start_idx:n]
                minibatch_theta = self.sess.run(self.theta, feed_dict={self.x: minibatch_X, self.y: minibatch_Y, self.c: minibatch_C, self.keep_prob: 1.0, self.l2_strengths: l2_strengths, self.l2_strengths_c: l2_strengths_c, self.l2_strengths_ci: l2_strengths_ci, self.var_scale: 0.0, self.is_training: False, self.theta_input: minibatch_theta_input})
                theta[start_idx:end_idx] = minibatch_theta[:num_actual]
            else:
                end_idx = start_idx + batch_size
                minibatch_X = X[start_idx:end_idx].astype('float32')
                if Y is not None:
                    minibatch_Y = Y[start_idx:end_idx].astype('float32')
                if C is not None:
                    minibatch_C = C[start_idx:end_idx].astype('float32')
                minibatch_theta = self.sess.run(self.theta, feed_dict={self.x: minibatch_X, self.y: minibatch_Y, self.c: minibatch_C, self.keep_prob: 1.0, self.l2_strengths: l2_strengths, self.l2_strengths_c: l2_strengths_c, self.l2_strengths_ci: l2_strengths_ci, self.var_scale: 0.0, self.is_training: False, self.theta_input: minibatch_theta_input})
                theta[start_idx:end_idx] = minibatch_theta

        return theta

    def get_survival_inner_prod(self, theta, X=None, C=None, eta_bn_prop=0.0):
        '''
        Runs the survival layer and get output.

        '''
        l2_strengths = np.zeros(self.network_weights['beta'].shape)
        l2_strengths_c = np.zeros(self.network_weights['beta_c'].shape)
        l2_strengths_ci = np.zeros(self.network_weights['beta_ci'].shape)

        n = self.get_batch_size(theta)
        if n == 1 and len(theta.shape) == 1:
            theta = np.expand_dims(theta, axis=0)
        if C is not None and n == 1 and len(C.shape) == 1:
            C = np.expand_dims(C, axis=0)

        batch_size = self.sess.run(self.batch_size)
        num_batches = n / batch_size
        if num_batches != int(num_batches):
            num_batches = np.ceil(num_batches)
        num_batches = int(num_batches)

        minibatch_X = np.zeros((batch_size, self.network_architecture['dv'])).astype('float32')
        minibatch_Y = np.zeros((batch_size, self.network_architecture['n_labels'])).astype('float32')
        if C is None:
            minibatch_C = None
        survival_inner_prod = np.zeros(n, dtype='float32')

        for batch_idx in list(range(num_batches)):
            start_idx = batch_idx*batch_size
            if batch_idx == num_batches - 1:
                end_idx = n
                minibatch_theta_input = np.zeros((batch_size, self.network_architecture['n_topics']), dtype='float32')
                num_actual = end_idx - start_idx
                minibatch_theta_input[:num_actual] = theta[start_idx:n]
                if C is not None:
                    minibatch_C = np.zeros((batch_size, C.shape[1]), dtype='float32')
                    minibatch_C[:num_actual] = C[start_idx:n]
                minibatch_survival_inner_prod = self.sess.run(self.survival_inner_prod, feed_dict={self.x: minibatch_X, self.y: minibatch_Y, self.c: minibatch_C, self.keep_prob: 1.0, self.l2_strengths: l2_strengths, self.l2_strengths_c: l2_strengths_c, self.l2_strengths_ci: l2_strengths_ci, self.var_scale: 0.0, self.is_training: False, self.theta_input: minibatch_theta_input, self.use_theta_input: 1, self.eta_bn_prop: eta_bn_prop})
                survival_inner_prod[start_idx:end_idx] = minibatch_survival_inner_prod.flatten()[:num_actual]
            else:
                end_idx = start_idx + batch_size
                minibatch_theta_input = theta[start_idx:end_idx].astype('float32')
                minibatch_survival_inner_prod = self.sess.run(self.survival_inner_prod, feed_dict={self.x: minibatch_X, self.y: minibatch_Y, self.c: minibatch_C, self.keep_prob: 1.0, self.l2_strengths: l2_strengths, self.l2_strengths_c: l2_strengths_c, self.l2_strengths_ci: l2_strengths_ci, self.var_scale: 0.0, self.is_training: False, self.theta_input: minibatch_theta_input, self.use_theta_input: 1, self.eta_bn_prop: eta_bn_prop})
                survival_inner_prod[start_idx:end_idx] = minibatch_survival_inner_prod.flatten()

        return survival_inner_prod

    def get_weights(self):
        """
        Return the current values of the topic-vocabulary weights
        """
        decoder_weight = self.network_weights['beta']
        emb = self.sess.run(decoder_weight)
        return emb

    def get_bg(self):
        """
        Return the current values of the background term
        """
        decoder_weight = self.network_weights['background']
        bg = self.sess.run(decoder_weight)
        return bg

    def get_covar_weights(self):
        """
        Return the current values of the per-covariate vocabulary deviations
        """
        decoder_weight = self.network_weights['beta_c']
        emb = self.sess.run(decoder_weight)
        return emb

    def get_covar_inter_weights(self):
        """
        Return the current values of the interactions terms between topics and covariates
        """
        decoder_weight = self.network_weights['beta_ci']
        emb = self.sess.run(decoder_weight)
        return emb

    def get_label_embeddings(self):
        """
        Return the embeddings of labels used by the encoder
        """
        param = self.network_weights['label_embeddings']
        emb = self.sess.run(param)
        return emb

    def get_covar_embeddings(self):
        """
        Return the embeddings of covariates used by the encoder and decoder
        """
        param = self.network_weights['covariate_embeddings']
        emb = self.sess.run(param)
        return emb

    def get_batch_size(self, X):
        """
        Determine the number of instances in a given minibatch
        """
        if len(X.shape) == 1:
            batch_size = 1
        else:
            batch_size, _ = X.shape
        return batch_size

    def close_session(self):
        self.sess.close()

    def deep_explain(self, x_train, y_train, c_train, x_test, y_test, c_test, feature_names):

        input_train = self.compute_theta(x_train, y_train, c_train)
        input_test = self.compute_theta(x_test, y_test, c_test)
        sample_size = self.sess.run(self.batch_size)

        # select a set of background examples to take an expectation over
        background = input_train[np.random.choice(input_train.shape[0], self.sess.run(self.batch_size), replace=False)]

        print(background.shape)
        print(self.survival_input.shape)
        print(self.survival_inner_prod.shape)
        # print(input_test[1:5].shape)

        # explain predictions of the model on four images
        e = shap.DeepExplainer((self.survival_input, self.survival_inner_prod), background, session=self.sess)
        # ...or pass tensors directly
        # e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
        # print(e.explainer.data[0].shape)
        e.explainer.data = [background[:sample_size//2]]
        # print(e.explainer.data[0].shape)
        # sys.exit(0)
        shap_values = e.shap_values(input_test[:sample_size], check_additivity=False)

        print(shap_values)
        print(shap_values[0].shape)

        shap.summary_plot(shap_values[0][:sample_size//2], sort=False)
        cox_beta = self.sess.run(self.survival_weights).flatten()
        print(cox_beta)

        # # plot the feature attributions
        # shap.image_plot(shap_values, -x_test[1:5])


        # print("[Test] All operations in graph")
        # all_ops = self.sess.graph.get_operations()
        # for op_i, op in enumerate(all_ops):
        #     if op.name.startswith("survival_inner_prod"):
        #         print("[{}]".format(op_i))
        #         print("Operation Name:", op.name)
        #         print("Operation Output:", op.values())
        #         print("--")







