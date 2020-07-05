
'''
    NPSurvival experiment script

'''

import sys, os, argparse, json, pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold
from multiprocessing import Pool
from bayes_opt import BayesianOptimization, UtilityFunction
try:
    from bayes_opt.observer import JSONLogger
except:
    from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from lifelines.utils import concordance_index
from pycox.evaluation import EvalSurv
from progressbar import ProgressBar

parser = argparse.ArgumentParser(description='Configure experiments as follows:')
parser.add_argument('dataset', type=str)
parser.add_argument('model', type=str)
parser.add_argument('n_outer_iter', type=int, help="Number of random train/test split")
parser.add_argument('tuning_scheme', type=str, help="One of 'grid' or 'bayesian'")
parser.add_argument('tuning_config', type=str, help="Path suffix to tuning instructions")
parser.add_argument('experiment_id', type=str, help="Unique identifier for experiment")
parser.add_argument('saved_experiment_id', type=str, help="Unique identifier for saved experiment outputs")
parser.add_argument('readme_msg', type=str, help="Experiment notes")
parser.add_argument('preset_dir', type=str, help="Path suffix to preset parameters (for testing purposes)")
# optional parameter
parser.add_argument('--data_dir', default='dataset', type=str)
parser.add_argument('--log_dir', default='logs', type=str, help="Folder name to store logs")
parser.add_argument('--frac_to_use_as_training', default=0.8, type=float)
parser.add_argument('--seed', default=47, type=int)
parser.add_argument('--cv_fold', default=5, type=int)
# parser.add_argument('--cv_repeats', default=1, type=int, \
#     help="One repeat corresponds to standard cross validation, multiple repeats useful for small datasets")
parser.add_argument('--verbosity', default=1, type=int, help="verbosity = {1,2,3}")
parser.add_argument('--run_unit_tests', default=True, type=bool)

def load_data(path, dataset, suffix):
    X = np.load(os.path.join(path, dataset, "X{}.npy".format(suffix)))
    Y = np.load(os.path.join(path, dataset, "Y{}.npy".format(suffix)))
    F = []
    with open(os.path.join(path, dataset, "F{}.npy".format(suffix)), 'r') as feature_names:
        for line in feature_names.readlines():
            F.append(line.strip())

    if VERBOSE:
        print("Loading {} dataset...\n  Total samples={}\n  Total features={}".format(\
                                                        dataset, X.shape[0], X.shape[1]))
    print("Loading {} dataset...\n  Total samples={}\n  Total features={}".format(\
                                     dataset, X.shape[0], X.shape[1]), file=TRANSCRIPT)
    if TEST:
        assert(Y.shape[1] == 2)
        assert(set(np.unique(Y[:,1])).issubset({0.0, 1.0}))
        assert(X.shape[0] == Y.shape[0])
        assert(X.shape[1] == len(F))
        print("  Test passed!")

    return X,Y,F

def load_config(tuning_config_path):
    f = open(tuning_config_path)
    param_ranges = json.load(f)
    f.close()
    return param_ranges

def load_perset_params(preset_dir):
    if not preset_dir.endswith("None.json"):
        f = open(preset_dir)
        preset_params = json.load(f)
        f.close()
    else:
        preset_params = None
    return preset_params

def get_tuned_params(tuning_scheme, config_dict, model, train_data, cv_fold, outer_iter_i, experiment_dir):
    if tuning_scheme == "grid":
        # TODO
        return get_tuned_params_grid(model, config_dict, train_data, cv_fold, outer_iter_i, experiment_dir)
    elif tuning_scheme == "bayesian":
        return get_tuned_params_bayesian(model, config_dict, train_data, cv_fold, outer_iter_i, experiment_dir)
    elif tuning_scheme == "random":
        return get_tuned_params_random(model, config_dict, train_data, cv_fold, outer_iter_i, experiment_dir)

def get_tuned_params_grid(model, config_dict, train_data, cv_fold, outer_iter_i, experiment_dir):
    raise NotImplementedError 

def get_tuned_params_bayesian(model, config_dict, train_data, cv_fold, outer_iter_i, experiment_dir):
    optimizer = BayesianOptimization(f=get_cross_validated_cindex_fn(model, train_data, cv_fold, outer_iter_i, experiment_dir),
                                     pbounds=config_dict['params'],
                                     verbose=2,
                                     random_state=SEED)

    logger = JSONLogger(path=os.path.join(experiment_dir, "btune_logs_{}.json".format(outer_iter_i)))
    optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

    optimizer.maximize(
        init_points=config_dict['bayesian']['n_init'],
        n_iter=config_dict['bayesian']['n_bayesian']
    )
    return optimizer.max['params']

def get_tuned_params_random(model, config_dict, train_data, cv_fold, outer_iter_i, experiment_dir):
    # note that here we are still using the BayesianOpt package under the hood
    # becuase it allows us to customize sweeping to become entirely random within a specified grid
    optimizer = BayesianOptimization(f=get_cross_validated_cindex_fn(model, train_data, cv_fold, outer_iter_i, experiment_dir),
                                     pbounds=config_dict['params'],
                                     verbose=2,
                                     random_state=SEED)

    logger = JSONLogger(path=os.path.join(experiment_dir, "btune_logs_{}.json".format(outer_iter_i)))
    try:
        optimizer.subscribe(Events.OPTMIZATION_STEP, logger)
    except:
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)


    optimizer.maximize(
        init_points=config_dict['random']['n_probes'],  
        # init_points: How many steps of random exploration you want to perform. 
        n_iter=0 # zero bayesian optimization steps 
    )
    return optimizer.max['params']

def get_cross_validated_cindex_fn(model_name, train_data, cv_fold, outer_iter_i, experiment_dir):
    model_fn = get_model_fn(model_name)
    metric_fn = get_metric_fn(model_name)
    experiment_cv_metrics_path = os.path.join(experiment_dir, "cv_metrics.npy")
    experiment_cv_model_path = os.path.join(experiment_dir, "saved_models", "{}_{}_{}.pickle") # save intermediate models NOT trained on entire training

    def cross_validated_cindex(**kwargs):
        kf = StratifiedKFold(n_splits=cv_fold, shuffle=True, random_state=SEED)        
        metric_table = np.zeros((cv_fold, 7)) # 2 indices and 5 metrics
        inner_iter_i = 0
        for train_index, val_index in kf.split(train_data['x'], train_data['y'][:, 1]):
            train_x, train_y = (train_data['x'][train_index], train_data['y'][train_index])
            val_x, val_y = (train_data['x'][val_index], train_data['y'][val_index])
            observed_time_list = list(np.sort(np.unique(train_y[:, 0])))
            model = model_fn(**kwargs)
            model.fit(train_x, train_y, train_data['f'])

            if model_name in {"survscholar_linear", "lda_cox"}: 
            # predict_lazy only computes hazard scores, and does not compute median survival time or the survival function
            # for models using Cox regression in the end, c-index based on negative hazards is identical to time-dependent c-index
                predicted_neg_hazards, predicted_test_times, predicted_survival_functions = model.predict_lazy(val_x, observed_time_list)
                metrics = metric_fn(predicted_neg_hazards, predicted_survival_functions, val_y)
                metric_table[inner_iter_i] = [outer_iter_i, inner_iter_i,
                                              metrics['concordance_antolini'], metrics['concordance_median'], \
                                              metrics['integrated_brier'], \
                                              metrics['rmse'], metrics['mae']]

            else:
                predicted_test_times, predicted_survival_functions = model.predict(val_x, observed_time_list)
                metrics = metric_fn(predicted_test_times, predicted_survival_functions, val_y)
                metric_table[inner_iter_i] = [outer_iter_i, inner_iter_i,
                                              metrics['concordance_antolini'], metrics['concordance_median'], \
                                              metrics['integrated_brier'], \
                                              metrics['rmse'], metrics['mae']]

            # # Optional step, added when trying the SHAP explainers
            # if model_name.startswith("survscholar"):
            #     model.beta_explain(train_x.astype(dtype='float32'), val_x.astype(dtype='float32'), train_data['f'])
            #     model.kernel_explain(train_x.astype(dtype='float32'), val_x.astype(dtype='float32'), train_data['f'])
            #     model.close_sess()
            #     sys.exit(0)

            # # Optional step, uncomment if you would like to save these intermediate models
            # if model_name not in {"weibull", "weibull_pca", "deepsurv", "deephit", "survscholar_linear", "survscholar_nonlinear", "lda_cox"}:
            #     with open(experiment_cv_model_path.format(outer_iter_i, str(kwargs), inner_iter_i), 'wb') as model_write:
            #         pickle.dump(model, model_write)
            
            if model_name.startswith("survscholar"):
                model.close_sess()

            print(metric_table[inner_iter_i])

            inner_iter_i += 1

        if not os.path.exists(experiment_cv_metrics_path):
            np.save(experiment_cv_metrics_path, metric_table)
        else:
            curr_table = np.load(experiment_cv_metrics_path)
            np.save(experiment_cv_metrics_path, np.append(curr_table, metric_table, axis=0))

        print(">>>> Params: {} Validation Avg C-index: {}".format(kwargs, np.mean(metric_table[:, 2])))
        return np.mean(metric_table[:, 2]) # avg c-index is used for cross validated hyperparam selection
        # this is ignored for random sweeping scheme

    return cross_validated_cindex

def get_model_fn(model_name):

    if model_name == "coxph":
        from survival_baselines.CoxPHModel import CoxPHModel
        return CoxPHModel

    elif model_name == "coxph_pca":
        from survival_baselines.CoxPHModel import CoxPHModel_PCA
        return CoxPHModel_PCA

    elif model_name == "coxph_unreg":
        from survival_baselines.CoxPHModel import CoxPHModel0
        return CoxPHModel0

    elif model_name == 'knnkm':
        from survival_baselines.KNNKaplanMeier import KNNKaplanMeier
        return KNNKaplanMeier

    elif model_name == 'knnkm_pca':
        from survival_baselines.KNNKaplanMeier import KNNKaplanMeier_PCA
        return KNNKaplanMeier_PCA

    elif model_name == 'weibull':
        from survival_baselines.WeibullRegressionModel import WeibullRegressionModel
        return WeibullRegressionModel

    elif model_name == 'weibull_pca':
        from survival_baselines.WeibullRegressionModel import WeibullRegressionModel_PCA
        return WeibullRegressionModel_PCA

    elif model_name == 'rsf':
        from survival_baselines.RandomSurvivalForest import RandomSurvivalForest
        return RandomSurvivalForest

    elif model_name == 'deepsurv':
        from survival_baselines.DeepSurv import DeepSurv_pycox
        return DeepSurv_pycox

    elif model_name == "deephit":
        from survival_baselines.DeepHit import DeepHit_pycox
        return DeepHit_pycox

    elif model_name == "survlda":
        print("SurvLDA requires debugging...")
        raise NotImplementedError
        # from topic_models.survlda.SurvLDA import SurvLDA
        # return SurvLDA

    elif model_name == "survscholar_linear":
        from survival_topics.SurvScholarModel import SurvivalScholarTrainer_Linear
        return SurvivalScholarTrainer_Linear

    elif model_name == "survscholar_nonlinear":
        from survival_topics.SurvScholarModel import SurvivalScholarTrainer_NonLinear
        return SurvivalScholarTrainer_NonLinear

    elif model_name == "lda_cox":
        from survival_topics.LDACox import LDACox
        return LDACox

    raise NotImplementedError

def get_metric_fn(model_name):

    def standard_metric_fn(predicted_test_times, predicted_survival_functions, observed_y):
        # predicted survival function dim: n_train * n_test
        obs_test_times = observed_y[:, 0].astype('float32')
        obs_test_events = observed_y[:, 1].astype('float32')
        results = dict()

        ev = EvalSurv(predicted_survival_functions, obs_test_times, obs_test_events, censor_surv='km')
        results["concordance_antolini"] = ev.concordance_td('antolini')
        results["concordance_median"] = concordance_index(obs_test_times, predicted_test_times, obs_test_events.astype(np.bool))

        # we ignore brier scores at the highest test times because it becomes unstable
        time_grid = np.linspace(obs_test_times.min(), obs_test_times.max(), 100)[:80]
        results["integrated_brier"] = ev.integrated_brier_score(time_grid)

        if sum(obs_test_events) > 0:
            # only noncensored samples are used for rmse/mae calculation
            pred_obs_differences = predicted_test_times[obs_test_events.astype(np.bool)] - obs_test_times[obs_test_events.astype(np.bool)]
            results["rmse"] = np.sqrt(np.mean((pred_obs_differences)**2))
            results["mae"] = np.mean(np.abs(pred_obs_differences))
        else:
            print("[WARNING] All samples are censored.")
            results["rmse"] = 0
            results["mae"] = 0

        return results

    def standard_metric_fn_lazy(predicted_neg_hazards, predicted_survival_functions, observed_y):
        # predicted survival function dim: n_train * n_test
        obs_test_times = observed_y[:, 0].astype('float32')
        obs_test_events = observed_y[:, 1].astype('float32')
        results = dict()

        assert(predicted_survival_functions is None)
        try:
            lifeline_cindex = concordance_index(obs_test_times, predicted_neg_hazards, obs_test_events.astype(np.bool))
        except:
            print("Lifelines detected NaNs in input...")
            lifeline_cindex = 0.0

        results["concordance_antolini"] = lifeline_cindex
        results["concordance_median"] = results["concordance_antolini"]
        results["integrated_brier"] = results["concordance_antolini"]
        results["rmse"] = results["concordance_antolini"]
        results["mae"] = results["concordance_antolini"]

        return results

    def topic_metric_fn(preds, test_y):
        raise NotImplementedError

    if model_name in {'survscholar_linear', 'lda_cox'}:
        return standard_metric_fn_lazy
    else:
        return standard_metric_fn

    # if model_name not in {'survscholar_linear', 'survscholar_nonlinear', 'survlda', 'acnmf', 'lda_cox'}:
    #     return standard_metric_fn
    # else:
    #     return standard_metric_fn

        # return topic_metric_fn
        # other metrics are possible but not yet implemented

def metric_fn_par(args):
    predicted_test_times, predicted_survival_functions, observed_y = args

    # predicted survival function dim: n_train * n_test
    obs_test_times = observed_y[:, 0].astype('float32')
    obs_test_events = observed_y[:, 1].astype('float32')
    results = [0, 0, 0, 0, 0]

    ev = EvalSurv(predicted_survival_functions, obs_test_times, obs_test_events, censor_surv='km')
    results[0] = ev.concordance_td('antolini') # concordance_antolini
    results[1] = concordance_index(obs_test_times, predicted_test_times, obs_test_events.astype(np.bool)) # concordance_median

    # we ignore brier scores at the highest test times because it becomes unstable
    time_grid = np.linspace(obs_test_times.min(), obs_test_times.max(), 100)[:80]
    results[2] = ev.integrated_brier_score(time_grid) # integrated_brier

    if sum(obs_test_events) > 0:
        # only noncensored samples are used for rmse/mae calculation
        pred_obs_differences = predicted_test_times[obs_test_events.astype(np.bool)] - obs_test_times[obs_test_events.astype(np.bool)]
        results[3] = np.sqrt(np.mean((pred_obs_differences)**2)) # rmse
        results[4] = np.mean(np.abs(pred_obs_differences)) # mae
    else:
        print("[WARNING] All samples are censored.")
        results[3] = 0
        results[4]  = 0

    return results

def metric_fn_par_lazy(args):
    predicted_test_times, predicted_survival_functions, observed_y = args
    # predicted_test_times is actually predicted_neg_hazards here

    # predicted survival function dim: n_train * n_test
    obs_test_times = observed_y[:, 0].astype('float32')
    obs_test_events = observed_y[:, 1].astype('float32')

    assert(predicted_survival_functions is None) 
    try:
        lifeline_cindex = concordance_index(obs_test_times, predicted_test_times, obs_test_events.astype(np.bool))
    except:
        print("Lifelines detected NaNs in inputs...")
        lifeline_cindex = 0.0

    result = lifeline_cindex
    results = [result] * 5

    return results

if __name__ == '__main__':
    args = parser.parse_args()
    topic_models = {'survscholar_linear', 'survscholar_nonlinear', 'survlda', 'acnmf', 'lda_cox'}
    model_name = args.model
    dataset = args.dataset
    log_dir = args.log_dir
    data_dir = args.data_dir
    experiment_id = args.experiment_id
    n_outer_iter = args.n_outer_iter
    frac_to_use_as_training = args.frac_to_use_as_training
    tuning_scheme = args.tuning_scheme
    tuning_config = args.tuning_config # configuration's version number
    preset_dir = args.preset_dir
    cv_fold = args.cv_fold
    saved_experiment_id = args.saved_experiment_id
    readme_msg = args.readme_msg
    
    experiment_dir = os.path.join(log_dir, dataset, model_name, experiment_id)
    experiment_log_path = os.path.join(experiment_dir, "transcript.txt")
    experiment_model_path = os.path.join(experiment_dir, "saved_model_{}.pickle") # models trained on entire training data
    experiment_metrics_path = os.path.join(experiment_dir, "metrics.npy")

    saved_experiment_dir = os.path.join(log_dir, dataset, model_name, saved_experiment_id)

    if readme_msg != "None":
        readme_msg_file = open(os.path.join(experiment_dir, "exp_readme.txt"), "w")
        readme_msg_file.write(readme_msg)
        readme_msg_file.close()

    global VERBOSE, TEST, SEED
    VERBOSE = args.verbosity
    TEST = args.run_unit_tests
    TRANSCRIPT = open(experiment_log_path, 'w')
    SEED = args.seed

    if model_name in {"coxph_unreg", "coxph"}:
        dataset_suffix = "_cox" # remove one reference column for linear regression with intercept (baseline hazard)
    elif model_name in topic_models:
        dataset_suffix = "_discretized"
    else:
        dataset_suffix = ""

    X,Y,F = load_data(data_dir, dataset, suffix=dataset_suffix)
    config_dict = load_config(os.path.join("configurations", "{}-{}-{}.json".format(dataset, model_name, tuning_config)))
    preset_params = load_perset_params(os.path.join("configurations", "{}-{}-{}-{}.json".format(dataset, model_name, tuning_config, preset_dir)))

    if VERBOSE:
        print("Configuring experiments...\n  Model={}\n  Params={}".format(model_name, config_dict))
        print("  Tuning scheme={}".format(tuning_scheme))
        print("  CV folds={}; Train/Test Split Repeats={}".format(cv_fold, n_outer_iter))
        print("Runing {} train/test splits for error bars...\n  Training fraction={}\n  Random seed={}".format(n_outer_iter, frac_to_use_as_training, SEED))

    print("Configuring experiments...\n  Model={}\n  Params={}".format(model_name, config_dict), file=TRANSCRIPT)
    print("  Tuning scheme={}".format(tuning_scheme), file=TRANSCRIPT)
    print("  CV folds={}; Train/Test Split Repeats={}".format(cv_fold, n_outer_iter), file=TRANSCRIPT)
    print("Runing {} train/test splits for error bars...\n  Training fraction={}\n  Random seed={}".format(n_outer_iter, frac_to_use_as_training, SEED), file=TRANSCRIPT)

    train_test_random_spliter = StratifiedKFold(n_splits=n_outer_iter, shuffle=True, random_state=SEED)
    # train_test_random_spliter = ShuffleSplit(n_splits=n_outer_iter, train_size=frac_to_use_as_training, random_state=SEED)
    model_fn = get_model_fn(model_name)
    metric_fn = get_metric_fn(model_name)

    metric_table = np.zeros((n_outer_iter, 5)) # right now, we log 5 different metrics
    # concordance_antolini, concordance_median, integrated_brier, rmse, mae
    outer_iter_i = 0

    for train_index, test_index in train_test_random_spliter.split(X, Y[:, 1]):  # preserve censor rate between train and test
        
        # For producing bootstrapped test outputs: only look at iter 0
        if outer_iter_i != 0:
            outer_iter_i += 1
            continue

        print("  >> Iter {} in progress...".format(outer_iter_i))

        train_data = {"x": X[train_index], "y": Y[train_index], "f": F}
        test_data = {"x": X[test_index], "y": Y[test_index], "f": F}
        train_x, train_y = (train_data['x'], train_data['y'])
        test_x, test_y = (test_data['x'], test_data['y'])
        observed_time_list = list(np.sort(np.unique(train_y[:, 0])))

        if saved_experiment_id != "None":
            # load presaved models
            if model_name.startswith("survscholar"):
                with open(os.path.join(saved_experiment_dir, "saved_best_params_{}.pickle").format(outer_iter_i), 'rb') as model_read:
                    best_params = pickle.load(model_read)
                # best_params['saved_model'] = os.path.join(saved_experiment_dir, "saved_model_{}").format(outer_iter_i)
                print("  >> Iter {} best params: ".format(outer_iter_i), best_params)
                print("  >> Iter {} best params: ".format(outer_iter_i), best_params, file=TRANSCRIPT)

                model = model_fn(**best_params)
                model.fit(train_x, train_y, train_data['f']) 
                # although it does not re-train, scholar requires this function to be called to load the saved graph

            elif model_name in {"weibull", "weibull_pca", "deepsurv", "deephit"}:
                raise NotImplementedError

            else:
                with open(os.path.join(saved_experiment_dir, "saved_model_{}.pickle").format(outer_iter_i), 'rb') as model_read:
                    model = pickle.load(model_read)

        else:
            # hyperparam sweeping
            if model_name in {"coxph_unreg", "weibull"}: # no hyperparam to sweep over
                best_params = {}
            elif preset_params is not None:
                best_params = preset_params
            else:
                best_params = get_tuned_params(tuning_scheme, config_dict, model_name, train_data, cv_fold, \
                                               outer_iter_i, experiment_dir)  # for saving logs

            # save best params
            with open(os.path.join(experiment_dir, "saved_best_params_{}.pickle").format(outer_iter_i), 'wb') as model_write:
                pickle.dump(best_params, model_write)

            print("  >> Iter {} best params: ".format(outer_iter_i), best_params)
            print("  >> Iter {} best params: ".format(outer_iter_i), best_params, file=TRANSCRIPT)

            model = model_fn(**best_params)
            model.fit(train_x, train_y, train_data['f'])

        if "explain" in experiment_id:

            if model_name == "coxph" or model_name == "coxph_unreg":
                # here explain means beta coefficients

                getcoxbeta_fname_pickle = os.path.join(experiment_dir, "{}_{}_{}_{}_beta.pickle".format(model_name, dataset, experiment_id, outer_iter_i))
                getcoxbeta_fname_text = os.path.join(experiment_dir, "{}_{}_{}_{}_beta.txt".format(model_name, dataset, experiment_id, outer_iter_i))

                sorted_beta_indices = np.argsort(-abs(model.beta))
                sorted_beta = model.beta[sorted_beta_indices]
                sorted_feature_names = np.array(model._feature_names)[sorted_beta_indices]
                n_nonzero_betas = sum(sorted_beta != 0)

                result_strs = [">>> Cox PH regression (regularized) beta coefficients sorted by absolute value : {}".format(dataset),
                               ">>> Total number of features : {}".format(len(model.beta)),
                               ">>> Number of features with nonzero betas: {}\n".format(n_nonzero_betas)]
                for b_i in range(n_nonzero_betas):
                    curr_beta = sorted_beta[b_i]
                    curr_feature = sorted_feature_names[b_i]
                    curr_str = "{} : ({})".format(curr_feature, curr_beta)
                    result_strs.append(curr_str)

                all_str = "\n".join(result_strs)
                
                with open(getcoxbeta_fname_text, "w") as coxfile:
                    coxfile.write(all_str)
    
                cox_beta_explain = dict()
                cox_beta_explain['beta'] = model.beta
                cox_beta_explain['features'] = model._feature_names
                with open(getcoxbeta_fname_pickle, 'wb') as coxfile:
                    pickle.dump(cox_beta_explain, coxfile)

            elif model_name in topic_models:
                model.beta_explain(feature_names=train_data['f'], \
                    save_path=os.path.join(experiment_dir, "{}_{}_{}_{}_beta.pickle".format(model_name, dataset, experiment_id, outer_iter_i)))
                # model.kernel_explain(train_x.astype(dtype='float32'), test_x.astype(dtype='float32'), train_data['f'],
                #                     save_path=os.path.join(experiment_dir, "{}_{}_{}_{}_kernel_shap.pickle".format(model_name, dataset, experiment_id, outer_iter_i)))

        if "bootstrap_predictions" in experiment_id:

            if model_name in {"survscholar_linear", "lda_cox"}:
                predicted_neg_hazards, predicted_test_times, predicted_survival_functions = model.predict_lazy(test_x, observed_time_list)
                metrics = metric_fn(predicted_neg_hazards, predicted_survival_functions, test_y)
                metric_table[outer_iter_i] = [metrics['concordance_antolini'], metrics['concordance_median'], \
                                              metrics['integrated_brier'], \
                                              metrics['rmse'], metrics['mae']]

            else:
                predicted_test_times, predicted_survival_functions = model.predict(test_x, observed_time_list)
                metrics = metric_fn(predicted_test_times, predicted_survival_functions, test_y)
                metric_table[outer_iter_i] = [metrics['concordance_antolini'], metrics['concordance_median'], \
                                              metrics['integrated_brier'], \
                                              metrics['rmse'], metrics['mae']]

            np.save(experiment_metrics_path, metric_table) # save and update most recent metrics to prevent crashing

            print("  >> Iter {} metrics: ".format(outer_iter_i), metrics)
            print("  >> Iter {} metrics: ".format(outer_iter_i), metrics, file=TRANSCRIPT)

            print("  >> Entering prediction bootstrapping...")
            bootstrap_B = 1000
            
            if predicted_test_times is None: # in this case we only predicted negative hazard scores
                n_test = predicted_neg_hazards.shape[0]
                np.random.seed(SEED)
                bootstrap_pool_metrics = []
                pbar = ProgressBar()
                for B_i in pbar(list(range(bootstrap_B))):
                    Bsample_indices = np.random.choice(n_test, size=n_test, replace=True)
                    predicted_neg_hazards_Bsamples = predicted_neg_hazards[Bsample_indices]
                    test_y_Bsamples = test_y[Bsample_indices]
                    bootstrap_pool_metrics.append(metric_fn_par_lazy((predicted_neg_hazards_Bsamples, None, test_y_Bsamples)))
            else:
                n_test = predicted_test_times.shape[0]
                # bootstrapped_predictions_inputs = []
                bootstrap_rng = np.random.RandomState(SEED)
                bootstrap_pool_metrics = []
                pbar = ProgressBar()
                for B_i in pbar(list(range(bootstrap_B))):
                    predicted_survival_functions_Bsamples = predicted_survival_functions.sample(n=n_test, replace=True, random_state=bootstrap_rng, axis=1)
                    Bsample_indices = np.array(predicted_survival_functions_Bsamples.columns, dtype=int)
                    predicted_test_times_Bsamples = predicted_test_times[Bsample_indices]
                    test_y_Bsamples = test_y[Bsample_indices]
                    bootstrap_pool_metrics.append(metric_fn_par((predicted_test_times_Bsamples, predicted_survival_functions_Bsamples, test_y_Bsamples)))
            
            bootstrap_pool_metrics = np.array(bootstrap_pool_metrics)
            np.save(os.path.join(experiment_dir, "metrics_bootstrapped.npy"), bootstrap_pool_metrics)

            print("  >> Iter {} bootstrapped : MEAN".format(outer_iter_i), np.mean(bootstrap_pool_metrics, axis=0))
            print("  >> Iter {} bootstrapped : MEDIAN".format(outer_iter_i), np.median(bootstrap_pool_metrics, axis=0))
            
            print("  >> Iter {} bootstrapped : Q=0.025".format(outer_iter_i), np.quantile(bootstrap_pool_metrics, q=0.025, axis=0))
            print("  >> Iter {} bootstrapped : Q=0.975".format(outer_iter_i), np.quantile(bootstrap_pool_metrics, q=1-0.025, axis=0))
            

            print("  >> Iter {} bootstrapped : MEAN".format(outer_iter_i), np.mean(bootstrap_pool_metrics, axis=0), file=TRANSCRIPT)
            print("  >> Iter {} bootstrapped : MEDIAN".format(outer_iter_i), np.median(bootstrap_pool_metrics, axis=0), file=TRANSCRIPT)
            
            print("  >> Iter {} bootstrapped : Q=0.025".format(outer_iter_i), np.quantile(bootstrap_pool_metrics, q=0.025, axis=0), file=TRANSCRIPT)
            print("  >> Iter {} bootstrapped : Q=0.975".format(outer_iter_i), np.quantile(bootstrap_pool_metrics, q=1-0.025, axis=0), file=TRANSCRIPT)

        else: # no bootstrapping

            predicted_test_times, predicted_survival_functions = model.predict(test_x, observed_time_list)
            metrics = metric_fn(predicted_test_times, predicted_survival_functions, test_y)

            metric_table[outer_iter_i] = [metrics['concordance_antolini'], metrics['concordance_median'], \
                                          metrics['integrated_brier'], \
                                          metrics['rmse'], metrics['mae']]
            np.save(experiment_metrics_path, metric_table) # save and update most recent metrics to prevent crashing

            print("  >> Iter {} metrics: ".format(outer_iter_i), metrics)
            print("  >> Iter {} metrics: ".format(outer_iter_i), metrics, file=TRANSCRIPT)

        if model_name == "survscholar_linear": # optional, only needed if you need to save the topic model's theta matrix
            thetas = model.theta
            theta_path = os.path.join(experiment_dir, "{}_{}_{}_{}_theta.npy".format(model_name, dataset, experiment_id, outer_iter_i))
            y_path = os.path.join(experiment_dir, "{}_{}_{}_{}_testy.npy".format(model_name, dataset, experiment_id, outer_iter_i))
            np.save(theta_path, thetas)
            np.save(y_path, test_y)
            print("Theta saved!")

        # save model as an object, not available for WeibullRegression, 
        # which could not be pickled due to underlying R implementation
        # if model_name not in {"weibull", "weibull_pca", "deepsurv", "deephit", "survscholar_linear", "survscholar_nonlinear"}:
        #     with open(experiment_model_path.format(outer_iter_i), 'wb') as model_write:
        #         pickle.dump(model, model_write)
        if model_name.startswith("survscholar"):
            model.save_to_disk(os.path.join(experiment_dir, "saved_model_{}").format(outer_iter_i))
            model.close_sess()

        outer_iter_i += 1

    if "bootstrap_predictions" in experiment_id:
        print("Finished!")
        print("Finished!", file=TRANSCRIPT)
    else:
        print("Finished! Test metrics mean and std:", np.mean(metric_table, axis=0), np.std(metric_table, axis=0))
        print("Finished! Test metrics mean and std:", np.mean(metric_table, axis=0), np.std(metric_table, axis=0), file=TRANSCRIPT)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(metric_table)
        print(metric_table, file=TRANSCRIPT)

    TRANSCRIPT.close()












