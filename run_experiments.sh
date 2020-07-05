
dataset=METABRIC               # Name of dataset, must match a folder name in the dataset directory
model=survscholar_linear       # Name of model, must be one of the supported models' names 
n_outer_iter=5                 # Number of random train/test splits, this is not the same as number of folds in cross-validation
tuning_scheme=random           # Hyperparameter sweeping scheme, either random or bayesian
tuning_config=demo             # Suffix identifier for configuration json file, which contains hyperparams' search box
log_dir=logs                   # Directory to save experiment outputs to (trained models, best parameters, transcripts etc. ) (no need to modify)
experiment_id=bootstrap_predictions_demo_explain  # Should be named as a unique identifier for an experiment, 
                                               # "bootstrap_predictions" must be in the name to turn on the bootstrapping option, 
                                               # "explain" must be in the name to turn on the explainer option
saved_experiment_id=None       # If you would like to load a trained model from a previous experiment, put the previous experiment's experiment_id here
saved_model_path=saved_models  # Sub-directory to save trained models to (no need to modify)
readme_msg=EnterAnyMessageHere # An option to add comments to this experiment
preset_dir=None                # Suffix identifier for configuration json file, which contains a preset set of hyperparameters

mkdir -p ${log_dir}/${dataset}/${model}/${experiment_id}/${saved_model_path}

python3 experiments.py ${dataset} ${model} ${n_outer_iter} ${tuning_scheme} ${tuning_config} ${experiment_id} ${saved_experiment_id} ${readme_msg} ${preset_dir} --log_dir ${log_dir}


# for data in SUPPORT_Coma SUPPORT_Cancer SUPPORT_COPD_CHF_Cirrhosis SUPPORT_ARF_MOSF
# do
#     dataset=${data} 
#     model=weibull_pca
#     n_outer_iter=5
#     tuning_scheme=random
#     tuning_config=vanilla
#     log_dir=logs
#     experiment_id=bootstrap_predictions_0_explain
#     saved_experiment_id=None
#     saved_model_path=saved_models
#     readme_msg=None
#     preset_dir=None

#     mkdir -p ${log_dir}/${dataset}/${model}/${experiment_id}/${saved_model_path}

#     python3 experiments.py ${dataset} ${model} ${n_outer_iter} ${tuning_scheme} ${tuning_config} ${experiment_id} ${saved_experiment_id} ${readme_msg} ${parallel} ${preset_dir} --log_dir ${log_dir}

# done