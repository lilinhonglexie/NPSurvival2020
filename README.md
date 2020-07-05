# NPSurvival2020

Public repository for Neural Topic Models with Survival Supervision

## Table of Content

* [Requirements](#required-packages)
* [Datasets & Models](#required-packages)
* [Experiments & Demo](#required-packages)
  * [Model Outputs](#required-packages)

## Required packages

Package requirements could be found [here](requirements.txt). You could set up the required environment in the command line: 

```
>>> python3 -m venv env_npsurvival
>>> source env_npsurvival/bin/activate
>>> pip3 install -r Survival2019/requirements.txt
```

## Datasets and Models

Data preprocessing details, scripts and a list of supported datasets are documented [here](dataset/). A list of implemented survival models could be found below. The **Data Format** column is explained in detail on [this page](dataset/). Each model is linked to its implementation script.

| Model  | Descriptions | Type | Data Format |
| ------ | ------------ | ---- | ----------- |
| [coxph](survival_baselines/CoxPHModel.py)  | Cox regression with lasso regularization | baseline | cox |
| [coxph_pca](survival_baselines/survival_baselines/CoxPHModel.py) | Lasso-regularized cox preceded by PCA | baseline | original |
| [coxph_unreg](survival_baselines/CoxPHModel.py) | Unregualrized cox regression        | baseline | cox |
| [knnkm](survival_baselines/KNNKaplanMeier.py) | KNN-Kaplan-Meier | baseline | original |
| [knnkm_pca](survival_baselines/KNNKaplanMeier.py) | KNN-Kaplan-Meier preceded by PCA | baseline | original |
| [weibull](survival_baselines/WeibullRegressionModel.py) | Weibull regression | baseline | original |
| [weibull_pca](survival_baselines/WeibullRegressionModel.py) | Weibull regression preceded by PCA | baseline | original |
| [rsf](survival_baselines/RandomSurvivalForest.py) | Survival random forest | baseline | original |
| [deepsurv](survival_baselines/DeepSurv.py) | [DeepSurv](https://github.com/havakv/pycox) | baseline | original |
| [deephit](survival_baselines/DeepHit.py) | [DeepHit](https://github.com/havakv/pycox) | baseline | original |
| [lda_cox](survival_topics/LDACox.py) | Cox regression preceded by LDA | topic | discretize |
| [survscholar_linear](survival_topics/SurvScholarModel.py) | Supervised cox regression preceded by [scholar](https://github.com/dallascard/scholar) | topic | discretize |
| [survscholar_nonlinear](survival_topics/SurvScholarModel.py) | Supervised cox regression preceded by scholar, with nonlinear survival layers | topic | discretize |

## Running Experiments

To run an experiment:

1. ```git clone``` this repo to a local directory.
2. Make sure all required packages are installed (see section **Required packages**).
3. ```cd``` into the repo directory, replace the ```dataset/``` folder with one that actually contains the data. On George's server, such folder could be found at ```/media/latte/npsurvival/dataset```.
4. Make sure hyperparameter search boxes are configured in a ```json``` file under ```configurations/```. You could find plenty of examples [here](configurations/). 
5. Modify experiment settings in the bash script ```run_experiments.sh```, and type ```sh run_experiments.sh``` in the command line.
6. This will kick off the experiment. Be sure to name experiments properly using the ```experiment_id``` option, and note that rerunning using the same ```experiment_id``` will erase saved outputs from the last experiment with the same ```experiment_id```. Results reported in the submitted paper are also saved under their respective ```experiment_id```, so make sure you are not erasing them by mistake.

### Demo: SurvScholar + METABRIC

Follow this demo to see how experiments are configured.

1. For all experiments, hyperparameter search configuration should be specificied using a ```json``` file under ```configurations/```. The ```json``` file's naming convention should follow ```dataset-model-suffix_identifier.json```. For this demo, we use [```METABRIC-survscholar_linear-demo.json```](configurations/METABRIC-survscholar_linear-demo.json). 

```
{"params": {"n_topics": [1, 10], "survival_loss_weight": [0, 5], "batch_size": [32, 1024]}, 
 "random": {"n_probes": 5}}
```

By such configuration:

- We search three hyperparameters' values: ```n_topics```, ```survival_loss_weight```, ```batch_size```
- For ```n_topics```, we search over the range: \[1, 10\]
- For ```survival_loss_weight```, we search over the range: \[10^0, 10^5\]. The exponentiation is done within the model's implementation, meaning that if the model takes in ```survival_loss_weight``` as 5, it converts ```survival_loss_weight``` into 10^5. This serves as an example that the user should always check the code and make sure to understand how configurations are set.
- For ```batch_size```, we search over the range: \[32, 1024\]
- We use random sweeping, with only 5 random attemps. (We only try 5 different hyperparameter combinations within the specified ranges.)

2. Modify settings in the bash script to specify which dataset and model to use, name the experiment, and specify whether a previously trained model should be loaded etc. Details are documented in the bash script [```run_experiments.sh```](run_experiments.sh). For this demo:

```
dataset=METABRIC               
model=survscholar_linear       
n_outer_iter=5                 
tuning_scheme=random           
tuning_config=demo       # this will locate the configuration json file to be METABRIC-survscholar_linear-demo.json         
log_dir=logs                   
experiment_id=bootstrap_predictions_demo_explain 
saved_experiment_id=None       
saved_model_path=saved_models  
readme_msg=EnterAnyMessageHere 
preset_dir=None                

mkdir -p ${log_dir}/${dataset}/${model}/${experiment_id}/${saved_model_path}

python3 experiments.py ${dataset} ${model} ${n_outer_iter} ${tuning_scheme} ${tuning_config} ${experiment_id} ${saved_experiment_id} ${readme_msg} ${preset_dir} --log_dir ${log_dir}
```

Experiment outputs will be saved to ```${log_dir}/${dataset}/${model}/${experiment_id}/```. For this demo, this evaluates to [```logs/METABRIC/survscholar_linear/bootstrap_predictions_demo_explain/```](logs/METABRIC/survscholar_linear/bootstrap_predictions_demo_explain). Some experiment outputs, such as saved models, and ```pickle``` objects used for plotting the heatmaps are not synced to GitHub (see [```.gitignore```](.gitignore)). You will need to navigate to the local directory to obtain those.

As documented in the experiment transcript [here](logs/METABRIC/survscholar_linear/bootstrap_predictions_demo_explain/transcript.txt), using only 5 random hyperparameter combinations, we get mean bootstrapped time-dependent c-index **0.66058302** on the test set. (95% confidence interval **\[0.62127882, 0.70199634\]**).

## Heatmaps

For SurvScholar, [this notebook](run_visualizations.ipynb) demonstrates how to obtain the all-topic heatmaps, single-topic heatmaps, and per topic top-words printouts. Running visualization requires you to specify a directory that contains the saved model outputs, which is usually ```${log_dir}/${dataset}/${model}/${experiment_id}/```. In the notebook, we used an experiment on the SUPPORT_Cancer dataset, whose ```experiment_id``` is ```bootstrap_predictions_3_explain```. 
