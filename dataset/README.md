# Survival Datasets

Because some datasets require applying for access or are simply too large, we did not include the actual data in this repo. To be able to run any experiments, after git cloning, make sure to populate this **dataset/** directory with actual data. A dataset should correspond to a folder with the same name as the dataset and containing all files detailed in [this section].

## Table of Contents

* [Dataset Descriptions]
  * [SUPPORT]
  * [UNOS]
  * [METABRIC]
  * [MIMIC(Ich)]
* [Preprocessing]
* [Preprocessed Format]

## Dataset Descriptions

| Dataset  | Descriptions | # Subjects | # Features | % Censored |
| -------- | ------------ | ---------- | ---------- | ---------- |
| SUPPORT-1  | acute resp. failure/multiple organ sys. failure  | 4194 | 14 | 35.6% |
| SUPPORT-2  | COPD/congestive heart failure/cirrhosis  | 2804 | 14 | 38.8% |
| SUPPORT-3  | cancer  | 1340 | 13 | 11.3% |
| SUPPORT-4  | coma  | 591 | 14 | 18.6% |
| UNOS  | heart transplantation | 62644 | 49 | 50.2% |
| METABRIC  | breast cancer  | 1981 | 24 | 55.2% |
| MIMIC-Ich  | intracerebral hemorrhage  | 1010 | 1157 | 0% |

### SUPPORT

The dataset from the Study to Understand Prognoses Preferences Outcomes and Risks of Treatment (SUPPORT)[[1]](#1) is freely available [online](http://biostat.mc.vanderbilt.edu/wiki/Main/SupportDesc). This dataset contains 14 clinical features collected from seriously ill hospitalized adults, such as their age, presence of cancer, and neurologic function. These features were collected from patients on the third day after the study started, and patients were followed for survival until 180 days after the study entry. For our purposes, the dataset was split into four datasets corresponding to different disease groups (acute respiratory failure/multiple organ system failure, cancer, coma, COPD/congestive heart failure/cirrhosis), as done by [[2]](#2). After we created these four subsets, all subjects from the cancer group have identical values for a clinical feature related to cancer presence, so this feature was removed only for the cancer cohort, resulting in 13 clinical features for the **SUPPORT-3** dataset.

## Preprocessing

Each datasets came in in different formats, and we added these datasets throughout the time I worked on this project. Therefore, there is currently no unified data preprocessing script, and each dataset is handled slighly differently by the following scripts:

#### [METABRIC & SUPPORT](preprocessing_metabric_support.py)

#### [MIMIC2 Datasets: Ich, Pancreatitis, Sepsis](preprocessing_mimic2.py)
Feature engineering logic is implemented [here](preprocessing_mimic2_lib/FeatureEngineerNew2.py). 

#### [UNOS](preprocessing_unos.ipynb)
This notebook tries to recover UNOS's preprocessing in some previous papers. Please do not share this file with unauthorized users of the UNOS data.

#### The three data formats
Across scripts, you will find that there are three data formats one could output, "cox", "original", and "discretize". 

- **Mode "cox"** refers to one-hot encoding all categorical features, with one reference column removed per categorical feature; each feature that has missing entries will also have a separate feature indicating whether a subject is missing information for this feature; this format should be used for any models that directly apply Cox regression on input data. 

- **Mode "original"** refers to one-hot encoding all categorical features, **without** removing a reference column per feature; each feature that has missing entries will also have a separate feature indicating whether a subject is missing information for this feature; this format should be used for any models that do not directly apply Cox regression, or topic modelling on input data.

- **Mode "discretize"** refers to one-hot encoding all categorical features, **without** removing a reference column per feature; besides, continuous features are discretized by percentiles; also, there are no features that explicitly encode missingness like the two formats above; this format should be used for any models that apply topic modelling on input data. 

Applying different formats will give you datasets with different number of features. For sanity checks, please check the dataset you use against the dimensions I had below:

| Dataset  | # Subjects | # Features (cox) | # Features (original) | # Features (discretize) | 
| -------- | ---------- | ---------------- | --------------------- | ----------------------- | 
| SUPPORT-1  | 4194 | 18 | 21 | 55 |
| SUPPORT-2  | 2804 | 18 | 21 | 56 |
| SUPPORT-3  | 1340 | 16 | 18 | 52 |
| SUPPORT-4  | 591 | 18 | 21 | 55 | 
| UNOS  | 62644 | 104 | 127 | 183 |
| METABRIC  | 1981 | 81 | 100 | 107 |
| MIMIC-Ich  |  1010 | 4885 | 4939 | 4229 | 
| MIMIC-Pancreatitis  | 371 | 1858 | 1926 | 1985 |
| MIMIC-Sepsis  |  12612 | 3871 | 3896 | 4030 |

## Processed Datasets

All processed datasets live in their individual folders, in experiment-ready format. In each dataset's folder, you will find:

- *X.npy*: Feature matrix, where each row is a patient, and each column is a feature. Taken in by models. 
- *Y.npy*: Labels, with a column containing survival times, and a column containing censoring indicators. Taken in by models.
- *F.npy*: Feature names. Taken in by models.
- *feature_list.txt*: Feature names, for humans to read.

Each one of the three data formats will have a set of these. The files' names should explain themselves.

If you want to add in a new dataset in the folder, simply make sure the set of files described above is available under a directory named after the dataset.

## References
<a id="1">[1]</a> 
W. A. Knaus, F. E. Harrell, J. Lynn, L. Goldman, R. S. Phillips, A. F. Connors,N. V. Dawson, W. J. Fulkerson, R. M. Califf, and N. Desbiens. The SUPPORTprognostic model: Objective estimates of survival for seriously ill hospitalizedadults.Annals of Internal Medicine, 122(3):191–203, 1995.

<a id="1">[2]</a> 
F. E. Harrell Jr.Regression  Modeling  Strategies:  With  Applications  to  LinearModels,  Logistic  and  Ordinal  Regression,  and  Survival  Analysis.   Springer,2015.
