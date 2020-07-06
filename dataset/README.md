# Survival Datasets

We tested baselines and our proposed approach on 7 heathcare datasets, whose origins and preprocessing we describe below.

Because some datasets require applying for access or are simply too large, we did not include the actual data in this repo. To be able to run any experiments, after git cloning, make sure to populate this **dataset/** directory with actual data. A dataset should correspond to a folder with the same name as the dataset and containing all files detailed in [this section](#processed-datasets).

## Table of Contents

* [Dataset Descriptions](#dataset-descriptions)
  * [SUPPORT](#support)
  * [UNOS](#unos)
  * [METABRIC](#metabric)
  * [MIMIC(Ich)](#mimicich)
* [Preprocessing](#preprocessing)
  * [Features & Data Formats](#preprocessing-explained)
  * [Scripts](#preprocessing-scripts)
* [Preprocessing Outcome](#processed-datasets)

## Dataset Descriptions

A summary table followed by detailed description for each dataset.

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

The dataset from the Study to Understand Prognoses Preferences Outcomes and Risks of Treatment (SUPPORT) [[1]](#1) is freely available [online](http://biostat.mc.vanderbilt.edu/wiki/Main/SupportDesc). This dataset contains 14 clinical features collected from seriously ill hospitalized adults, such as their age, presence of cancer, and neurologic function. These features were collected from patients on the third day after the study started, and patients were followed for survival until 180 days after the study entry. For our purposes, the dataset was split into four datasets corresponding to different disease groups (acute respiratory failure/multiple organ system failure, cancer, coma, COPD/congestive heart failure/cirrhosis), as done by [[2]](#2). After we created these four subsets, all subjects from the cancer group have identical values for a clinical feature related to cancer presence, so this feature was removed only for the cancer cohort, resulting in 13 clinical features for the **SUPPORT-3** dataset.

### UNOS

The UNOS dataset was extracted from the United Network for Organ Sharing (UNOS) [database](https://www.unos.org/data/), and curated in order to replicate the pre-processing documented by [[3]](#3) and [[4]](#4). We selected only patients who went through heart transplantations in the 30-year window from January 1985 to December 2015. Because [[4]](#4) did not document the exact list of feature names that we could directly extract from the database, we attempted to the best of our ability to curate a list of features that overlaps the most with the feature table presented by them. We ended up with 49 features in total, among which 31 are recipient-related, 12 are donor-related, 6 are compatibility related. For this dataset, our objective is to predict patients' post-transplantation survival time. Because we assumed December 2015 to be the end of data collection, patients who were still alive as of December 2015 are all considered censored samples. Among 62644 patients who underwent transplantation, around 50.2% are censored samples.

### METABRIC

We obtained the Molecular Taxonomy of Breast Cancer International Consortium (METABRIC) dataset from [the Synapse platform](https://www.synapse.org/). The METABRIC dataset contains clinical and genetic features from breast cancer patients, and their respective survival durations. We only used a subset of 24 features that are available for open use through Synapse. This dataset includes 1981 breast cancer patients in total, around 55.2% of whom were censored and not followed until death. The original METABRIC paper [[5]](#5) discusses how the dataset's clinical features were defined in more detail. 

### MIMIC(Ich)

The intracerebral hemorrhage (ICH) dataset we evaluated on is created from MIMIC-III (version 1.4), a critical care health records database containing 52 thousand individuals and their hospital encounters involving admission to the ICU at Beth Israel Deaconess center between 2001 and 2012 [[6]](#6). Experiments were conducted using a subset of the MIMIC-III data consisting of patients having spontaneous intracerebral hemorrhage requiring admission to the ICU. Patients were included in the study if they have an ICU admission with a primary billing code of intracerebral hemorrhage, resulting in a cohort of 1010 individuals. For patients who are admitted to the ICU multiple times, we only consider their first visit to the ICU. This subset of the data has no right-censoring.

Features extracted include demographics, medications, billing codes, procedures, laboratory measurements, events recorded into charts, and vitals. Features were extracted from the relational database into a 4-column format for *patient id*, *time*, *event*, and *event value*. To prevent erroneous merging of different events into a single event, and to provide more informative events, event strings are concatenations of the event descriptor prefixed with the table from which they are derived and additional relevant information such as measurement type, measurement units, etc. Because events recorded in charts are sometimes automated and sometimes manually entered, a physician-developed mapping and lower-casing all fields were used to resolve duplicate entries. As we aim to predict the patient length of stay in ICU, we extract clinical events from the subjects' electronic health records strictly before ICU admission. After preprocessing, the total number of features used for prediction is 1157.

## Preprocessing

### Preprocessing Explained

For all of our datasets, categorical features were one-hot encoded. Specifically to the Cox proportional hazards and lasso-regularized Cox baselines, for each categorical feature, one category was removed as the reference column. For methods that use topic modeling, we realized it does not make sense to encode numeric clinical events as they are. Instead, numeric clinical events were treated as categorical by mapping observed values to equally spaced ranges by quantile (5 bins of roughly equal number of subjects per bin). When values of a numeric clinical event are highly cluttered, the number of bins could go below 5 and result in bins with unequal number of subjects.

#### Features for the MIMIC(Ich) dataset were created slightly differently.

Our definition of clinical events mean that a subject can have multiple instances of one event; for example, one patient might have multiple results for a particular lab test on file. Under this case, single-occurrence categorical events (e.g., gender) were one-hot encoded as usual; multiple-occurrence categorical events (e.g., urine color) were encoded by counting each category's occurrences in a single subject's records. For numeric clinical events, as a subject may have a list of numeric values recorded, we engineered numeric features that captured the minimum, maximum, median, and length of a subject's list of recordings. However, this was not necessary for methods that use topic modeling, because mapping values to equally spaced bins took care of multiple-occurrence numeric events for us.

#### Handling Missingness

We would also like to note that missing records were not imputed as missing certain events can have clinical significance. Therefore, for features with incomplete records, the missing entries were first filled with zeros, and then an additional feature was added solely to indicate whether missingness is observed for each subject; this approach to handling missing data is motivated by the work of [[7]](#7). While we added features that solely indicate missingness for all baseline methods, methods that use topic modeling do not require encoding missingness explicitly. For topic modeling based methods, feature vectors encode number of occurrences, so a patient with missing feature simply has that feature's number of occurrences set to 0. For this reason, we did not explicitly encode missingness as a separate feature for methods that use topic modeling.

#### This gives rise to our three data formats.

Across scripts, you will find that there are three data formats one could output, "cox", "original", and "discretize". 

- **Mode "cox"** refers to one-hot encoding all categorical features, with one reference column removed per categorical feature; each feature that has missing entries will also have a separate feature indicating whether a subject is missing information for this feature; this format should be used for any models that directly apply Cox regression on input data. 

- **Mode "original"** refers to one-hot encoding all categorical features, **without** removing a reference column per feature; each feature that has missing entries will also have a separate feature indicating whether a subject is missing information for this feature; this format should be used for any models that do not directly apply Cox regression, or topic modelling on input data.

- **Mode "discretize"** refers to one-hot encoding all categorical features, **without** removing a reference column per feature; besides, continuous features are discretized by percentiles; also, there are no features that explicitly encode missingness like the two formats above; this format should be used for any models that apply topic modelling on input data. 

Applying different formats will give you datasets with different number of features. To reproduce our results, please check the dataset you use against the dimensions we had below:

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

### Preprocessing Scripts

Each datasets came in in different formats, so each dataset's preprocessing is handled slighly differently by the following scripts:

#### [METABRIC & SUPPORT](preprocessing_metabric_support.py)

#### [MIMIC2 Datasets: Ich, Pancreatitis, Sepsis](preprocessing_mimic2.py)
Feature engineering logic is implemented [here](preprocessing_mimic2_lib/FeatureEngineerNew2.py). 

#### [UNOS](preprocessing_unos.ipynb)
This notebook tries to recover UNOS's preprocessing in some previous papers. Please do not share this file with unauthorized users of the UNOS data.

## Processed Datasets

An experimented-ready dataset is a folder named after the dataset. In the folder, the following should be present:

- *X.npy*: Feature matrix, where each row is a patient, and each column is a feature. Taken in by models. 
- *Y.npy*: Labels, with a column containing survival times, and a column containing censoring indicators. Taken in by models.
- *F.npy*: Feature names. Taken in by models.
- *feature_list.txt*: Feature names, for humans to read.

Each one of the three data formats will have a set of these. Take *X.npy*, for example, expect *X_cox.npy*, *X_discretized.npy*, and *X.npy* to contain feature matrices in the formats of *cox*, *discretized*, and *original*.

## References
<a id="1">[1]</a> 
W. A. Knaus, F. E. Harrell, J. Lynn, L. Goldman, R. S. Phillips, A. F. Connors,N. V. Dawson, W. J. Fulkerson, R. M. Califf, and N. Desbiens. The SUPPORT prognostic model: Objective estimates of survival for seriously ill hospitalized adults. Annals of Internal Medicine, 122(3):191–203, 1995.

<a id="2">[2]</a> 
F. E. Harrell Jr.Regression  Modeling  Strategies:  With  Applications  to  LinearModels,  Logistic  and  Ordinal  Regression,  and  Survival  Analysis. Springer,2015.

<a id="3">[3]</a> 
C. Lee, W. R. Zame, J. Yoon, and M. van der Schaar. DeepHit: A deep learning approach to survival analysis with competing risks.  In AAAI Conference on Artificial Intelligence, 2018

<a id="4">[4]</a> 
J. Yoon, W. R. Zame, A. Banerjee, M. Cadeiras, A. M. Alaa, and M. van derSchaar. Personalized survival predictions via trees of predictors: An application to cardiac transplantation. PloS One, 13(3), 2018.

<a id="5">[5]</a> 
C.  Curtis,  S.  P.  Shah,  S.-F.  Chin,  G.  Turashvili,  O.  M.  Rueda,  M.  J.  Dun-ning, D. Speed, A. G. Lynch, S. Samarajiwa, and Y. Yuan. The genomic and transcriptomic architecture of 2,000 breast tumours reveals novel subgroups. Nature, 486(7403):346, 2012.

<a id="6">[6]</a> 
A. E. Johnson, T. J. Pollard, L. Shen, L.-w. H. Lehman, M. Feng, M. Ghassemi,B.  Moody,  P.  Szolovits,  L.  A.  Celi,  and  R.  G.  Mark.   MIMIC-III,  a  freely accessible critical care database. Scientific Data, 3, 2016.

<a id="7">[7]</a> 
Z. C. Lipton, D. C. Kale, and R. Wetzel. Modeling missing data in clinical timeseries with RNNs. In Machine Learning for Healthcare, 2016
