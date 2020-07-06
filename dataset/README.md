# Survival Datasets

Because some datasets require applying for access or are simply too large, we did not include the actual data in this repo. To be able to run any experiments, after git cloning, make sure to populate this **dataset/** directory with actual data. A dataset should correspond to a folder with the same name as the dataset and containing all files detailed in [this section].

## Table of Contents

* [Dataset Descriptions]
  * [SUPPORT](#support)
  * [UNOS](#unos)
  * [METABRIC](#metabric)
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

The dataset from the Study to Understand Prognoses Preferences Outcomes and Risks of Treatment (SUPPORT) [[1]](#1) is freely available [online](http://biostat.mc.vanderbilt.edu/wiki/Main/SupportDesc). This dataset contains 14 clinical features collected from seriously ill hospitalized adults, such as their age, presence of cancer, and neurologic function. These features were collected from patients on the third day after the study started, and patients were followed for survival until 180 days after the study entry. For our purposes, the dataset was split into four datasets corresponding to different disease groups (acute respiratory failure/multiple organ system failure, cancer, coma, COPD/congestive heart failure/cirrhosis), as done by [[2]](#2). After we created these four subsets, all subjects from the cancer group have identical values for a clinical feature related to cancer presence, so this feature was removed only for the cancer cohort, resulting in 13 clinical features for the **SUPPORT-3** dataset.

### UNOS

The UNOS dataset was extracted from the United Network for Organ Sharing (UNOS) [database](https://www.unos.org/data/), and curated in order to replicate the pre-processing documented by [[3]](#3) and [[4]](#4). We selected only patients who went through heart transplantations in the 30-year window from January 1985 to December 2015. Because [[4]](#4) did not document the exact list of feature names that we could directly extract from the database, we attempted to the best of our ability to curate a list of features that overlaps the most with the feature table presented by them. We ended up with 49 features in total, among which 31 are recipient-related, 12 are donor-related, 6 are compatibility related. For this dataset, our objective is to predict patients' post-transplantation survival time. Because we assumed December 2015 to be the end of data collection, patients who were still alive as of December 2015 are all considered censored samples. Among 62644 patients who underwent transplantation, around 50.2% are censored samples.

### METABRIC

We obtained the Molecular Taxonomy of Breast Cancer International Consortium (METABRIC) dataset from [the Synapse platform](https://www.synapse.org/). The METABRIC dataset contains clinical and genetic features from breast cancer patients, and their respective survival durations. We only used a subset of 24 features that are available for open use through Synapse. This dataset includes 1981 breast cancer patients in total, around 55.2% of whom were censored and not followed until death. The original METABRIC paper [[5]](#5) discusses how the dataset's clinical features were defined in more detail. 

### MIMIC(Ich)

The intracerebral hemorrhage (ICH) dataset we evaluated on is created from MIMIC-III (version 1.4), a critical care health records database containing 52 thousand individuals and their hospital encounters involving admission to the ICU at Beth Israel Deaconess center between 2001 and 2012 [[6]](#6). Experiments were conducted using a subset of the MIMIC-III data consisting of patients having spontaneous intracerebral hemorrhage requiring admission to the ICU. Patients were included in the study if they have an ICU admission with a primary billing code of intracerebral hemorrhage, resulting in a cohort of 1010 individuals. For patients who are admitted to the ICU multiple times, we only consider their first visit to the ICU. This subset of the data has no right-censoring.

Features extracted include demographics, medications, billing codes, procedures, laboratory measurements, events recorded into charts, and vitals. Features were extracted from the relational database into a 4-column format for *patient id*, *time*, *event*, and *event value*. To prevent erroneous merging of different events into a single event, and to provide more informative events, event strings are concatenations of the event descriptor prefixed with the table from which they are derived and additional relevant information such as measurement type, measurement units, etc. Because events recorded in charts are sometimes automated and sometimes manually entered, a physician-developed mapping and lower-casing all fields were used to resolve duplicate entries. As we aim to predict the patient length of stay in ICU, we extract clinical events from the subjects' electronic health records strictly before ICU admission. After preprocessing, the total number of features used for prediction is 1157.

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

<a id="2">[2]</a> 
F. E. Harrell Jr.Regression  Modeling  Strategies:  With  Applications  to  LinearModels,  Logistic  and  Ordinal  Regression,  and  Survival  Analysis. Springer,2015.

<a id="3">[3]</a> 
C. Lee, W. R. Zame, J. Yoon, and M. van der Schaar. DeepHit: A deep learningapproach to survival analysis with competing risks.  InAAAI  Conference  onArtificial Intelligence, 2018

<a id="4">[4]</a> 
J. Yoon, W. R. Zame, A. Banerjee, M. Cadeiras, A. M. Alaa, and M. van derSchaar.  Personalized survival predictions via trees of predictors: An applica-tion to cardiac transplantation.PloS One, 13(3), 2018.

<a id="5">[5]</a> 
C.  Curtis,  S.  P.  Shah,  S.-F.  Chin,  G.  Turashvili,  O.  M.  Rueda,  M.  J.  Dun-ning, D. Speed, A. G. Lynch, S. Samarajiwa, and Y. Yuan.  The genomic andtranscriptomic architecture of 2,000 breast tumours reveals novel subgroups.Nature, 486(7403):346, 2012.

<a id="6">[6]</a> 
A. E. Johnson, T. J. Pollard, L. Shen, L.-w. H. Lehman, M. Feng, M. Ghassemi,B.  Moody,  P.  Szolovits,  L.  A.  Celi,  and  R.  G.  Mark.   MIMIC-III,  a  freelyaccessible critical care database.Scientific Data, 3, 2016.
