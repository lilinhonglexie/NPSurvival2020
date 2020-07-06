'''
    This script processes 3 MIMIC2 datasets and save them as experiment-ready .npy
    files. Data cleaning logic is implemented in preprocessing_mimic2_lib/FeatureEngineerNew2.py.

'''

import os
import datetime
import numpy as np
from preprocessing_mimic2_lib.FeatureEngineerNew2 import FeatureEngineerNew2

def load_whole_data(dataset_idxs, verbose, data_path, discretized, onehot, zerofilling):
    """
    Load complete train and test dataset with arbitrarily selected hyper-params

    :param dataset_idxs: list of any combination of 0, 1, 2
    :param verbose
    :param data_path
    :return train_dfs: dict, dataset_name -> train_dataset [DataFrame]
    :return test_dfs: dict, dataset_name -> test_dataset [DataFrame]
    :return unique_times: dict, dataset_name -> LOS checkpoints, from train
    :return dataset_names
    """
    fe = FeatureEngineerNew2(verbose=verbose, dataPath=data_path)

    # arbitrarily selected params for feature engineering part (based on cross validation by Ren)
    low_freq_event_thds = [0.02, 0.01, 0.003]
    low_freq_value_thds = [0.01, 0.005, 0.001]

    names_ref = ["pancreatitis", "ich", "sepsis"]
    names_ref_dir = ["Pancreatitis", "Ich", "Sepsis"]

    # each dataset index
    for dataset_idx in dataset_idxs:
        dataset_name = names_ref[dataset_idx]
        if verbose:
            print("current dataset:", dataset_name)

        fe.Fit(sourceIndex=dataset_idx,
               filePrefix="",
               rareFrequency=low_freq_event_thds[dataset_idx],
               rareFrequencyCategory=low_freq_value_thds[dataset_idx])

        fe.FitTransform(filePrefix="", discretized=discretized, \
                        onehot=onehot, zerofilling=zerofilling)

        train_df, test_df, feature_list = fe.ExportData()
        print("[TEST] Train dimension: {}, Test dimension: {}, Feature Len: {}".format(\
                                    train_df.shape, test_df.shape, len(feature_list)))

        # save data
        os.makedirs(names_ref_dir[dataset_idx], exist_ok=True)
        if discretized == "discretized_1":
            assert(onehot == "default")
            output_data_fname = names_ref_dir[dataset_idx] + "/" + "{}_discretized.npy"
            save_data(train_df, test_df, output_data_fname)
        elif discretized == "original" and onehot == "default":
            output_data_fname = names_ref_dir[dataset_idx] + "/" + "{}.npy"
            save_data(train_df, test_df, output_data_fname)
        elif discretized == "original" and onehot == "reference":
            output_data_fname = names_ref_dir[dataset_idx] + "/" + "{}_cox.npy"
            save_data(train_df, test_df, output_data_fname)
        else:
            raise NotImplementedError


def save_data(train_df, test_df, scholar_data_name):
    train_X = train_df.drop(columns=['LOS', 'OUT']).values
    test_X = test_df.drop(columns=['LOS', 'OUT']).values

    np.save(scholar_data_name.format("X"), train_X)
    np.save(scholar_data_name.format("Y"), train_df[['LOS', 'OUT']])
    np.savetxt(scholar_data_name.format("F"),\
               train_df.drop(columns=['LOS', 'OUT']).columns, fmt='%s')

    feature_list = train_df.drop(columns=['LOS', 'OUT']).columns

    with open(scholar_data_name.format("feature_list_sorted").replace(".npy", ".txt"), "w") as ft:
        for feature_name in sorted(feature_list):
            ft.write(feature_name)
            ft.write("\n")

    with open(scholar_data_name.format("feature_list").replace(".npy", ".txt"), "w") as ft:
        for feature_name in feature_list:
            ft.write(feature_name)
            ft.write("\n")

    print("{} saved!".format(scholar_data_name.format("")))


if __name__ == '__main__':
    dataset_idxs = [1, 2, 0] # indices into the list of dataset names: ["pancreatitis", "ich", "sepsis"]

    mode = "cox" # change this in to one of ["discretize", "original", "cox"]

    if mode == "discretize":
        load_whole_data(dataset_idxs, verbose=True, \
        data_path='original_data/',\
        discretized="discretized_1", onehot="default", zerofilling=True)
        # discretize, do not remove reference column, impute with zeros, no missing flags

    elif mode == "original":
        load_whole_data(dataset_idxs, verbose=True, \
        data_path='original_data/',\
        discretized="original", onehot="default", zerofilling=True)
        # do not discretize, do not remove reference column, impute with zeros, have missing flags

    elif mode == "cox":
        load_whole_data(dataset_idxs, verbose=True, \
        data_path='original_data/',\
        discretized="original", onehot="reference", zerofilling=True)
        # do not discretize, remove reference columns, impute with zeros, have missing flags

    else:
        raise NotImplementedError
