'''
    Preprocessing script for SUPPORT and METABRIC.
    Saves these two datasets into experiment-ready .npy files.

'''
import os, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold

def process_metabric(mode):
    '''
        mode could be one of {'discretize', 'original', 'cox'}

    '''
    print("Preprocessing METABRIC dataset, mode = {}".format(mode))
    os.makedirs("METABRIC", exist_ok=True)

    if mode == "discretize" or mode == "original":

        feature_vectors_df = pd.read_csv('original_data/METABRIC_full_imputed_X.csv', delimiter=',')
        labels_df = pd.read_csv('original_data/METABRIC_full_imputed_Y.csv', delimiter=',')
        # print(feature_vectors_df.shape)

        if mode == "discretize":
            # group features into continuous and categorical and discretize continuous features
            features = feature_vectors_df.columns
            continuous_features = ['age_at_diagnosis', 'size', 'lymph_nodes_positive', 'NOT_IN_OSLOVAL_lymph_nodes_removed', 'NOT_IN_OSLOVAL_NPI']
            continuous_features_quantiles = np.linspace(0, 1, 6)
            missing_flag_features = [] # drop missingness flags

            n_sample, n_feature = feature_vectors_df.shape
            for feature_i in range(n_feature):
                curr_feature = features[feature_i]
                curr_vals = feature_vectors_df[curr_feature]

                # The following code chunk is how I discretized continuous features in general
                # The limitation would be you need to know which features are continuous and explicitly hard-code them beforehand
                if curr_feature in continuous_features:
                    curr_quantile_edges = list(np.quantile(curr_vals, continuous_features_quantiles))
                    discretized_vals = np.digitize(curr_vals, bins = curr_quantile_edges[:-1])

                    discretized_bin_label = 1
                    for discretized_id in np.unique(discretized_vals):
                        curr_lo = curr_quantile_edges[discretized_id-1]
                        curr_hi = curr_quantile_edges[discretized_id]
                        new_feature_name = curr_feature + "(BIN#{}):{}-{}".format(discretized_bin_label, 
                                                                                  np.round(curr_lo, decimals=2), np.round(curr_hi, decimals=2))
                        new_feature_vals = (discretized_vals == discretized_id).astype(np.int32)
                        feature_vectors_df[new_feature_name] = new_feature_vals
                        discretized_bin_label += 1

                # When mode = discretize, we also drop any features that are intentionally created as missing flags
                # because topic models do not need missing flag features to encode missingness
                # Specifically for the METABRIC dataset, if feature name ends with "_nan", it is a missing flag
                elif curr_feature.endswith("_nan"):
                    missing_flag_features.append(curr_feature)

            feature_vectors_df = feature_vectors_df.drop(columns = continuous_features + missing_flag_features)

        feature_vectors_df = feature_vectors_df.astype("float64")
        labels_df = labels_df.astype("float64")

        if mode == "discretize":
            print("Discretized dataset dimensions: ", feature_vectors_df.shape)
            # These will be loaded for experiments
            np.save("METABRIC/X_discretized.npy", feature_vectors_df.values)
            np.save("METABRIC/Y_discretized.npy", labels_df.values)
            np.savetxt("METABRIC/F_discretized.npy", feature_vectors_df.columns, fmt="%s")
            # This is a human-readable version of the complete list of features
            with open("METABRIC/feature_list_discretized.txt", "w") as f:
                for feature in sorted(feature_vectors_df.columns):
                    f.write(feature)
                    f.write("\n")

        elif mode == "original":
            print("Non-discretized dataset dimensions: ", feature_vectors_df.shape)

            np.save("METABRIC/X.npy", feature_vectors_df.values)
            np.save("METABRIC/Y.npy", labels_df.values)
            np.savetxt("METABRIC/F.npy", feature_vectors_df.columns, fmt="%s")

            with open("METABRIC/feature_list.txt", "w") as f:
                for feature in sorted(feature_vectors_df.columns):
                    f.write(feature)
                    f.write("\n")

    elif mode == "cox":
        # Note that this option requires running process_metabric(mode="original") first

        X = np.load("METABRIC/X.npy")
        Y = np.load("METABRIC/Y.npy")
        F = []
        with open("METABRIC/F.npy", 'r') as feature_names:
            for line in feature_names.readlines():
                F.append(line.strip())
        df = pd.DataFrame(data=X, columns=F)

        categoricals = ['grade_2', 'grade_3', 'grade_nan', \
                        'histological_type_IDC', 'histological_type_IDC+ILC', 'histological_type_IDC-MED', 'histological_type_IDC-MUC', 'histological_type_IDC-TUB', 'histological_type_ILC', 'histological_type_INVASIVE TUMOUR', 'histological_type_MIXED NST AND A SPECIAL TYPE', 'histological_type_OTHER', 'histological_type_OTHER INVASIVE', 'histological_type_PHYL', \
                        'ER_IHC_status_pos', 'ER_IHC_status_nan', \
                        'ER.Expr_-', \
                        'PR.Expr_-', \
                        'HER2_IHC_status_1', 'HER2_IHC_status_2', 'HER2_IHC_status_3', 'HER2_IHC_status_nan', \
                        'HER2_SNP6_state_LOSS', 'HER2_SNP6_state_NEUT', 'HER2_SNP6_state_nan', \
                        'Her2.Expr_-', \
                        'Treatment_CT/HT', 'Treatment_CT/HT/RT', 'Treatment_CT/RT', 'Treatment_HT', 'Treatment_HT/RT', 'Treatment_NONE', 'Treatment_RT', \
                        'NOT_IN_OSLOVAL_menopausal_status_inferred_pre', 'NOT_IN_OSLOVAL_menopausal_status_inferred_nan', \
                        'NOT_IN_OSLOVAL_group_2', 'NOT_IN_OSLOVAL_group_3', 'NOT_IN_OSLOVAL_group_4', 'NOT_IN_OSLOVAL_group_other', \
                        'NOT_IN_OSLOVAL_stage_1', 'NOT_IN_OSLOVAL_stage_2', 'NOT_IN_OSLOVAL_stage_3', 'NOT_IN_OSLOVAL_stage_4', 'NOT_IN_OSLOVAL_stage_nan',\
                        'NOT_IN_OSLOVAL_cellularity_low', 'NOT_IN_OSLOVAL_cellularity_moderate', 'NOT_IN_OSLOVAL_cellularity_nan', \
                        'NOT_IN_OSLOVAL_P53_mutation_status_WT', 'NOT_IN_OSLOVAL_P53_mutation_status_nan', \
                        'NOT_IN_OSLOVAL_P53_mutation_type_MISSENSE', 'NOT_IN_OSLOVAL_P53_mutation_type_MISSENSE:TRUNCATING', 'NOT_IN_OSLOVAL_P53_mutation_type_TRUNCATING', 'NOT_IN_OSLOVAL_P53_mutation_type_nan', \
                        'NOT_IN_OSLOVAL_Pam50Subtype_Her2', 'NOT_IN_OSLOVAL_Pam50Subtype_LumA', 'NOT_IN_OSLOVAL_Pam50Subtype_LumB', 'NOT_IN_OSLOVAL_Pam50Subtype_NC', 'NOT_IN_OSLOVAL_Pam50Subtype_Normal', \
                        'NOT_IN_OSLOVAL_IntClustMemb_2', 'NOT_IN_OSLOVAL_IntClustMemb_3', 'NOT_IN_OSLOVAL_IntClustMemb_4', 'NOT_IN_OSLOVAL_IntClustMemb_5', 'NOT_IN_OSLOVAL_IntClustMemb_6', 'NOT_IN_OSLOVAL_IntClustMemb_7', 'NOT_IN_OSLOVAL_IntClustMemb_8', 'NOT_IN_OSLOVAL_IntClustMemb_9', 'NOT_IN_OSLOVAL_IntClustMemb_10', 'NOT_IN_OSLOVAL_IntClustMemb_nan', \
                        'NOT_IN_OSLOVAL_Site_2', 'NOT_IN_OSLOVAL_Site_3', 'NOT_IN_OSLOVAL_Site_4', 'NOT_IN_OSLOVAL_Site_5', \
                        'NOT_IN_OSLOVAL_Genefu_ER+/HER2- Low Prolif', 'NOT_IN_OSLOVAL_Genefu_ER-/HER2-', 'NOT_IN_OSLOVAL_Genefu_HER2+', 'NOT_IN_OSLOVAL_Genefu_nan']
        continuous = ['age_at_diagnosis', 'size', 'lymph_nodes_positive', 'NOT_IN_OSLOVAL_lymph_nodes_removed', 'NOT_IN_OSLOVAL_NPI']
        df = df[categoricals+continuous]

        print("Cox dataset dimensions: ", df.shape)

        np.save("METABRIC/X_cox.npy", df.values)
        np.save("METABRIC/Y_cox.npy", Y)
        np.savetxt("METABRIC/F_cox.npy", df.columns, fmt="%s")

        with open("METABRIC/feature_list_cox.txt", "w") as f:
            for feature in sorted(df.columns):
                f.write(feature)
                f.write("\n")

    else:
        raise NotImplementedError

def process_support(mode):
    '''
        mode could be one of {'discretize', 'original', 'cox'}

    '''
    support_dzclasses = ["ARF_MOSF", "COPD_CHF_Cirrhosis", "Cancer", "Coma"]
    for dzclass in support_dzclasses:
        print("Preprocessing SUPPORT_{} dataset, mode = {}".format(dzclass, mode))
        os.makedirs("SUPPORT_{}".format(dzclass), exist_ok=True)

        data_df = pd.read_csv("original_data/support2_{}_full.csv".format(dzclass), delimiter=',')
        
        #print(data_df[pd.isna(data_df['race'])]) # print rows with missing ethnicity information
        #print(data_df.isna().sum()) # print feature summary
        print("Dimensions of incoming {} dataset:".format(dzclass), data_df.shape)
        data_df.dropna(inplace=True) 
        assert(np.sum(data_df.isna().values) == 0)

        feature_vectors_df = data_df.drop(columns=["d.time", "death"])
        labels_df = data_df[["d.time", "death"]]

        cat_features = []
        universal_features = []
        cat_features_prefixes = dict()
        for cat_feature in ["sex", "race", "ca"]:
            if len(np.unique(feature_vectors_df.loc[:, cat_feature])) == 1: 
            # this handles a special case: feature "ca" takes the same value for all subjects for the support_cancer dataset
                universal_features.append(cat_feature)
            else:
                cat_features.append(cat_feature)
                cat_features_prefixes[cat_feature] = cat_feature

        feature_vectors_df = feature_vectors_df.drop(columns=universal_features)

        # one hot encode categorical
        # The difference between mode "original" and mode "cox" is that:
        # Mode cox drops one reference column for each categorical variable during one-hot encoding
        # This is a special requirement for cox models, which are linear about the features
        if mode == "original":
            feature_vectors_df = pd.get_dummies(feature_vectors_df, columns=cat_features,
                                                prefix=cat_features_prefixes,
                                                drop_first=False)
        elif mode == "cox":
            feature_vectors_df = pd.get_dummies(feature_vectors_df, columns=cat_features,
                                                prefix=cat_features_prefixes,
                                                drop_first=True)

        elif mode == "discretize":
            # we also do not remove reference columns for mode "discretize"
            feature_vectors_df = pd.get_dummies(feature_vectors_df, columns=cat_features,
                                                prefix=cat_features_prefixes,
                                                drop_first=False)

            n_sample, n_feature = feature_vectors_df.shape

            continuous_features = ["age", "meanbp", "hrt", "resp", "temp", "wblc", "sod", "crea", "num.co"]
            continuous_features_quantiles = np.linspace(0, 1, 6)

            for curr_feature in continuous_features:
                curr_vals = feature_vectors_df[curr_feature]
                curr_quantile_edges = list(np.quantile(curr_vals[curr_vals.notnull()], continuous_features_quantiles))
                discretized_vals = np.digitize(curr_vals, bins = curr_quantile_edges[:-1])

                discretized_bin_label = 1
                for discretized_id in np.unique(discretized_vals):
                    curr_lo = curr_quantile_edges[discretized_id-1]
                    curr_hi = curr_quantile_edges[discretized_id]
                    new_feature_name = curr_feature + "(BIN#{}):{}-{}".format(discretized_bin_label, 
                                                                              np.round(curr_lo, decimals=2), np.round(curr_hi, decimals=2))

                    new_feature_vals = (discretized_vals == discretized_id).astype(np.int32)
                    feature_vectors_df[new_feature_name] = new_feature_vals
                    discretized_bin_label += 1

            feature_vectors_df = feature_vectors_df.drop(columns = continuous_features)
            # Note here we don't have to remove the missingness flags, because we actually removed all samples with missing info for the SUPPORT dataset
            # We decided to do this because SUPPORT datasets have very few samples with missing values

        feature_vectors_df = feature_vectors_df.astype("float64")
        labels_df = labels_df.astype("float64")
        assert(np.sum(feature_vectors_df.isna().values) == 0)
        assert(np.sum(labels_df.isna().values) == 0)

        # there is no full or empty column
        try:
            assert(np.sum(np.sum(feature_vectors_df.values, axis=0)/feature_vectors_df.shape[0] == 1) == 0)
            assert(np.sum(np.sum(feature_vectors_df.values, axis=0)/feature_vectors_df.shape[0] == 0) == 0)
        except:
            full_col_ids = np.where(np.sum(feature_vectors_df.values, axis=0)/feature_vectors_df.shape[0] == 1)
            print("this column is always one for all samples:", feature_vectors_df.columns[full_col_ids])
            assert(False)

        if mode == "discretize":
            print("Discretized dataset dimensions: ", feature_vectors_df.shape)
            # These will be loaded for experiments
            np.save("SUPPORT_{}/X_discretized.npy".format(dzclass), feature_vectors_df.values)
            np.save("SUPPORT_{}/Y_discretized.npy".format(dzclass), labels_df.values)
            np.savetxt("SUPPORT_{}/F_discretized.npy".format(dzclass), feature_vectors_df.columns, fmt="%s")
            # This is a human-readable version of the complete list of features
            with open("SUPPORT_{}/feature_list_discretized.txt".format(dzclass), "w") as f:
                for feature in sorted(feature_vectors_df.columns):
                    f.write(feature)
                    f.write("\n")

        elif mode == "original":
            print("Non-discretized dataset dimensions: ", feature_vectors_df.shape)

            np.save("SUPPORT_{}/X.npy".format(dzclass), feature_vectors_df.values)
            np.save("SUPPORT_{}/Y.npy".format(dzclass), labels_df.values)
            np.savetxt("SUPPORT_{}/F.npy".format(dzclass), feature_vectors_df.columns, fmt="%s")

            with open("SUPPORT_{}/feature_list.txt".format(dzclass), "w") as f:
                for feature in sorted(feature_vectors_df.columns):
                    f.write(feature)
                    f.write("\n")

        elif mode == "cox":
            print("Cox dataset dimensions: ", feature_vectors_df.shape)

            np.save("SUPPORT_{}/X_cox.npy".format(dzclass), feature_vectors_df.values)
            np.save("SUPPORT_{}/Y_cox.npy".format(dzclass), labels_df.values)
            np.savetxt("SUPPORT_{}/F_cox.npy".format(dzclass), feature_vectors_df.columns, fmt="%s")

            with open("SUPPORT_{}/feature_list_cox.txt".format(dzclass), "w") as f:
                for feature in sorted(feature_vectors_df.columns):
                    f.write(feature)
                    f.write("\n")

        else:
            raise NotImplementedError

if __name__ == '__main__':
    process_metabric(mode="original")
    process_metabric(mode="discretize")
    process_metabric(mode="cox")

    process_support(mode="original")
    process_support(mode="discretize")
    process_support(mode="cox")


