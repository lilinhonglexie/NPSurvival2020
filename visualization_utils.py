
import os, pickle, json, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sb

import warnings
warnings.filterwarnings('ignore')

class MidpointNormalize(colors.Normalize):
    # https://matplotlib.org/3.1.0/gallery/userdemo/colormap_normalizations_custom.html
    # For adjusting mid values of the heatmaps
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

def get_shorter_name(vocab, dataset):
    '''
        Get better feature names for plotting the heatmaps.

    '''
    new_vocab = []
    if dataset == "METABRIC":
        for v in vocab:
            if "NOT_IN_OSLOVAL_" in v:
                new_v = v.replace("NOT_IN_OSLOVAL_", "")  
            else:
                new_v = v
            
            new_vocab.append(new_v)
    
    elif dataset.startswith("SUPPORT"):
        vmap = {"num.co": "num.comorbidities", "ca_metastatic": "cancer_metastatic", "ca_no": "cancer_no", "ca_yes": "cancer_yes",
                "wblc": "wbc_count", "crea": "serum_creatinine", "sod":"serum_sodium", "hrt": "heart_rate", "resp": "respiration_rate",
                "temp": "temperature_celcius", "meanbp": "mean_blood_pressure"}
        for v in vocab:
            for vpre in vmap.keys():
                if v.startswith(vpre):
                    new_v = v.replace(vpre, vmap[vpre])
                    break
                else:
                    new_v = v
            new_vocab.append(new_v)
    
    elif dataset == "UNOS":
        vmap = {"INIT_AGE": "AGE","AGE_DON":"AGE_DONER","PRAMR": "MOST_RECENT_PRA", 
                "CREAT_TRR": "CREATININE", "HGT_CM_CALC": "HEIGHT_CM", "HGT_CM_DON_CALC": "HEIGHT_CM_DONER",
                "PREV_TX": "PREVIOUS_TRANSPLANT", "BMI_DON_CALC": "BMI_DONER", 
                "DIAL_PRIOR_TX": "DIALYSIS_HISTORY", "PRAPK": "PEAK_PRA", "LV_EJECT": "LV_EJECT_FRACTION", 
                "INFECT_IV_DRUG_TRR": "INFECTION_REQUIRING_IV_DRUG", "AMIS": "A_LOCUS_MISMATCH_LEVEL", 
                "WGT_KG_CALC": "WEIGHT_KG", "WGT_KG_DON_CALC": "WEIGHT_KG_DONER",
                "TBILI": "DONOR_TERMINAL_TOTAL_BILIRUBIN", "BMIS": "B_LOCUS_MISMATCH_LEVEL", 
                "DIAB": "DIABETES", "BMI_CALC": "BMI", "ABO": "BLOOD_GROUP", "DAYS_STAT1A": "DAYS_IN_STATUS_1A",
                "DAYS_STAT1" : "DAYS_IN_STATUS_1", "IABP_TRR": "IABP", "ECMO_TRR":"ECMO_TRR",
                "HIST_DIABETES_DON": "DONER_DIABETES_HISTORY", "DAYS_STAT2":"DAYS_IN_STATUS_2",
                "HEP_C_ANTIBODY_DON": "HEP_C_ANTIBODY_DONER", "DRMIS": "DR_LOCUS_MISMATCH_LEVEL",
                "VAD_TAH_TRR": "VAD_TAH", "DAYS_STAT1B": "DAYS_IN_STATUS_1B", "ABO_DON": "BLOOD_GROUP_DONER",
                "CREAT_DON": "CREATININE_DONER", "HLAMIS": "HLA_MISMATCH_LEVEL", 
                "CLIN_INFECT_DON": "DONER_CLINICAL_INFECTION", "ISCHTIME": "ISCHEMIC_TIME_HOURS",
                "ABO_MAT": "ABO_MATCH_LEVEL", "VENTILATOR_TRR": "VENTILATOR", "GENDER_DON": "GENDER_DONER"}
        for v in vocab:
            for vpre in vmap.keys():
                if v.startswith(vpre):
                    new_v = v.replace(vpre, vmap[vpre])
                    break
                else:
                    new_v = v
            new_v = new_v.replace("_TRR", "")
            new_v = new_v.replace("_DON_", "_DONER_")
            new_v = new_v.replace("_DON(", "_DONER(")
            new_v = new_v.replace("_MAT_", "_MATCH_LEVEL_")
            new_v = new_v.replace("_MAT(", "_MATCH_LEVEL(")

            new_vocab.append(new_v)
    
    elif dataset == "Ich":
        return vocab

    return np.array(new_vocab)

def heatmap_plot_topic_reordered(model, dataset, topic_distributions, beta, vocabulary, 
                                 sort_by_beta=True, sort_by_feature_name=True, logscale=False, clip_negative=False,
                                 saveto="", show_plot=False):  
    '''
    All topic heatmaps, for non-MIMIC2 datasets

    topic_distributions: num_topics * num_vocab
    
    '''
    if sort_by_beta:
        topic_order = np.argsort(-beta)
        topic_distributions = topic_distributions[topic_order]
        beta = beta[topic_order]
    
    topic_distributions = topic_distributions.transpose()
    
    if sort_by_feature_name:
                
        con_vocab_ids = []
        cat_vocab_ids = []
        for v_i, v in enumerate(vocabulary):            
            if "BIN#" in v:
                con_vocab_ids.append(v_i)
            else:
                cat_vocab_ids.append(v_i)

        cat_sorted_args = np.argsort(vocabulary[cat_vocab_ids])
        con_sorted_args = np.argsort(vocabulary[con_vocab_ids])
                
        cat_topic_distributions = topic_distributions[cat_vocab_ids][cat_sorted_args]
        con_topic_distributions = topic_distributions[con_vocab_ids][con_sorted_args]
        topic_distributions = np.vstack((cat_topic_distributions, con_topic_distributions))
        
        vocabulary = np.concatenate((vocabulary[cat_vocab_ids][cat_sorted_args], vocabulary[con_vocab_ids][con_sorted_args]))

    if model == "survscholar_linear" and logscale:
        topic_distributions = np.exp(topic_distributions)
    
    if model == "survscholar_linear" and clip_negative:
        neg_threshold = 1 if logscale else 0
        topic_distributions[topic_distributions <= neg_threshold] = neg_threshold
        
    diff_dict = dict()
    for v_i, v in enumerate(vocabulary):
        if "BIN#" in v:
            base_feature = re.search(r'(\w+)\(BIN#\d\):', v).group(1)
        elif "_" in v:
            base_feature = "_".join(v.split("_")[:-1])
        else:
            base_feature = v
        
        curr_diff = np.ptp(topic_distributions[v_i])
        
        if base_feature in diff_dict:
            diff_dict[base_feature] = max(diff_dict[base_feature], curr_diff)
        else:
            diff_dict[base_feature] = curr_diff
    
    diff_ls = sorted(list(diff_dict.items()), key=lambda pair: -1 * pair[1])
    
    new_order = []
    for curr_base, _ in diff_ls:
        for v_i, v in enumerate(vocabulary):
            if "BIN#" in v:
                base_feature = re.search(r'(\w+)\(BIN#\d\):', v).group(1)
            elif "_" in v:
                base_feature = "_".join(v.split("_")[:-1])
            else:
                base_feature = v
            
            if curr_base == base_feature:
                new_order.append(v_i)
    new_order = np.array(new_order)
    vocabulary = vocabulary[new_order]
    topic_distributions = topic_distributions[new_order]
    
    vocabulary = get_shorter_name(vocabulary, dataset)
    
    with plt.style.context('seaborn-dark'):
        fig = plt.figure(figsize=(len(beta)*0.7,len(vocabulary)/3*0.7))
        plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
        plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
        sb.heatmap(topic_distributions, cmap="RdBu_r", norm=MidpointNormalize(midpoint=1,
                                                                              vmin=-np.max(np.abs(topic_distributions)), 
                                                                              vmax=np.max(np.abs(topic_distributions))))
        plt.yticks(np.arange(len(vocabulary)) + 0.5 , vocabulary, rotation='horizontal',
                   fontsize=12)
        plt.xticks(np.arange(len(beta)) + 0.5, np.round(beta, 2), fontsize=12)
        plt.savefig(saveto+"_heatmap.pdf", bbox_inches = 'tight')
        if not show_plot:
            plt.close()

def postprocess_ich(vocab):
    result = []
    for v in vocab:
        if "merged_others" in v:
            start_i = v.index("merged_others")
            end_i = start_i + len("merged_others")
            result.append(v[:end_i])
        else:
            result.append(v)
    return np.array(result)

def heatmap_plot_topic_reordered_ich(model, dataset, topic_distributions, beta, vocabulary, 
                                     sort_by_beta=True, sort_by_feature_name=True, logscale=False, clip_negative=False,
                                     saveto="", show_plot=False):  
    '''
    topic_distributions: num_topics * num_vocab
    
    '''
    if sort_by_beta:
        topic_order = np.argsort(-beta)
        topic_distributions = topic_distributions[topic_order]
        beta = beta[topic_order]
    
    topic_distributions = topic_distributions.transpose()
    
    if sort_by_feature_name:
                
        con_vocab_ids = []
        cat_vocab_ids = []
        for v_i, v in enumerate(vocabulary):            
            if "BIN#" in v:
                con_vocab_ids.append(v_i)
            else:
                cat_vocab_ids.append(v_i)

        cat_sorted_args = np.argsort(vocabulary[cat_vocab_ids])
        con_sorted_args = np.argsort(vocabulary[con_vocab_ids])
                
        cat_topic_distributions = topic_distributions[cat_vocab_ids][cat_sorted_args]
        con_topic_distributions = topic_distributions[con_vocab_ids][con_sorted_args]
        topic_distributions = np.vstack((cat_topic_distributions, con_topic_distributions))
        
        vocabulary = np.concatenate((vocabulary[cat_vocab_ids][cat_sorted_args], vocabulary[con_vocab_ids][con_sorted_args]))

    if model == "survscholar_linear" and logscale:
        topic_distributions = np.exp(topic_distributions)
    
    if model == "survscholar_linear" and clip_negative:
        neg_threshold = 1 if logscale else 0
        topic_distributions[topic_distributions <= neg_threshold] = neg_threshold
        
    diff_dict = dict()
    for v_i, v in enumerate(vocabulary):
        if "BIN#" in v:
            base_feature = re.search(r'(.+):::\(BIN#\d\):', v).group(1)
        elif ":::" in v:
            base_feature = ":::".join(v.split(":::")[:-1])
        else:
            raise NotImplementedError
        
        curr_diff = np.ptp(topic_distributions[v_i])
        
        if base_feature in diff_dict:
            diff_dict[base_feature] = max(diff_dict[base_feature], curr_diff)
        else:
            diff_dict[base_feature] = curr_diff
    
    diff_ls = sorted(list(diff_dict.items()), key=lambda pair: -1 * pair[1])
    
    new_order = []
    for curr_base, _ in diff_ls:
        for v_i, v in enumerate(vocabulary):
            if "BIN#" in v:
                base_feature = re.search(r'(.+):::\(BIN#\d\):', v).group(1)
            elif ":::" in v:
                base_feature = ":::".join(v.split(":::")[:-1])
            else:
                raise NotImplementedError
            
            if curr_base == base_feature:
                new_order.append(v_i)
                
    new_order = np.array(new_order)
    vocabulary = vocabulary[new_order]
    topic_distributions = topic_distributions[new_order]
    
    nrows = len(vocabulary)
    batch_size = 50
    nbatches = nrows//batch_size + (nrows%batch_size != 0)
    
    for batch_i in range(nbatches):
        batch_topic_distributions = topic_distributions[batch_size*batch_i:batch_size*(batch_i+1), :]
        bathc_vocabulary = vocabulary[batch_size*batch_i:batch_size*(batch_i+1)]

        with plt.style.context('seaborn-dark'):
            
            fig = plt.figure(figsize=(len(beta)*0.7,len(bathc_vocabulary)/3*0.7))
            plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
            plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
            
            sb.heatmap(batch_topic_distributions, cmap="RdBu_r", vmin=np.min(topic_distributions), vmax=np.max(topic_distributions), center=1.0) 

            plt.yticks(np.arange(len(bathc_vocabulary)) + 0.5 ,postprocess_ich(bathc_vocabulary), rotation='horizontal',
                       fontsize=12)
            plt.xticks(np.arange(len(beta)) + 0.5, np.round(beta, 2), fontsize=12)
    
            plt.savefig(saveto+"_heatmap_batch{}.pdf".format(batch_i), bbox_inches = 'tight') 
            plt.close()

def heatmap_plot_single_topic(model, dataset, topic_distributions, beta, vocabulary, saveto="", topwords=100):
    
    topic_distributions = np.exp(topic_distributions)
    vocabulary = get_shorter_name(vocabulary, dataset)

    topic_order = np.argsort(-beta)
    topic_distributions = topic_distributions[topic_order]
    beta = beta[topic_order]
    
    # get vmin/vmax
    vmin = None
    vmax = None
    if dataset == "Ich":
        for topic_i, curr_beta in enumerate(beta):
            curr_topic_distributions = topic_distributions[topic_i]
            curr_order = np.argsort(-curr_topic_distributions)
            curr_topic_distributions = curr_topic_distributions[curr_order]

            curr_topic_distributions = curr_topic_distributions[:topwords]
            
            if vmin is None or np.min(curr_topic_distributions) < vmin:
                vmin = np.min(curr_topic_distributions)
            
            if vmax is None or np.max(curr_topic_distributions) > vmax:
                vmax = np.max(curr_topic_distributions)

    else:
        vmin=np.min(topic_distributions)
        vmax=np.max(topic_distributions)

    for topic_i, curr_beta in enumerate(beta):

        curr_topic_distributions = topic_distributions[topic_i]
        curr_order = np.argsort(-curr_topic_distributions)
        curr_feature_labels = vocabulary[curr_order]
        curr_topic_distributions = curr_topic_distributions[curr_order]
        
        if dataset == "Ich":
            curr_feature_labels = curr_feature_labels[:topwords]
            curr_topic_distributions = curr_topic_distributions[:topwords]
            plotheight = topwords
        else:
            plotheight = len(vocabulary)
        
        midmap = MidpointNormalize(midpoint=1,vmin=np.min(topic_distributions), vmax=np.max(topic_distributions))
        
        with plt.style.context('seaborn-dark'):

            fig = plt.figure(figsize=(0.7,plotheight/3*0.7)) # same as prev settings
            plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
            plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
            sb.heatmap(curr_topic_distributions.reshape(-1, 1), cmap="Reds", vmin=vmin, vmax=vmax)
            plt.yticks(np.arange(len(curr_feature_labels)) + 0.5 , curr_feature_labels, rotation='horizontal',
                       fontsize=12)
            plt.xticks([0.5], [np.round(curr_beta, 2)], fontsize=12)
            
            plt.savefig(saveto+"_single_heatmap_{}.pdf".format(topic_i), bbox_inches = 'tight')
            plt.close()

def print_top_words(probs, betas, features, transcript_path, n_top=50):

    with open(transcript_path, "w") as transcript:

        topic_order = np.argsort(-betas)

        for topic_i in range(len(betas)):
            # print("Topic #{} (Beta={})".format(topic_i, betas[topic_order[topic_i]]))
            print("Topic #{} (Beta={})".format(topic_i, betas[topic_order[topic_i]]), file=transcript)

            curr_probs = probs[topic_order[topic_i]]
            curr_order = np.argsort(-curr_probs)
            for word_i, word_id in enumerate(curr_order[:100]):
                # print(features[word_id], curr_probs[word_id], np.exp(curr_probs[word_id]))
                print(features[word_id], curr_probs[word_id], np.exp(curr_probs[word_id]), file=transcript)

            # print("---\n")
            print("---\n", file=transcript)



