import fnmatch
import os
import random
import itertools
import pickle
import pacmap

from sklearn.metrics import confusion_matrix
from scipy import io
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import scipy as scipy
from collections import Counter
from sklearn import metrics
from textwrap import wrap

import matplotlib.pylab as pylab
base = 20
args = {
    # 'legend.fontsize': base - 4,
          'figure.figsize': (8, 6),
          # 'axes.labelsize': base-4,
          # 'axes.titlesize': base,
          # 'xtick.labelsize': base-10,
          # 'ytick.labelsize': base-10
    }
pylab.rcParams.update(args)



def find_files(directory, pattern='*.csv'):
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))

    return files


plot_name = "100_single_ep_corr_classification_rate_with_certain"
if plot_name == "100_single_ep_corr_classification_rate_with_certain":
    print("Plot_name: ", plot_name)
    """
    Get the correct classification rate with 100 runs of single-epoch-training
    """
    import ipdb
    from scipy.stats import spearmanr
    
    data_dirs = [
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/single-epoch-get-correct-classification-rate/2020-10-05T13-54-31-MLP-nonex0-factor-0-from-ep-0-from-lout40-data7-theta-None-s129-100rns-train/certains",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/single-epoch-get-correct-classification-rate/2020-10-05T13-54-30-MLP-nonex0-factor-0-from-ep-0-from-lout40-data5-theta-None-s129-100rns-train/certains",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/single-epoch-get-correct-classification-rate/2020-10-05T13-54-29-MLP-nonex0-factor-0-from-ep-0-from-lout40-data3-theta-None-s129-100rns-train",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/single-epoch-get-correct-classification-rate/2020-10-05T13-54-28-MLP-nonex0-factor-0-from-ep-0-from-lout40-data1-theta-None-s129-100rns-train",
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/single-epoch-get-correct-classification-rate/2020-10-05T13-54-32-MLP-nonex0-factor-0-from-ep-0-from-lout40-data9-theta-None-s129-100rns-train",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/single-epoch-get-correct-classification-rate/2020-10-09T21-42-58-MLP-nonex0-factor-0-from-ep-0-from-lout40-MNIST-theta-None-s129-100rns-train/certains"]
    num_smp_dataset = {"data0": 8357, "data1": 8326, "data2": 8566, "data3": 8454, "data4": 8440, "data5": 8231,
                       "data6": 8371, "data7": 8357, "data8": 8384, "data9": 7701, "mnist": 70000}
    
    for data_dir in data_dirs:
        files = find_files(data_dir, pattern="one_ep_data_train*.csv")
        
        # get correct count in 100 rauns
        data_source = os.path.basename(files[0]).split("_")[8]
        for theta in [90]:  # , 92.5, 95, 97.5, 99
            print(data_source, "theta:", theta)
            for ind, fn in enumerate(files[:10]):
                values = pd.read_csv(fn, header=0).values
                smp_ids = values[:, 0].astype(np.int)
                true_lables = values[:, 1].astype(np.int)
                noisy_lbs = values[:, 2]
                prob = values[:, 3:]
                if ind == 0:  # the first file to get the total number (3-class) of samples
                    total_num = num_smp_dataset[data_source]  # 3-class samples id
                    correct_ids_w_count = []
                    certain_w_count = []
                    certain_w_corr_count = []
                    correct_dict_id_w_count = {key: 0 for key in np.arange(total_num)}  # total number 9243
                    dict_count_certain = {key: 0 for key in np.arange(total_num)}
                    dict_corr_count_certain = {key: 0 for key in np.arange(total_num)}
                
                pred_lbs = np.argmax(prob, axis=1)
                right_inds = np.where(pred_lbs == noisy_lbs)[0]
                correct_sample_ids = np.unique(smp_ids[right_inds])
                correct_ids_w_count += list(correct_sample_ids)
                
                # Get certain with differnt threshold
                if theta > 1:  # percentile
                    larger_prob = [pp.max() for pp in prob]
                    threshold = np.percentile(larger_prob, theta)
                    slc_ratio = 1 - theta / 100.
                else:  # absolute prob. threshold
                    threshold = theta
                    slc_ratio = 1 - theta
                ct_smp_ids = np.where([prob[i] > threshold for i in range(len(prob))])[0]
                ct_corr_inds = ct_smp_ids[np.where(noisy_lbs[ct_smp_ids] == pred_lbs[ct_smp_ids])[0]]
                certain_w_count += list(np.unique(smp_ids[ct_smp_ids]))
                certain_w_corr_count += list(np.unique(smp_ids[ct_corr_inds]))
                num_certain = len(ct_smp_ids)
            
            correct_id_count_all = Counter(correct_ids_w_count)
            correct_dict_id_w_count.update(correct_id_count_all)
            
            dict_count_certain.update(Counter(certain_w_count))
            dict_corr_count_certain.update(Counter(certain_w_corr_count))
            
            # if theta == 0.975:
            #     ipdb.set_trace()
            counter_array = np.array([[key, val] for (key, val) in correct_dict_id_w_count.items()])
            sort_inds = np.argsort(counter_array[:, 1])
            sample_ids_key = counter_array[sort_inds, 0]
            # rates = counter_array[sort_inds, 1]/counter_array[:, 1].max()
            rates = counter_array[sort_inds, 1] / len(files)
            
            ct_counter_array = np.array([[key, val] for (key, val) in dict_count_certain.items()])
            ct_counter_array_corr = np.array([[key, val] for (key, val) in dict_corr_count_certain.items()])
            
            ct_sele_rates = ct_counter_array[sort_inds, 1] / len(files)
            ct_corr_rates = ct_counter_array_corr[sort_inds, 1] / len(files)
            # rates_certain_corr = counter_array_certain_corr[sort_inds, 1] / counter_array_certain_corr[:, 1].max()
            # sort_samp_ids_certain = ct_counter_array[sort_inds, 0]
            
            fig, ax1 = plt.subplots()
            ax1.set_xlabel("sample index (sorted)")
            ax1.set_ylabel("rate over 100 runs")
            ax1.plot(rates, label="correct clf. rate")
            ax1.tick_params(axis='y')
            ax1.set_ylim([0, 1.0])
            ax1.plot(np.arange(len(ct_sele_rates)), ct_sele_rates, label="distilled selection rate")
            ax1.plot(np.arange(len(ct_corr_rates)), ct_corr_rates, label="distilled corr. rate", color='m')
            plt.legend()
            plt.title("\n".join(wrap("distillation effect-(theta-{})-{}.png".format(theta, data_source), 60)))
            plt.savefig(
                data_dir + "/certain_correct_rate_with_certain-classfication-rate-in-100-runs-({}-{})-{}-({}_theta-{}).png".format(
                    os.path.basename(files[0]).split("_")[7], total_num, data_source, num_certain, theta)),
            plt.savefig(
                data_dir + "/certain_correct_rate_with_certain-classfication-rate-in-100-runs-({}-{})-{}-({}_theta-{}).pdf".format(
                    os.path.basename(files[0]).split("_")[7], total_num, data_source, num_certain, theta), format="pdf")
            print("ok")
            plt.close()
            
            num2select = np.int(np.int(os.path.basename(files[0]).split("_")[7]) * slc_ratio)
            ct_concat_data = np.concatenate((np.array(sort_inds).reshape(-1, 1)[-num2select:],
                                             rates.reshape(-1, 1)[-num2select:], ct_sele_rates.reshape(-1, 1)[-num2select:],
                                             ct_corr_rates.reshape(-1, 1)[-num2select:]), axis=1)
            np.savetxt(data_dir + "/certain_{}_({}-{})-({}_theta-{}).csv".format(data_source,
                                                                                 os.path.basename(files[0]).split("_")[7],
                                                                                 total_num, num2select, theta),
                       ct_concat_data, fmt="%.5f", delimiter=",",
                       header="ori_sort_rate_id,ori_sort_rate,certain_sele_rate,certain_corr_rate")
    concat_data = np.concatenate((np.array(sort_inds).reshape(-1, 1), rates.reshape(-1, 1), ct_sele_rates.reshape(-1, 1),
                                  ct_corr_rates.reshape(-1, 1)), axis=1)
    np.savetxt(data_dir + "/full_summary-{}_100_runs_sort_inds_rate_({}-{}).csv".format(data_source,
                                                                                        os.path.basename(files[0]).split(
                                                                                            "_")[7], total_num),
               concat_data, fmt="%.5f", delimiter=",",
               header="ori_sort_rate_id,ori_sort_rate,certain_sele_rate,certain_corr_rate")


elif plot_name == "100_single_ep_corr_classification_rate_mnist":
    """
    Get the correct classification rate with 100 runs of single-epoch-training
    """
    import ipdb
    from scipy.stats import spearmanr
    
    print("Plot_name: ", plot_name)
    original_data_dirs = [
        r"C:\Users\LDY\Desktop\metabolites-0301\metabolites_tumour_classifier\data\noisy_mnist\0.2_noisy_train_val_mnist_[samp_id,true,noise]-s5058.csv"]
    data_dirs = [
        r"C:\Users\LDY\Desktop\metabolites-0301\metabolites_tumour_classifier\results\old-mnist-single-ep-training-MLP\2021-03-02T21-17-27--MLP-from-mnist-ctFalse-theta-1-s3174-100rns-train-lbInd-2"
        # "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/9-train-with-MNIST-MLP/2021-01-27T13-14-34--MLP-both_meanx3-factor-0.5-from-mnist-certainFalse-theta-1-s5506-0.5-noise-100rns-train-with"
        # r"C:\Users\LDY\Desktop\EPG\PPS-EEG-anomaly"
    ]
    num_smp_dataset = {"data0": 8357, "data1": 8326, "data2": 8566, "data3": 8454, "data4": 8440, "data5": 8231,
                       "data6": 8371, "data7": 8357, "data8": 8384, "data9": 7701, "mnist": 60000}
    
    for data_dir, original in zip(data_dirs, original_data_dirs):
        files = find_files(data_dir, pattern="one*.csv")
        
        spearmanr_rec = []
        # get correct count in 100 rauns
        data_source = os.path.basename(files[0]).split("_")[-4]
        compare2_lb_ind = 2
        for ind in tqdm(range(len(files))):
            # fn = find_files(data_dir, pattern="one_ep_data_train_epoch_{}*.csv".format(ind))
            values = pd.read_csv(files[ind], header=0).values
            smp_ids = values[:, 0].astype(np.int)
            true_lables = values[:, 1].astype(np.int)  # true labels
            noisy_lbs = values[:, 2]  # noisy labels
            prob = values[:, 3:]
            if ind == 0:  # the first file to get the total number (3-class) of samples
                total_num = num_smp_dataset[data_source]  # 3-class samples id
                correct_ids_w_count = []
                noisy_lb_ids_w_count = []
                correct_dict_id_w_count = {key: 0 for key in np.arange(total_num)}  # total number 9243
                noisy_lb_rec = {key: 0 for key in np.arange(total_num)}  # total number 9243
                pre_rank = np.arange(total_num)  # indices
            
            pred_lbs = np.argmax(prob, axis=1)
            right_inds = np.where(pred_lbs == values[:, compare2_lb_ind])[0]
            correct_sample_ids = np.unique(smp_ids[right_inds])
            correct_ids_w_count += list(correct_sample_ids)
            
            noisy_lb_inds = np.where(true_lables != noisy_lbs)[0]
            noisy_lb_sample_ids = np.unique(smp_ids[noisy_lb_inds])  # sample ids that with noisy labels
            noisy_lb_ids_w_count += list(noisy_lb_sample_ids)  # sample ids that with noisy labels
        
        # if ind % 10 == 0:  #     count_all = Counter(ids_w_count)  #     dict_count.update(count_all)  #     curr_count_array = np.array([[key, val] for (key, val) in dict_count.items()])  #     curr_rank = curr_count_array[np.argsort(curr_count_array[:, 1]),0]  #     # spearmanr_rec.append([ind, np.sum(curr_rank==pre_rank)])  #     spearmanr_rec.append([ind, spearmanr(pre_rank, curr_rank)[0]])  #     pre_rank = curr_rank.copy()
        
        correct_id_count_all = Counter(correct_ids_w_count)
        correct_dict_id_w_count.update(correct_id_count_all)
        noisy_lb_rec.update(Counter(noisy_lb_ids_w_count))
        
        counter_array = np.array([[key, val] for (key, val) in correct_dict_id_w_count.items()])
        noisy_lb_rec_array = np.array([[key, val] for (key, val) in noisy_lb_rec.items()])
        noisy_inds_array = np.array([[key, val * 1.0 / len(files)] for (key, val) in noisy_lb_rec.items()])
        sort_inds = np.argsort(counter_array[:, 1])
        sample_ids_key = counter_array[sort_inds, 0]
        # rates = counter_array[sort_inds, 1]/counter_array[:, 1].max()
        rates = counter_array[sort_inds, 1] / len(files)
        noisy_lb_rate = noisy_inds_array[sort_inds, 1]
        
        assert np.sum(counter_array[sort_inds, 0] == noisy_inds_array[sort_inds, 0]), "sorted sample indices mismatch"
        
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("sample sorted by the correct clf. rate"),
        ax1.set_ylabel("correct clf. rate (over 100 runs)"),
        ax1.plot(rates, label="original data set"),
        ax1.tick_params(axis='y'),
        ax1.set_ylim([0, 1.0])
        ax1.legend(loc="upper left")
        
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('counts'),  # we already handled the x-label with ax1
        ax2.plot(noisy_lb_rate.cumsum(), "m", label="cum. # of noisy labels"),
        ax2.plot(np.ones(total_num).cumsum(), "c", label="cum. # of all samples")
        ax2.set_ylim([0, total_num])
        ax2.tick_params(axis='y')
        ax2.legend(loc="upper right")
        plt.title("distillation effect-{}.png".format(data_source))
        plt.tight_layout()
        # plt.savefig(
        #     data_dir + "/certain_correct_rate_with_certain-classfication-rate-in-100-runs-({}-{})-{}.png".format(
        #         os.path.basename(files[0]).split("_")[-6], total_num, data_source)),
        plt.savefig(data_dir + "/CCR_in-100-runs-({})-{}-pred-vs-{}.png".format(os.path.basename(files[0]).split("_")[-5],
            data_source, compare2_lb_ind)),
        # plt.savefig(
        #     data_dir + "/certain_correct_rate_with_certain-classfication-rate-in-100-runs-({}-{})-{}.pdf".format(
        #         os.path.basename(files[0]).split("_")[-6], total_num, data_source), format="pdf")
        print("ok")
        plt.close()
        
        original_data = pd.read_csv(original, header=None).values
        ordered_data_w_lbs = original_data[sort_inds]
        concat_data = np.concatenate((np.array(sort_inds).reshape(-1, 1), rates.reshape(-1, 1)), axis=1)
        np.savetxt(
            data_dir + "/full_summary-{}_sort_CCR_({}-{}).csv".format(data_source, os.path.basename(files[0]).split("_")[7],
                                                                      total_num), concat_data, fmt="%.5f", delimiter=",",
            header="ori_id,sort_rate")


elif plot_name == "100_single_ep_corr_classification_rate":
    """
    Get the correct classification rate with 100 runs of single-epoch-training
    """
    import ipdb
    
    print("Plot_name: ", plot_name)
    from scipy.stats import spearmanr
    
    data_dirs = [
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Inception/2021-01-12T11-47-50--Inception-nonex0-factor-0-from-data9-certainFalse-theta-0-s989-100rns-train",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Inception/2021-01-12T11-47-51--Inception-nonex0-factor-0-from-data8-certainFalse-theta-0-s989-100rns-train",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Inception/2021-01-12T11-47-52--Inception-nonex0-factor-0-from-data7-certainFalse-theta-0-s989-100rns-train",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Inception/2021-01-12T11-47-54--Inception-nonex0-factor-0-from-data6-certainFalse-theta-0-s989-100rns-train",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Inception/2021-01-12T11-47-55--Inception-nonex0-factor-0-from-data4-certainFalse-theta-0-s989-100rns-train",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/100-single-epoch-runs-Inception/2021-01-12T11-47-56--Inception-nonex0-factor-0-from-data3-certainFalse-theta-0-s989-100rns-train"]
    num_smp_dataset = {"data0": 8357, "data1": 8326, "data2": 8566, "data3": 8454, "data4": 8440, "data5": 8231,
                       "data6": 8371, "data7": 8357, "data8": 8384, "data9": 7701, "mnist": 70000}
    
    for data_dir in data_dirs:
        print(data_dir)
        files = find_files(data_dir, pattern="one_ep_data_train*.csv")
        
        # get correct count in 100 rauns
        data_source = os.path.basename(files[0]).split("_")[8]
        for ind, fn in enumerate(files):
            values = pd.read_csv(fn, header=0).values
            smp_ids = values[:, 0].astype(np.int)
            true_lables = values[:, 1].astype(np.int)
            noisy_lbs = values[:, 2]
            prob = values[:, 3:]
            if ind == 0:  # the first file to get the total number (3-class) of samples
                total_num = num_smp_dataset[data_source]  # 3-class samples id
                correct_ids_w_count = []
                correct_dict_id_w_count = {key: 0 for key in np.arange(total_num)}  #
            
            pred_lbs = np.argmax(prob, axis=1)
            right_inds = np.where(pred_lbs == noisy_lbs)[0]
            correct_sample_ids = np.unique(smp_ids[right_inds])
            correct_ids_w_count += list(correct_sample_ids)
        
        correct_id_count_all = Counter(correct_ids_w_count)
        correct_dict_id_w_count.update(correct_id_count_all)
        
        # if theta == 0.975:
        #     ipdb.set_trace()
        counter_array = np.array([[key, val] for (key, val) in correct_dict_id_w_count.items()])
        sort_inds = np.argsort(counter_array[:, 1])
        sample_ids_key = counter_array[sort_inds, 0]
        # rates = counter_array[sort_inds, 1]/counter_array[:, 1].max()
        rates = counter_array[sort_inds, 1] / len(files)
        
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("sample index (sorted)"),
        ax1.set_ylabel("correct clf. rate (over 100 runs)"),
        ax1.plot(rates, label="whole data set"),
        ax1.tick_params(axis='y'),
        ax1.set_ylim([0, 1.0])
        ax1.legend(loc="upper left")
        
        plt.title("distillation effect {}.png".format(data_source))
        plt.savefig(data_dir + "/certain_correct_rate_with_certain-classfication-rate-in-100-runs-({}-{})-{}.png".format(
            os.path.basename(files[0]).split("_")[7], total_num, data_source)),
        plt.savefig(data_dir + "/certain_correct_rate_with_certain-classfication-rate-in-100-runs-({}-{})-{}.pdf".format(
            os.path.basename(files[0]).split("_")[7], total_num, data_source), format="pdf")
        print("ok")
        plt.close()
        
        concat_data = np.concatenate((np.array(sort_inds).reshape(-1, 1), rates.reshape(-1, 1)), axis=1)
        np.savetxt(data_dir + "/full_summary-{}_100_runs_sort_inds_rate_({}-{}).csv".format(data_source, os.path.basename(
            files[0]).split("_")[7], total_num), concat_data, fmt="%.5f", delimiter=",",
                   header="sort_samp_ids,sort_corr_rate")
    

elif plot_name == "100_single_ep_patient_wise_rate":
    # load original data to get patient-wise statistics
    from scipy.io import loadmat as loadmat
    import scipy.io as io
    
    print("Plot_name: ", plot_name)
    ori_data = "/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325_DATA.mat"
    ## Get the selection rate patien-wise, corr_rate also patient-wise
    sort_inds_files = "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/single-epoch-get-correct-classification-rate/100_runs_sort_inds_rate_certain.csv"
    sort_data = pd.read_csv(sort_inds_files, header=0).values
    sort_inds = sort_data[:, 0].astype(np.int)
    sort_ori_corr_rate = sort_data[:, 1]
    sort_select_rate = sort_data[:, 2]
    sort_select_corr_rate = sort_data[:, 3]
    
    original_data = loadmat(ori_data)["DATA"]
    sort_pat_ids = original_data[:, 0][sort_inds]
    labels = original_data[:, 1]
    new_mat = np.zeros((original_data.shape[0], original_data.shape[1] + 1))
    new_mat[:, 0] = np.arange(original_data.shape[0])  # tag every sample
    new_mat[:, 1:] = original_data
    train_data = {}
    test_data = {}
    true_lables = original_data[:, 0].astype(np.int)
    
    uniq_pat_ids = np.unique(sort_pat_ids)
    pat_summary = []
    for pid in uniq_pat_ids:
        pat_inds = np.where(sort_pat_ids == pid)[0]
        corr_rate = np.mean(sort_ori_corr_rate[pat_inds])
        select_rate = np.mean(sort_select_rate[pat_inds])
        select_corr_rate = np.mean(sort_select_corr_rate[pat_inds])
        pat_summary.append([pid, corr_rate, select_rate, select_corr_rate, len(pat_inds)])
    print("ok")
    pat_sort_sum = sorted(pat_summary, key=lambda x: x[1])
    np.savetxt(os.path.dirname(
        sort_inds_files) + "/100-runs-pat-ids-sorted-[pid,corr_rate,select_rate, select_corr_rate,num_samples].csv",
               np.array(pat_sort_sum), fmt="%.5f", delimiter=",")
    
    plt.plot(np.array(pat_sort_sum)[:, 1], label="ori. corr rate"),
    plt.plot(np.array(pat_sort_sum)[:, 2], label="dist. select rate"),
    plt.plot(np.array(pat_sort_sum)[:, 3], label="dist. corr rate"),
    plt.legend()
    plt.xlabel("patient index (sorted)")
    plt.ylabel("normalized rate (100 runs)")
    plt.title("Patient-wise sorted by correct rate")
    plt.savefig(os.path.dirname(sort_inds_files) + "/100-runs-patient-wise-statistics-sort-by-ori-corr-rate.png")
    plt.savefig(os.path.dirname(sort_inds_files) + "/100-runs-patient-wise-statistics-sort-by-ori-corr-rate.pdf",
                format="pdf")
    plt.close()
    
    plt.plot(np.array(pat_sort_sum)[:, 4], color="c", label="# of samples")
    plt.xlabel("patient index (sorted)")
    plt.ylabel("# of samples")
    plt.savefig(os.path.dirname(sort_inds_files) + "/100-runs-patient-wise-statistics-sort-by-num-amples.png")
    plt.savefig(os.path.dirname(sort_inds_files) + "/100-runs-patient-wise-statistics-sort-by-num-amples.pdf", format="pdf")
    plt.close()

