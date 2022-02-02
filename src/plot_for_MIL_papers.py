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

from dataio import introduce_label_noisy

# import matplotlib.pylab as pylab
# base = 22
# args = {'legend.fontsize': base - 8,
#           'figure.figsize': (10, 7),
#          'axes.labelsize': base-4,
#         #'weight' : 'bold',
#          'axes.titlesize':base,
#          'xtick.labelsize':base-8,
#          'ytick.labelsize':base-8}
# pylab.rcParams.update(args)

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


def get_scalar_performance_matrices_2classes(true_labels, pred_logits, if_with_logits=False):
	"""
	Get all relavant performance metrics
	:param true_labels: 1d array, int labels
	:param predictions: 1d array, logits[:, 1]
	:param if_with_logits: if with logits, it is with probabilities, otherwise are predicted int labels
	:return:
	"""
	# get predicted labels based on optimal threshold
	if if_with_logits:
		cutoff_thr, auc = find_optimal_cutoff(true_labels, pred_logits)
		pred_labels = (pred_logits > cutoff_thr).astype(np.int)
	else:
		pred_labels = pred_logits
		auc = metrics.roc_auc_score(true_labels, pred_logits)
	
	confusion = metrics.confusion_matrix(true_labels, pred_labels)
	TN = confusion[0][0]
	FN = confusion[1][0]
	TP = confusion[1][1]
	FP = confusion[0][1]
	
	# accuracy
	accuracy = (TP + TN) / np.sum(confusion)
	# Sensitivity, hit rate, recall, or true positive rate
	sensitivity = TP / (TP + FN)
	# Specificity or true negative rate
	specificity = TN / (TN + FP)
	# Precision or positive predictive value
	precision = TP / (TP + FP)
	# Negative predictive value
	NPV = TN / (TN + FN)
	# Fall out or false positive rate
	FPR = FP / (FP + TN)
	# False negative rate
	FNR = FN / (TP + FN)
	# False discovery rate
	FDR = FP / (TP + FP)
	# F1-score
	F1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
	# F1_score = TP / (TP + 0.5*(FN + FP))
	# get tpr, fpr
	fpr, tpr, _ = metrics.roc_curve(true_labels, pred_logits)
	# Matthews corrrelation coefficient
	MCC = metrics.matthews_corrcoef(true_labels, pred_labels)
	MCC = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
	
	return accuracy, sensitivity, specificity, precision, F1_score, auc, fpr, tpr, MCC


def patient_wise_performance_ICLR_generate_latex():
	global data_dirs, data_dir, num_spec, files, fn, data, ind
	# data_dirs
	data_dirs = [
		# r"\\I7-4770\d\metabolite_maskAE\autoenc-src\results\2021-05-08T16-17-26-Hatami2018_with_3pool-NoDA-46spec-LOO"
		r"C:\1-study\FIAS\1-My-papers\1-11-submitted-2021.03 MLHC patient-wise-classification\results\results monday\2021-03-15T15-47-13-InceptionV3-NoDA-31spec"]
	file_patterns = ["*test.csv", "*-test_doc.csv"]
	# file = "classifier4-spec51-CV9--ROC-AUC-[n_cv_folds,n_spec_per_pat].csv"
	# data = pd.read_csv(os.path.join(data_dirs[0], file), header=None).values
	#
	# plt.figure(figsize=[7, 4.8])
	# mean = np.mean(data, axis=0)
	# std = np.std(data, axis=0)
	# plt.errorbar(np.arange(1, 52, 5), mean, yerr=std, capsize = 9)
	# plt.xlabel("# of spectra per paptient ($N$)")
	# plt.ylabel("area under the ROC curve")
	# plt.tight_layout()
	# plt.savefig(os.path.join(data_dirs[0], "classifier4-spec51-CV9--ROC-AUC[1,51].png"))
	# plt.savefig(os.path.join(data_dirs[0], "classifier4-spec51-CV9--ROC-AUC[1,51].pdf"), format="pdf")
	for data_dir in data_dirs:
		for num_spec in [1, 31]:
			spec_file_patterns = ["*spec{}-".format(num_spec) + pt for pt in file_patterns]
			for pattern, group in zip(spec_file_patterns, ["all-CVs.csv", "doctors.csv"]):
				files = find_files(data_dir, pattern=pattern)
				
				performance = {"ACC": np.empty((0,)), "patient_AUC": np.empty((0,)), "AUC": np.empty((0,)),
				               "SEN": np.empty((0,)), "SPE": np.empty((0,)), "F1_score": np.empty((0,)),
				               "MCC": np.empty((0,))}
				performance_summary = []
				folder_name = ""
				if len(files) > 0:
					for fn in files:
						latex_string1 = ""
						if "alwaystestmodeFalse" in os.path.basename(os.path.dirname(fn)):
							folder_name = os.path.basename(os.path.dirname(fn)).replace("alwaystestmodeFalse", "+DA")
						elif "alwaystestmodeTrue" in os.path.basename(os.path.dirname(fn)):
							folder_name = os.path.basename(os.path.dirname(fn)).replace("alwaystestmodeTrue", "NoDA")
						else:
							folder_name = os.path.basename(os.path.dirname(fn))
						data = pd.read_csv(fn, header=None).values
						sample_ids = data[:, 0]
						patient_ids = data[:, 1]
						true_labels = data[:, 2]
						pred_logits = data[:, 3:]
						
						# get the prob of one patient, average them, get predicted label, get right or wrong.
						# get true labels for all unique patients
						uniq_patients, uniq_patients_ind = np.unique(patient_ids, return_index=True)
						patient_true_lb = {pat_id: true_labels[index] for pat_id, index in
						                   zip(uniq_patients, uniq_patients_ind)}
						# get the index of each patient bag
						uniq_patients_inds = {pat_id: [] for pat_id in uniq_patients}
						uniq_patients_pred_lb = {pat_id: [] for pat_id in uniq_patients}
						for ind, pat_id in enumerate(patient_ids):
							uniq_patients_inds[pat_id].append(ind)
						# get aggregated prob of each patient
						aggregated_prob_per_patient = {pat_id: np.mean(pred_logits[uniq_patients_inds[pat_id]], axis=0)
						                               for pat_id in uniq_patients}
						# get predicted label from the aggregated prob for each patient
						aggregated_pred_lb_per_patient = {pat_id: np.argmax(aggregated_prob_per_patient[pat_id], axis=0)
						                                  for pat_id in aggregated_prob_per_patient.keys()}
						# patient_auc = np.sum([aggregated_pred_lb_per_patient[pat_id] == patient_true_lb[pat_id] for pat_id in uniq_patients]) / len(uniq_patients) * 1.0
						# extract label and aggregated prob for each patient from the dicts
						true_labels_pat = []
						prob_pat = []
						for pat_id in aggregated_prob_per_patient.keys():
							true_labels_pat.append(patient_true_lb[pat_id])
							prob_pat.append(aggregated_prob_per_patient[pat_id])
						# Get the patient-wise accuracy
						patient_auc = metrics.roc_auc_score(np.array(true_labels_pat), np.array(prob_pat)[:,
						                                                               1])  # has to be 1, that is the defination
						
						accuracy, sensitivity, specificity, precision, F1_score, auc, fpr, tpr, mcc = get_scalar_performance_matrices_2classes(
							true_labels, pred_logits[:, 1], if_with_logits=True)
						
						performance["ACC"] = np.append(performance["ACC"], accuracy)
						performance["SEN"] = np.append(performance["SEN"], sensitivity)
						performance["SPE"] = np.append(performance["SPE"], specificity)
						performance["AUC"] = np.append(performance["AUC"], auc)
						performance["F1_score"] = np.append(performance["F1_score"], F1_score)
						performance["MCC"] = np.append(performance["MCC"], mcc)
						performance["patient_AUC"] = np.append(performance["patient_AUC"], patient_auc)
						
						latex_string1 = "${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$  & ${:.2f} \pm {:.2f} $ & ${:.2f} \pm {:.2f}$".format(
							np.mean(performance["AUC"]), np.std(performance["AUC"]),
							np.mean(performance["patient_AUC"]), np.std(performance["patient_AUC"]),
							np.mean(performance["F1_score"]), np.std(performance["F1_score"]),
							np.mean(performance["MCC"]), np.std(performance["MCC"]))
						latex_string2 = "{} & {}   & {}   &{} $".format("AUC", "patient_AUC", "F1_score", "MCC")
						performance_summary.append(["SPEC{}".format(num_spec) + "performance of {}\n".format(
							os.path.dirname(fn)) + "AUC: mean-{:.3f}, std-{:.3f}\n".format(np.mean(performance["AUC"]),
						                                                                   np.std(performance[
							                                                                          "AUC"])) + "patient AUC: mean-{:.3f}, std-{:.3f}\n".format(
							np.mean(performance["patient_AUC"]), np.std(performance["patient_AUC"]))
						
						                            + "F1-score: mean-{:.3f}, std-{:.3f}\n".format(
							np.mean(performance["F1_score"]),
							np.std(performance["F1_score"])) + "MCC: mean-{:.3f}, std-{:.3f}\n".format(
							np.mean(performance["MCC"]),
							np.std(performance["MCC"])) + "Sensitivity: mean-{:.3f}, std-{:.3f}\n".format(
							np.mean(performance["SEN"]),
							np.std(performance["SEN"])) + "specificity: mean-{:.3f}, std-{:.3f}\n".format(
							np.mean(performance["SPE"]), np.std(performance[
								                                    "SPE"])) + "Bag-accuracy, Patient-ACC, AUC, F1-score, MCC\n" + latex_string1 + "----------------------------------------------\n"])
					
					np.savetxt(os.path.join(data_dir,
					                        "spec{}-AUC-{:.4f}-performance-summarries-of-{}-{}".format(num_spec,
					                                                                                   np.mean(
						                                                                                   performance[
							                                                                                   "AUC"]),
					                                                                                   os.path.basename(
						                                                                                   data_dir).split(
						                                                                                   "-")[-1],
					                                                                                   group)),
					           np.array(performance_summary), fmt="%s", delimiter=",")
					print("---SPEC{}---{}-------{}---------".format(num_spec, os.path.basename(os.path.dirname(fn)),
					                                                group))
					# print("Bag ACC: mean-{:.3f}, std-{:.3f}\n".format(
					#     np.mean(performance["ACC"]),
					#     np.std(performance["ACC"])))
					# print("patient acc: mean-{:.3f}, std-{:.3f}\n".format(
					#     np.mean(performance["patient_AUC"]),
					#     np.std(performance["patient_AUC"])))
					# print("AUC: mean-{:.3f}, std-{:.3f}\n".format(
					#     np.mean(performance["AUC"]),
					#     np.std(performance["AUC"])))
					# print("F1-score: mean-{:.3f}, std-{:.3f}\n".format(
					#     np.mean(performance["F1_score"]),
					#     np.std(performance["F1_score"])))
					# print("MCC: mean-{:.3f}, std-{:.3f}\n".format(
					#     np.mean(performance["MCC"]),
					#     np.std(performance["MCC"])))
					# print("Sensitivity: mean-{:.3f}, std-{:.3f}\n".format(
					#     np.mean(performance["SEN"]),
					#     np.std(performance["SEN"])))
					# print("specificity: mean-{:.3f}, std-{:.3f}\n".format(
					#     np.mean(performance["SPE"]),
					#     np.std(performance["SPE"])))
					print(latex_string2)
					print(latex_string1)
					print("--------------END-----------------")
				else:
					print(data_dir, "No {} , {} data!".format(pattern, num_spec))


if __name__ == "__main__":
	
	plot_name = "patient_wise_performance_ICLR_generate_latex"
	if plot_name == "patient_wise_performance_ICLR_generate_latex":
		patient_wise_performance_ICLR_generate_latex()
	
	elif plot_name == "patient_wise_MLHC_attention":
		from scipy.stats import ks_2samp
		
		data_dir = r"\\I7-4770\d\metabolite_maskAE\autoenc-src\results\2021-05-08T01-38-12-Hatami2018_with_3pool-+DA-46spec-LOO"
		
		## plot the fucn_of_auc_as_num_spectra
		
		for num_spec in [16, 31]:
			files = find_files(data_dir, pattern="*spec{}*attention_weights*.csv".format(num_spec))
			if len(files) > 0:
				all_attentions = np.empty((0, 16))
				all_labels = []
				for fn in files:
					data = pd.read_csv(fn, header=None).values
					labels = data[:, 0]
					attention = data[:, 1:]
					all_attentions = np.vstack((all_attentions, attention))
					all_labels += list(labels)
					print("ok")
					colors = ["c", "m"]
					for c in range(2):
						clas = 1 - c
						plt.hist(attention[labels == clas].reshape(-1), bins=20, color=colors[clas], alpha=0.85,
						         label="Class {}".format(clas), density=True)
					plt.legend()
					stats, p_value = ks_2samp(attention[labels == 0].reshape(-1), attention[labels == 1].reshape(-1))
					plt.title("pvalue {:.3E}".format(p_value))
					plt.savefig(fn[0:-4] + ".png")
					plt.savefig(fn[0:-4] + ".pdf", format="pdf")
					plt.close()
			
			all_labels = np.array(all_labels).astype(np.int)
			
			for c in range(2):
				clas = 1 - c
				plt.hist(all_attentions[all_labels == clas].reshape(-1), bins=20, color=colors[clas], alpha=0.85,
				         label="Class {}".format(clas), density=True)
			
			plt.legend()
			plt.savefig(os.path.dirname(fn) + "/spec-{}-all-CVs-attention-hist.png".format(num_spec))
			plt.savefig(os.path.dirname(fn) + "/spec-{}-all-CVs-attention-hist.pdf".format(num_spec), format="pdf")
			plt.close()
		
	elif plot_name == "patient-wise-MLHC-func-AUC":
		data_dirs = [
			# r"C:\Users\LDY\Desktop\1-all-experiment-results\Gk-patient-wise\results 1-51 spectra\2021-03-18T23-16-03-InceptionV3_with_attention-+DA-1spec",
			# r"C:\Users\LDY\Desktop\1-all-experiment-results\Gk-patient-wise\results 1-51 spectra\2021-03-19T01-53-01-InceptionV3-+DA-51spec",
			# r"C:\Users\LDY\Desktop\1-all-experiment-results\Gk-patient-wise\results 1-51 spectra\2021-03-19T04-32-29-MLP_classifier_with_attention-+DA-51spec",
			# r"C:\Users\LDY\Desktop\1-all-experiment-results\Gk-patient-wise\results 1-51 spectra\2021-03-19T09-45-52-MLP_classifier-NoDA-51spec"
			r"C:\1-study\FIAS\1-My-papers\1-11-submitted-2021.03 MLHC patient-wise-classification\results\results-Hatami\2021-05-08T03-36-33-Hatami2018_with_attention-+DA-46spec-LOO",
			r"C:\1-study\FIAS\1-My-papers\1-11-submitted-2021.03 MLHC patient-wise-classification\results\results-Hatami\2021-05-08T01-38-12-Hatami2018_with_3pool-+DA-46spec-LOO"]
		
		## plot the fucn_of_auc_as_num_spectra
		plt.figure(figsize=[6, 4])
		# plt.subplot(1, 2, 1)
		for data_dir, model_name in zip(data_dirs[0:2], ["Hatami-Att", "Hatami-3Pool"]):
			full_metrix = pd.read_csv(os.path.join(data_dir, "test-AUC-[n_cv_folds,n_spec_per_pat]-LOO.csv"),
			                          header=None).values
			data_mean = np.mean(full_metrix, axis=0)
			data_std = np.std(full_metrix, axis=0)
			plt.errorbar(np.arange(1, 50, 5), data_mean, data_std, capsize=9, label=model_name)
			# plt.legend()
			plt.ylim([0.6, 0.9])
			plt.ylabel("AUC")
		# plt.xlabel("number of spectra per bag (M)")
		# plt.subplot(1, 2, 2)
		# for data_dir, model_name in zip(data_dirs[0:2], ["Hatami-Att", "Hatami-3Pool"]):
		#     full_metrix = pd.read_csv(os.path.join(data_dir,
		#                                            "test-AUC-[n_cv_folds,n_spec_per_pat]-LOO.csv"),
		#                               header=None).values
		#     data_mean = np.mean(full_metrix, axis=0)
		#     data_std = np.std(full_metrix, axis=0)
		#     plt.errorbar(np.arange(1, 50, 5), data_mean, data_std, capsize=9,
		#                  label=model_name)
		#     plt.ylim([0.6, 0.9])
		plt.legend(loc="best", fontsize=16)
		plt.ylabel("AUC", fontsize=15)
		plt.xticks(fontsize=15)
		plt.yticks(fontsize=15)
		plt.xlabel("number of spectra per bag (M)", fontsize=15)
		plt.tight_layout()
		plt.savefig(os.path.join(os.path.dirname(data_dir), "Hatami-3Pool-Att-func_AUC_of_num_spectra.pdf"), format="pdf")
	
		plt.savefig(os.path.join(os.path.dirname(data_dir), "Hatami-3Pool-Att-func_AUC_of_num_spectra.png"))

	elif plot_name == "patient-wise-NeurIPS-stats-test-of-results":
		from scipy.stats import ttest_ind
		
		hatami_3pool_wDA = np.array([[0.670, 0.818, 0.816, 0.845, 0.812, 0.829, 0.886, 0.874, 0.883, 0.871],
			[0.638, 0.690, 0.683, 0.671, 0.700, 0.694, 0.687, 0.708, 0.696, 0.726],
			[0.668, 0.792, 0.788, 0.818, 0.822, 0.811, 0.827, 0.810, 0.789, 0.828],
			[0.660, 0.765, 0.798, 0.808, 0.812, 0.828, 0.831, 0.823, 0.819, 0.828],
			[0.716, 0.852, 0.873, 0.852, 0.883, 0.898, 0.884, 0.871, 0.881, 0.859]])
		hatami_3pool_withoutDA = np.array([[0.684, 0.802, 0.832, 0.835, 0.842, 0.868, 0.862, 0.855, 0.836, 0.868],
			[0.640, 0.721, 0.734, 0.692, 0.711, 0.706, 0.684, 0.671, 0.723, 0.708],
			[0.668, 0.794, 0.790, 0.824, 0.825, 0.821, 0.814, 0.818, 0.826, 0.815],
			[0.664, 0.791, 0.808, 0.793, 0.812, 0.826, 0.843, 0.809, 0.811, 0.817],
			[0.731, 0.832, 0.872, 0.840, 0.891, 0.892, 0.878, 0.877, 0.873, 0.868]])
		hatami_att_withDA = np.array([[0.698, 0.794, 0.826, 0.780, 0.809, 0.850, 0.873, 0.824, 0.892, 0.856],
			[0.686, 0.684, 0.697, 0.696, 0.714, 0.694, 0.709, 0.702, 0.680, 0.692],
			[0.705, 0.808, 0.770, 0.800, 0.804, 0.798, 0.781, 0.763, 0.793, 0.794],
			[0.671, 0.790, 0.796, 0.797, 0.793, 0.825, 0.816, 0.827, 0.809, 0.811],
			[0.768, 0.839, 0.855, 0.844, 0.880, 0.873, 0.864, 0.861, 0.862, 0.862]])
		hatami_att_withoutDA = np.array([[0.702, 0.812, 0.844, 0.815, 0.806, 0.855, 0.818, 0.850, 0.880, 0.877],
			[0.649, 0.686, 0.713, 0.677, 0.715, 0.718, 0.712, 0.670, 0.667, 0.743],
			[0.691, 0.763, 0.782, 0.790, 0.781, 0.802, 0.793, 0.780, 0.791, 0.793],
			[0.685, 0.803, 0.814, 0.803, 0.808, 0.831, 0.820, 0.842, 0.809, 0.820],
			[0.732, 0.841, 0.874, 0.853, 0.881, 0.847, 0.881, 0.864, 0.850, 0.856]])
		bb = 0
		MLP_3pool_withoutDA = np.array(
			[[0.751, 0.807, 0.620], [0.696, 0.685, 0.573], [0.687, 0.714, 0.696], [0.720, 0.772, 0.747],
				[0.726, 0.708, 0.771], ])
		MLP_3pool_withDA = np.array(
			[[0.744, 0.829, 0.796], [0.685, 0.613, 0.503], [0.681, 0.717, 0.736], [0.689, 0.771, 0.825], [0.748, 0.778, 0.739]])
		MLP_att_withoutDA = np.array(
			[[0.751, 0.807, 0.620], [0.696, 0.685, 0.573], [0.687, 0.714, 0.696], [0.720, 0.772, 0.747],
				[0.726, 0.708, 0.771], ])
		MLP_att_withDA = np.array(
			[[0.724, 0.859, 0.854], [0.681, 0.693, 0.691], [0.690, 0.735, 0.748], [0.700, 0.769, 0.823], [0.739, 0.815, 0.827]])
		
		for ind in range(hatami_3pool_wDA.shape[1]):
			for ind2 in range(hatami_3pool_wDA.shape[1]):
				_, p_value = ttest_ind(hatami_3pool_withoutDA[:, ind], hatami_3pool_withoutDA[:, ind2])
				if p_value <= 0.3:
					print("3Pool withoutDA: {}, 3POol withoutDA {} significant. p-value: {}".format(ind, ind2, p_value))
		
		dd = np.mean(np.array(
			[0.001616149873240889, 0.001480193984922760, 0.005165387719800698, 0.003061095939849105, 0.003432514553562985,
				0.00636372703405346, 0.010885652653437545, 0.001579911398216797, 0.003046545456535175]))
		
