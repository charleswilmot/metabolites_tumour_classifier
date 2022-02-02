import fnmatch
import os


def find_files(directory, pattern='*.csv'):
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))

    return files


def find_folderes(directory, pattern='*.csv'):
    folders = []
    for root, dirnames, filenames in os.walk(directory):
        for subfolder in fnmatch.filter(dirnames, pattern):
            folders.append(os.path.join(root, subfolder))

    return folders


plot_name = "rename_test_folders"

if plot_name == "rename_test_folders":
    print("Plot_name: ", plot_name)
    results = "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN"
    folders = find_folderes(results, pattern="*-test")
    pattern = "accuracy_step_0.0_acc_*"
    for fn in folders:
        print(fn)
        test_result = find_files(fn, pattern=pattern)
        
        if len(test_result) >= 1:
            splits = os.path.basename(test_result[0]).split("_")
            new_name = os.path.basename(fn).replace("_", "-")
            auc = splits[-2]
            # os.rename(fn, os.path.join(os.path.dirname(fn), new_name))
            os.rename(fn, os.path.join(os.path.dirname(fn), new_name + "-{}".format(auc)))
    
    # new_name = os.path.basename(fn).replace("MLP", "Res_ECG_CAM")  # os.rename(fn, os.path.join(os.path.dirname(fn), new_name))

elif plot_name == "rename_files":
    print("Plot_name: ", plot_name)
    results = "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/9-train-with-MNIST-MLP/2021-02-10T08-18-38--MLP-both_meanx0-factor-0-from-mnist-certainFalse-theta-1-s5058-100rns-train/certains"
    filenames = find_files(results, pattern="*.csv")
    for fn in filenames:
        print(fn)
        new_name = os.path.basename(fn).replace("-", "_")
        os.rename(fn, os.path.join(os.path.dirname(fn), new_name))
    

elif plot_name == "move_folder":
    import shutil
    
    print("Plot_name: ", plot_name)
    
    dirs = ["/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA2-Res_ECG_CAM",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-Inception",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-RandomDA-MLP",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-RNN",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA+noise-Res_ECG_CAM",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-Inception",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-MLP",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-Res_ECG_CAM",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-Res7-Res_ECG_CAM",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN"]
    # dirs = ["C:/Users/LDY/Desktop/metabolites-0301/metabolites_tumour_classifier/results/1-Pure-new-Inception"]
    
    for dd in dirs:
        sub_folders = find_folderes(dd, pattern=os.path.basename(dd) + "-data*")
        for sub_fd in sub_folders:
            # test_folders = find_folderes(fd, pattern="*-test-0.*")
            # os.rename(fd, fd+"-len{}".format(len(test_folders)))
            test_folders = find_folderes(sub_fd, pattern="*-train")
            for t_fd in test_folders:
                print("sub folders", test_folders)
                data_cv = os.path.basename(t_fd).split("-")[-3]
                new_dest_root = dd
                
                if not os.path.isdir(new_dest_root):
                    os.mkdir(new_dest_root)
                else:
                    print(
                        "Move {} to {}".format(os.path.basename(t_fd), os.path.join(new_dest_root, os.path.basename(t_fd))))
                    shutil.move(t_fd, os.path.join(new_dest_root, os.path.basename(t_fd)))
    

elif plot_name == "delete_folders":
    import shutil
    
    print("Plot_name: ", plot_name)
    target_dirs = [
        "/home/epilepsy-data/data/PPS-rats-from-Sebastian/resultsl-7rats/run_dim_128_loss_weights_EPG_anomaly_2021-03-12T08-28-34_pps20h_ctrl100h_LOO_32141",
        "/home/epilepsy-data/data/PPS-rats-from-Sebastian/resultsl-7rats/run_dim_16_loss_weights_EPG_anomaly_2021-03-11T13-09-47_pps20h_ctrl100h_LOO_1227"]
    for dd in target_dirs:
        print("Deleting ", dd)
        shutil.rmtree(dd)
        print("Done")
    print("All Done")


elif plot_name == "generate_empty_folders":
    print("Plot_name: ", plot_name)
    dirs = ["/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-Res_ECG_CAM",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA2-Res_ECG_CAM",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA+noise-Res_ECG_CAM",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-Res_ECG_CAM",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-RNN",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-Res7-Res_ECG_CAM",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-Inception",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-RNN",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-RandomDA-MLP",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-MLP",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/2-randomDA-Inception",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-RNN",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/3-certain-DA-MLP",
        "/home/epilepsy-data/data/metabolites/2020-08-30-restuls_after_review/1-Pure-new-Inception", ]
    
    for fd in dirs:
        for jj in range(10):
            new_dirs = os.path.join(fd, os.path.basename(fd) + "-data{}".format(jj))
            print("Make dir ", new_dirs)
            os.mkdir(new_dirs)
    
