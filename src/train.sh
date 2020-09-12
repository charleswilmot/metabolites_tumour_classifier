$TF_CPP_MIN_LOG_LEVEL = 3
rm -r /tmp/test*

python3 classifier.py \
-d 288 4 \
-n true \
-p 0.95 \
-A None \
-b 450 \
-vvv train -L mse -S 50 -e 4000 -t 100 ../data/ /tmp/test0

python3 classifier.py \
-d 288 288 4 \
-n true true \
-p 0 0 \
-A lrelu None \
-b 450 \
-vvv train -L mse -S 50 -e 4000 -t 100 ../data/ /tmp/test1

python3 classifier.py \
-d 288 4 \
-n false \
-p 0.99 \
-A None \
-b 450 \
-vvv train -L mse -S 50 -e 4000 -t 100 ../data/ /tmp/test2

python3 classifier.py \
-d 288 4 \
-n false \
-p 0.9 \
-A None \
-b 450 \
-vvv train -L mse -S 50 -e 4000 -t 100 ../data/ /tmp/test3

"
/home/epilepsy-data/data/metabolites/results/2020-04-23T11-52-04-class2-Res_ECG_CAM-new-aug_noisex5-0.1-train/  2
/home/epilepsy-data/data/metabolites/results/2020-04-23T12-16-53-class2-Res_ECG_CAM-new-aug_noisex5-0.1-train/  7

"
srun -p sleuths -w jetski --mem=8000 --reservation triesch-shared --gres gpu:rtx2080ti:1 python3 classifier.py test --restore_from=/home/epilepsy-data/data/metabolites/results/2020-04-23-17-48-34_noisex5_factor_0.02_from-epoch_99_from-lout40_data5_train/network

srun -p sleuths -w jetski --mem=10000 --reservation triesch-shared --gres gpu:rtx2080ti:1 python3 classifier.py train

srun -p sleuths -w jetski --mem=5000 --reservation triesch-shared python3 classifier.py train

srun -p sleuths -w jetski --mem=5000 --reservation triesch-shared python3 classifier.py train --aug_method="ops_mean" --aug_scale=0.5 --from_epoch=3 --aug_folds=5


srun -p sleuths -w jetski --mem=5000 --reservation triesch-shared python3 classifier.py train

srun -p sleuths -w jetski --mem=5000 --reservation triesch-shared --gres gpu:rtx2080ti:1 python3 vae_metabolite.py

srun -p sleuths -w jetski --mem=5000 --reservation triesch-shared python3 classifier.py test --restore_from=/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/results/2019-10-08T14-22-53-data-lout40-data5-1d-class-2-Res_ECG_CAM-relu-aug_ops_meanx10-0.3-train-auc0.766


srun -p sleuths -w jetski --mem=5000 --reservation triesch-shared python3 classifier.py test /home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/results/saved_certain/2019-10-08T14-51-40-data-lout40-data5-1d-class2-Res_ECG_CAM-certainEp3-aug_ops_meanx10-0.3-train-auc0.858

srun -p x-men python --pty env PYTHONPATH=~software/tensorflow-py3-amd64-gpu  classifier.py test /home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/results/2019-07-19T15-42-14-data-20190325-3class_lout40_val_data5-class-2-Res_ECG_CAM-train/network