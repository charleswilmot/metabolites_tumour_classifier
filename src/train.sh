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


srun -p sleuths -w jetski --mem=20000 --reservation triesch-shared --gres gpu:rtx2080ti:1 python3 classifier.py test /home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/results/2019-07-19T17-00-06-data-20190325-3class_lout40_train_test_data5-class-2-Res_ECG_CAM-train/network

srun -p x-men python classifier.py test /home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/results/1-20190325-data-trained-models/Res_ECG_CAM/lout40/2019-06-19T17-42-46-data-20190325-3class_lout40_train_test_data9-class-2-Res_ECG_CAM-train/network

srun -p x-men python --pty env PYTHONPATH=~software/tensorflow-py3-amd64-gpu  classifier.py test /home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/results/2019-07-19T15-42-14-data-20190325-3class_lout40_val_data5-class-2-Res_ECG_CAM-train/network