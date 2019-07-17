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


srun -p sleuths --gres gpu:titanblack:1 python3 classifier.py test /home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/results/1-20190325-data-trained-models/CNN_CAM/20190325-lout40-training/2019-03-29T16-55-11-data-lout40_train_test_data0-class-2-CNN_CAM-train/network

srun -p x-men python classifier.py test /home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/results/1-20190325-data-trained-models/Res_ECG_CAM/lout40/2019-06-19T17-42-46-data-20190325-3class_lout40_train_test_data9-class-2-Res_ECG_CAM-train/network