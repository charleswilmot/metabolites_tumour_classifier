

"
/home/epilepsy-data/data/metabolites/results/2020-04-23T11-52-04-class2-Res_ECG_CAM-new-aug_noisex5-0.1-train/  2
/home/epilepsy-data/data/metabolites/results/2020-04-23T12-16-53-class2-Res_ECG_CAM-new-aug_noisex5-0.1-train/  7
/home/elu/LU/2_Neural_Network/2_NN_projects_codes/Epilepsy/metabolites_tumour_classifier/data/20190325/20190325-3class_lout40_train_test_data7.mat
"

srun -p sleuths -w jetski --mem=15000 --reservation triesch-shared --gres gpu:rtx2080ti:1  python3 Hpopt.py

srun -p sleuths -w speedboat --mem=6000 --reservation triesch-shared --gres gpu:rtx2070super:1 python3 harrisonNN.py
srun -p sleuths -w jetski --mem=8000 --reservation triesch-shared --gres gpu:rtx2080ti:1 python3 classifier.py test --restore_from=/home/epilepsy-data/data/metabolites/results/2020-04-23-17-48-34_noisex5_factor_0.02_from-epoch_99_from-lout40_data5_train/network

srun -p sleuths -w jetski --mem=10000 --reservation triesch-shared --gres gpu:rtx2080ti:1 python3 classifier.py train

srun -p sleuths -w jetski --mem=5000 --reservation triesch-shared python3 classifier.py train

srun -p sleuths -w jetski --mem=5000 --reservation triesch-shared python3 classifier.py train --aug_method="ops_mean" --aug_scale=0.5 --from_epoch=3 --aug_folds=5
srun -p sleuths -w speedboat --mem=6000 --reservation triesch-shared python3 classifier.py

srun  -p sleuths -w speedboat --mem=8000 --reservation triesch-shared python3 classifier.py

# successfully allocate space for runing wtih bash
salloc -p sleuths -w speedboat --mem=20000 --reservation triesch-shared --gres gpu:rtx2070super:1 -t UNLIMITED
salloc -p sleuths -w scuderi --mem=20000 --gres gpu:titanxp:1 -t UNLIMITED
salloc -p sleuths -w turbine --mem=20000 --gres gpu:titanx:1 -t UNLIMITED
salloc -p sleuths -w jetski --mem=20000 --reservation triesch-shared --gres gpu:rtx2080ti:1 -t UNLIMITED
salloc -p sleuths -w vane --mem=20000 --gres gpu:rtx2080tirev.a:1 -t UNLIMITED


# Usufull SLURM command
scontrol show job 440644

sinfo -p sleuths -n vane -o %e
sinfo -p sleuths -n jetski -e
sinfo -p sleuths -n turbine -o %e
sinfo -p sleuths -n scuderi -o %e



login jetski (ssh jetski) -- free
scontrol show nodes | grep gpu -B 3 -A 3   # check GPU nodes, scuderi, turbine and vane
srun -p sleuths -w jetski --mem=15000 --reservation triesch-shared --gres gpu:rtx2080ti:1 --pty bash -i
srun -p sleuths -w turbine -N 2 --mem=15000 --pty bash -i

srun -p sleuths --mem=6000 python3 classifier.py


srun -p sleuths -w scuderi --mem=15000 --gres gpu:titanxp:1 python3 cluster.py
srun -p sleuths -w turbine --mem=10000 --gres gpu:titanx:1 python3 cluster.py
srun -p sleuths -w jetski --mem=15000  --reservation triesch-shared --gres gpu:rtx2080ti:1 python3 vae_main.py
srun -p sleuths -w vane --mem=15000 --gres gpu:rtx2080tirev.a:1
srun -p x-men --mem=15000 python3 acti_max.py
srun -p x-men --mem=15000 python3 plot_average_plots.py

srun -p sleuths -w jetski --mem=15000 --reservation triesch-shared --gres gpu:rtx2080ti:1  python3 EPG_classification.py
srun -p x-men python3 EPG_classification.py
srun -p sleuths -w jetski --mem=15000 --reservation triesch-shared --gres gpu:rtx2080ti:1  python3 acti_max.py



