#!/bin/bash

#SBATCH --account=ml4science #imageomics-biosci  #mabrownlab   #ml4science
#SBATCH --partition=dgx_normal_q
#SBATCH --time=1-0:00:00 
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH -o /home/elhamod/projects/deepxdeplayground/SLURM/slurm-%j.out

# setup
# module reset
module load Anaconda3/2020.11
# module load gcc/8.2.0
# module load OpenMPI/4.0.5-GCC-10.2.0
# source activate landscapesvisenv
# module reset
source activate landscapesvisenv

cd /home/elhamod/projects/deepxdeplayground/AE/

train="yes" #"yes"

#variables
weights="--rec_weight 1.0 --wellspacedtrajectory_weight 1.0"


wheretosave=PINN_Failures_Beta

num_of_layers= #"--num_of_layers 5"
learning_rate="--learning_rate 5e-4" # 1e-2
epochs="--epochs 200000" #"--epochs 400000"
patience_scheduler="--patience_scheduler 5000"
cosine="--cosine_Scheduler_patience 2000"

models_path=/home/elhamod/projects/characterizing-pinns-failure-modes/pbc_examples/saved_models/Beta_L1


vlevel=30
vmin=-1
vmax=-1

x="-1.2:1.2:25"

whichlosses=("total") # "b" "f" "u")
betas=(20.0 30.0) #(1.0 10.0  # 40.0)
ls=(1.0)

key_models="0" 

resume="--resume"

for beta in "${betas[@]}"
do
    for L in "${ls[@]}"
    do
        everynth="--every_nth 10"

        if [ "$train" == "yes" ]; then
            python train_PINN_Failures.py --model_file /home/elhamod/projects/deepxdeplayground/AE/saved_models/$wheretosave/beta$beta\_L$L/model.pt --model_folder $models_path/beta$beta\_L$L $weights $everynth $num_of_layers $learning_rate $resume $epochs $patience_scheduler $cosine
        fi

        everynth="--every_nth 1"
        
        for whichloss in "${whichlosses[@]}"
        do
            python plot_PINN_Failures.py --model_file /home/elhamod/projects/deepxdeplayground/AE/saved_models/$wheretosave/beta$beta\_L$L/model.pt --model_folder $models_path/beta$beta\_L$L --vlevel $vlevel --vmin $vmin --vmax $vmax --x=$x --whichloss $whichloss $everynth --L $L --beta $beta --key_models $key_models --key_modelnames "beta=$beta" #--density_type CKA 
        done
    done
done


exit;