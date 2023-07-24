#!/bin/bash

#SBATCH --account=mabrownlab #imageomics-biosci  #mabrownlab   #ml4science
#SBATCH --partition=dgx_normal_q
#SBATCH --time=0-24:00:00 
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

train="no" #"yes"

#variables
wheretosave=PINN_NTK
weights="--rec_weight 1.0 --anchor_weight 10000.0 --anchor_mode circle"

everynth="--every_nth 10"
num_of_layers= #"--num_of_layers 5"
layers="--layers_AE 1000 500 100"
learning_rate="--learning_rate 1e-5"
epochs="--epochs 100000"
patience_scheduler="--patience_scheduler 100000"
resume= #"--resume"
cosine="--cosine_Scheduler_patience 600"

models_path=/home/elhamod/projects/deepxdeplayground/PINN_NTK    

if [ "$train" == "yes" ]; then
    python train_Convection.py --model_file /home/elhamod/projects/deepxdeplayground/AE/saved_models/$wheretosave/model.pt --model_folder $models_path $weights $everynth $num_of_layers $learning_rate $resume $layers $epochs $patience_scheduler $cosine
fi







vlevel=30
vmin=-1
vmax=-1

x="-1.2:1.2:25"

whichlosses=("r" "ic" "bc" "u")

everynth="--every_nth 1"

key_models="0 801" 
key_modelnames="adaptive constant"

for whichloss in "${whichlosses[@]}"
do
    python plot_PINN_NTK.py --model_file /home/elhamod/projects/deepxdeplayground/AE/saved_models/$wheretosave/model.pt --model_folder $models_path --vlevel $vlevel --vmin $vmin --vmax $vmax --x=$x --whichloss $whichloss --key_models $key_models $layers --key_modelnames $key_modelnames $everynth --density_type CKA 
done



exit;