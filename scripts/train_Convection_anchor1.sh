#!/bin/bash

#SBATCH --account=imageomics-biosci  #imageomics-biosci  #mabrownlab   #ml4science
#SBATCH --partition=dgx_normal_q
#SBATCH --time=0-4:00:00 
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

num_of_layers=3

epochs=80000 #40000

#variables
wheretosave=Convection_anchor1
weights="--polars_weight 100"
resume= #"--resume" #""

beta=30.0
everynth="--every_nth 1"

models_path=/home/elhamod/projects/deepxdeplayground/MTL_same_initialization/Beta30/CW


if [ "$train" == "yes" ]; then
    python train_Convection.py --model_file /home/elhamod/projects/deepxdeplayground/AE/saved_models/$wheretosave/model.pt --model_folder $models_path $weights $everynth --epochs $epochs $resume
fi



key_models="0" 
key_modelnames="CW" #NOTE: make sure the order here is correct!!





vlevel=30
vmin=-1
vmax=-1

x="-1.2:1.2:25"

whichlosses=("residual" "ic" "bc")

colorFromGridOnly="--colorFromGridOnly"

for whichloss in "${whichlosses[@]}"
do
    python plot_Convection.py --model_file /home/elhamod/projects/deepxdeplayground/AE/saved_models/$wheretosave/model.pt --model_folder $models_path --vlevel $vlevel --vmin $vmin --vmax $vmax --x=$x --whichloss $whichloss --layers "2, 50, 50, 50, 50, 1" --activation tanh --nt 251 --N_f 10000 --beta $beta --xgrid 512 --u0_str "sin(x)" --key_models $key_models --key_modelnames $key_modelnames $colorFromGridOnly $everynth
done

exit;