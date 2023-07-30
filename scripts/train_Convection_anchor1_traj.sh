#!/bin/bash

#SBATCH --account=imageomics-biosci  #imageomics-biosci  #mabrownlab   #ml4science
#SBATCH --partition=dgx_normal_q
#SBATCH --time=0-4:00:00 
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH -o ../SLURM/slurm-%j.out

#TODO: remove the sbatch stuff

# setup
module load Anaconda3/2020.11
source activate landscapesvisenv

train="yes" #"yes" "no"
resume= #"--resume"
num_of_layers=3
epochs=80000
models_path=../trajectories/MTL_same_initialization/Beta30/CW
wheretosave=Convection_anchor1_traj


#training
weights="--polars_weight 10000 --equidistant_weight 10000.0"
beta=30.0
everynth="--every_nth 1"

if [ "$train" == "yes" ]; then
    python ../train_Convection.py --model_file ../saved_models/$wheretosave/model.pt --model_folder $models_path $weights $everynth --epochs $epochs $resume
fi




#plotting
key_models="0" 
key_modelnames="CW"
vlevel=30
vmin=-1
vmax=-1
x="-1.2:1.2:25"
whichlosses=("residual")
colorFromGridOnly="--colorFromGridOnly"

for whichloss in "${whichlosses[@]}"
do
    python ../plot_Convection.py --model_file ../saved_models/$wheretosave/model.pt --model_folder $models_path --vlevel $vlevel --vmin $vmin --vmax $vmax --x=$x --whichloss $whichloss --layers "2, 50, 50, 50, 50, 1" --activation tanh --nt 251 --N_f 10000 --beta $beta --xgrid 512 --u0_str "sin(x)" --key_models $key_models --key_modelnames $key_modelnames $everynth $colorFromGridOnly
done

exit;