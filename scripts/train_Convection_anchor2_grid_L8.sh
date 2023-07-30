#!/bin/bash

#SBATCH --account=imageomics-biosci  #imageomics-biosci  #mabrownlab   #ml4science
#SBATCH --partition=dgx_normal_q
#SBATCH --time=0-4:00:00 
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH -o ../SLURM/slurm-%j.out

# setup
module load Anaconda3/2020.11
source activate landscapesvisenv



train="yes" #"yes" "no"
num_of_layers=3
epochs=80000
wheretosave=Convection_anchor2_grid_L8
models_path=../trajectories/MTL_same_initialization/Beta30/CW




#training
weights="--lastzero_weight 10000 --gridscaling_weight 1.0 --latentfactor 8.0 --grid_step 0.2"
beta=30.0
everynth="--every_nth 1"

if [ "$train" == "yes" ]; then
    python ../train_Convection.py --model_file ../saved_models/$wheretosave/model.pt --model_folder $models_path $weights $everynth --epochs $epochs $resume
fi





# plotting
key_models="0" 
key_modelnames="CW"
vlevel=30
vmin=-1
vmax=-1
x="-1.2:1.2:25"
whichlosses=("residual")
colorFromGridOnly="--colorFromGridOnly"
density="--density_type CKA --density_vmin 150 --density_vmax 550"

for whichloss in "${whichlosses[@]}"
do
    python ../plot_Convection.py --model_file ../saved_models/$wheretosave/model.pt --model_folder $models_path --vlevel $vlevel --vmin $vmin --vmax $vmax --x=$x --whichloss $whichloss --layers "2, 50, 50, 50, 50, 1" --activation tanh --nt 251 --N_f 10000 --beta $beta --xgrid 512 --u0_str "sin(x)" --key_models $key_models --key_modelnames $key_modelnames $colorFromGridOnly $density $everynth
done

exit;