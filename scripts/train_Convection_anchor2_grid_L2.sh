#!/bin/bash

train="yes" #"yes" "no"
num_of_layers=3
epochs=80000
wheretosave=Convection_anchor2_grid_L2
models_path=../trajectories/LoadBalancing_same_initialization/Beta30/CW



#training
weights="--lastzero_weight 100 --gridscaling_weight 1.0 --d_max_latent 2.0 --grid_step 0.2"
beta=30.0
everynth="--every_nth 1"

if [ "$train" == "yes" ]; then
    python ../train_Convection.py --model_file ../saved_models/$wheretosave/model.pt --model_folder $models_path $weights $everynth --epochs $epochs $resume
fi




#plotting
key_models="0" 
key_modelnames="CW" 
density="--density_type CKA --density_vmin 150 --density_vmax 550"
vlevel=30
vmin=-1
vmax=-1
x="-1.2:1.2:25"
whichlosses=("residual")

for whichloss in "${whichlosses[@]}"
do
    python ../plot_Convection.py --model_file ../saved_models/$wheretosave/model.pt --model_folder $models_path --vlevel $vlevel --vmin $vmin --vmax $vmax --x=$x --whichloss $whichloss --layers "2, 50, 50, 50, 50, 1" --activation tanh --nt 251 --N_f 10000 --beta $beta --xgrid 512 --u0_str "sin(x)" --key_models $key_models --key_modelnames $key_modelnames $density $everynth
done

exit;