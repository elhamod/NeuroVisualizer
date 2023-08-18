#!/bin/bash

train="yes" #"yes" "no"
wheretosave=PINN_NTK
models_path=../trajectories/PINN_NTK 



#training
weights="--rec_weight 1.0 --anchor_weight 10000.0 --anchor_mode circle"
everynth="--every_nth 10"
num_of_layers=
layers="--layers_AE 1000 500 100"
learning_rate="--learning_rate 1e-5"
epochs="--epochs 100000"
patience_scheduler="--patience_scheduler 100000"
resume= #"--resume"
cosine="--cosine_Scheduler_patience 600"

if [ "$train" == "yes" ]; then
    python ../train_Convection.py --model_file ../saved_models/$wheretosave/model.pt --model_folder $models_path $weights $everynth $num_of_layers $learning_rate $resume $layers $epochs $patience_scheduler $cosine
fi





#plotting
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
    python ../plot_PINN_NTK.py --model_file ../saved_models/$wheretosave/model.pt --model_folder $models_path --vlevel $vlevel --vmin $vmin --vmax $vmax --x=$x --whichloss $whichloss --key_models $key_models $layers --key_modelnames $key_modelnames $everynth --density_type CKA 
done



exit;