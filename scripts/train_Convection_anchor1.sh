#!/bin/bash

train="yes" #"yes" "no"
models_path=../trajectories/LoadBalancing_same_initialization/Beta30/CW
wheretosave=Convection_anchor1
num_of_layers=3
epochs=80000


#training
weights="--polars_weight 100"
beta=30.0
everynth="--every_nth 1"

if [ "$train" == "yes" ]; then
    python ../train_Convection.py --model_file ../saved_models/$wheretosave/model.pt --model_folder $models_path $weights $everynth --epochs $epochs
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
    python ../plot_Convection.py --model_file ../saved_models/$wheretosave/model.pt --model_folder $models_path --vlevel $vlevel --vmin $vmin --vmax $vmax --x=$x --whichloss $whichloss --layers "2, 50, 50, 50, 50, 1" --activation tanh --nt 251 --N_f 10000 --beta $beta --xgrid 512 --u0_str "sin(x)" --key_models $key_models --key_modelnames $key_modelnames $colorFromGridOnly $everynth
done

exit;