#!/bin/bash

train="yes" #"yes"
wheretosave=PINN_Failures_L
models_path=../trajectories/PINN_Failures/saved_models/L_Beta30


#training
weights="--rec_weight 1.0 --wellspacedtrajectory_weight 1.0"
num_of_layers=
learning_rate="--learning_rate 5e-4"
epochs="--epochs 600000" 
patience_scheduler="--patience_scheduler 5000"
cosine="--cosine_Scheduler_patience 2000"
resume= #"--resume"


# plotting
vlevel=30
vmin=-1
vmax=-1
x="-1.2:1.2:25"
whichlosses=("total")
ls=(0.1 0.001 1e-05 1e-06)
betas=(30.0)
key_models="0" 




for beta in "${betas[@]}"
do
    for L in "${ls[@]}"
    do
        everynth="--every_nth 10"

        if [ "$train" == "yes" ]; then
            python ../train_PINN_Failures.py --model_file ../saved_models/$wheretosave/beta$beta\_L$L/model.pt --model_folder $models_path/beta$beta\_L$L $weights $everynth $num_of_layers $learning_rate $resume $epochs $patience_scheduler $cosine
        fi

        everynth="--every_nth 1"
        
        for whichloss in "${whichlosses[@]}"
        do   
            python ../plot_PINN_Failures.py --model_file ../saved_models/$wheretosave/beta$beta\_L$L/model.pt --model_folder $models_path/beta$beta\_L$L --vlevel $vlevel --vmin $vmin --vmax $vmax --x=$x --whichloss $whichloss $everynth --L $L --beta $beta --key_models $key_models --key_modelnames "L=$L"
        done
    done
done


exit;