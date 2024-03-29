#!/bin/bash

everynth=1

train="yes" #"yes" "no"
prefix="--prefix model_epoch_" 
modelPath="../trajectories/models_mlp_cifar10"
wheretosave=MLP_grid_polar_every_$everynth
dataset_name="cifar10"
model_name="mlp"
key_modelnames="MLP"

#training
weights="--anchor_weight 100 --gridscaling_weight 1.0 --d_max_latent 0.7 --grid_step 0.2"
resume= #"--resume" 

if [ "$train" == "yes" ]; then
    python ../train_CNN_MLP.py --model_file ../saved_models/$wheretosave/model.pt --model_folder $modelPath $weights $prefix --every_nth $everynth $resume
fi


# plotting
vlevel=30
vmin=1
vmax=4
x="-1.2:1.2:25"
whichlosses=("mse")
loss_names=("test_loss" "train_loss")
key_models="0" 

for whichloss in "${whichlosses[@]}"
do
    for loss_name in "${loss_names[@]}"
    do
        python ../plot_CNN_MLP.py --model_file ../saved_models/$wheretosave/model.pt --dataset_name $dataset_name --model_name $model_name --model_folder $modelPath --vlevel $vlevel --vmin $vmin --vmax $vmax --x=$x --whichloss $whichloss --loss_name $loss_name --every_nth $everynth $prefix --key_models $key_models --key_modelnames $key_modelnames
    done
done


exit;