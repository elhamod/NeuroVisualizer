#!/bin/bash

train="yes" #"yes" "no"
DNN_type=PGNN_ 
modelPath="../trajectories/CoPhy/CoPhy_landscapes/states"
wheretosave=CoPhy_every$everynth
dataPath="../data/CoPhyData/"
prefix="--prefix state_"





#training
weights=""
everynth=1

if [ "$train" == "yes" ]; then
    python ../train_CoPhy.py --model_file ../saved_models/$wheretosave/model.pt --model_folder $modelPath $weights $prefix --every_nth $everynth
fi





# plotting
vlevel=30
vmin=-1
vmax=-1
x="-1.2:1.2:25"
# x="-1:-0.1:25" # zoomed
whichlosses=("phy")
loss_names=("total")
key_models="0" 
key_modelnames="CoPhy"

for whichloss in "${whichlosses[@]}"
do
    for loss_name in "${loss_names[@]}"
    do
        python ../plot_CoPhy.py --model_file ../saved_models/$wheretosave/model.pt --model_folder $modelPath --vlevel $vlevel --vmin $vmin --vmax $vmax --x=$x --whichloss $whichloss --DNN_type $DNN_type --dataPath $dataPath --loss_name $loss_name --every_nth $everynth $prefix --key_models $key_models --key_modelnames $key_modelnames
    done
done

exit;