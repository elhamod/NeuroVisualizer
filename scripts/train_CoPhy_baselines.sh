
#!/bin/bash

export OMP_NUM_THREADS=1

#variables
prefix="--prefix state_"
everynth=1
modelPath="../trajectories/CoPhy/CoPhy_landscapes/states"
wheretosave=CoPhy_baselines
DNN_type="PGNN_"
dataPath="../data/CoPhyData/"
vlevel=30
vmin=-1
vmax=-1
x="-1.2:1.2:25"
whichlosses=("phy")
loss_names=( "total")
key_models="0" 
key_modelnames="CoPhy"

for whichloss in "${whichlosses[@]}"
do
    for loss_name in "${loss_names[@]}"
    do
        python ../plot_CoPhy_baselines.py --model_file ../saved_models/$wheretosave --model_folder $modelPath --vlevel $vlevel --vmin $vmin --vmax $vmax --x=$x --whichloss $whichloss --DNN_type $DNN_type --dataPath $dataPath --loss_name $loss_name --every_nth $everynth $prefix --key_models $key_models --key_modelnames $key_modelnames
    done
done




exit;