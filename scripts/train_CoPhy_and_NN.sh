#!/bin/bash

#SBATCH --account=imageomics-biosci #imageomics-biosci  #mabrownlab   #ml4science
#SBATCH --partition=dgx_normal_q
#SBATCH --time=0-4:00:00 
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH -o /home/elhamod/projects/deepxdeplayground/SLURM/slurm-%j.out


module load Anaconda3/2020.11
source activate landscapesvisenv

cd /home/elhamod/projects/deepxdeplayground/AE/

train="no" #"yes"

#variables

weights=""
prefix="--prefix state_"


# 
dataPath="/home/elhamod/projects/deepxdeplayground/AE/CoPhyData/"

everynth=1


# DNN_type=NN
# modelPath="/home/elhamod/projects/cophy/models/NN_landscapes_-0x76bf66285ad523a5/states"
# wheretosave=NN_every$everynth

DNN_type=PGNN_ 
modelPath="/home/elhamod/projects/cophy/models"
wheretosave=CoPhy_and_NN_every$everynth
resume="--resume"


if [ "$train" == "yes" ]; then
    python train_CoPhy.py --model_file /home/elhamod/projects/deepxdeplayground/AE/saved_models/$wheretosave/model.pt --model_folder $modelPath $weights $prefix --every_nth $everynth $resume
fi



vlevel=30
vmin=-1
vmax=-1

x="-1.2:1.2:25"

whichlosses=("mse" "e" "phy" )
loss_names=("test_loss" "total")

key_models="0 503" 
key_modelnames="CoPhy NN"



for whichloss in "${whichlosses[@]}"
do
    for loss_name in "${loss_names[@]}"
    do
        python plot_CoPhy.py --model_file /home/elhamod/projects/deepxdeplayground/AE/saved_models/$wheretosave/model.pt --model_folder $modelPath --vlevel $vlevel --vmin $vmin --vmax $vmax --x=$x --whichloss $whichloss --DNN_type $DNN_type --dataPath $dataPath --loss_name $loss_name --every_nth $everynth $prefix --key_models $key_models --key_modelnames $key_modelnames
    done
done


exit;