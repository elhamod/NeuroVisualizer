#!/bin/bash

#SBATCH --account=imageomics-biosci #imageomics-biosci  #mabrownlab   #ml4science
#SBATCH --partition=dgx_normal_q
#SBATCH --time=0-4:00:00 
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH -o /home/elhamod/projects/AE/SLURM/slurm-%j.out


module load Anaconda3/2020.11
source activate landscapesvisenv

cd /home/elhamod/projects/AE/


#variables

prefix="--prefix state_"
everynth=1
modelPath="/home/elhamod/projects/AE/trajectories/CoPhy/CoPhy_landscapes/states"
wheretosave=CoPhy_baselines
DNN_type="PGNN_"
dataPath="/home/elhamod/projects/AE/data/CoPhyData/"

vlevel=30
vmin=-1
vmax=-1

x="-1.2:1.2:25"

whichlosses=("phy")
loss_names=( "total")

key_models="0" 
key_modelnames="CoPhy"


export OMP_NUM_THREADS=1

for whichloss in "${whichlosses[@]}"
do
    for loss_name in "${loss_names[@]}"
    do
        python plot_CoPhy_baselines.py --model_file /home/elhamod/projects/AE/saved_models/$wheretosave --model_folder $modelPath --vlevel $vlevel --vmin $vmin --vmax $vmax --x=$x --whichloss $whichloss --DNN_type $DNN_type --dataPath $dataPath --loss_name $loss_name --every_nth $everynth $prefix --key_models $key_models --key_modelnames $key_modelnames
    done
done




exit;