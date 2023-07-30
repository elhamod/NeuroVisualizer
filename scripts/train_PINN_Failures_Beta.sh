#!/bin/bash

#SBATCH --account=ml4science #imageomics-biosci  #mabrownlab   #ml4science
#SBATCH --partition=dgx_normal_q
#SBATCH --time=1-0:00:00 
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH -o ../SLURM/slurm-%j.out

# setup
module load Anaconda3/2020.11
source activate landscapesvisenv



train="yes" #"yes"
wheretosave=PINN_Failures_Beta
models_path=../trajectories/PINN_Failures/saved_models/Beta_L1




#training
weights="--rec_weight 1.0 --wellspacedtrajectory_weight 1.0"
num_of_layers=
learning_rate="--learning_rate 5e-4"
epochs="--epochs 200000" #"
patience_scheduler="--patience_scheduler 5000"
cosine="--cosine_Scheduler_patience 2000"
resume= #"--resume"




#plotting
vlevel=30
vmin=-1
vmax=-1
x="-1.2:1.2:25"
whichlosses=("total")
betas=(1.0 10.0 20.0 30.0)
ls=(1.0)
key_models="0" 





for beta in "${betas[@]}"
do
    for L in "${ls[@]}"
    do
        everynth="--every_nth 10"

        if [ "$train" == "yes" ]; then
            python train_PINN_Failures.py --model_file ../saved_models/$wheretosave/beta$beta\_L$L/model.pt --model_folder $models_path/beta$beta\_L$L $weights $everynth $num_of_layers $learning_rate $resume $epochs $patience_scheduler $cosine
        fi

        everynth="--every_nth 1"
        
        for whichloss in "${whichlosses[@]}"
        do
            python ../plot_PINN_Failures.py --model_file ../saved_models/$wheretosave/beta$beta\_L$L/model.pt --model_folder $models_path/beta$beta\_L$L --vlevel $vlevel --vmin $vmin --vmax $vmax --x=$x --whichloss $whichloss $everynth --L $L --beta $beta --key_models $key_models --key_modelnames "beta=$beta"
        done
    done
done


exit;