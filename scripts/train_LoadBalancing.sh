#!/bin/bash

#SBATCH --account=ml4science #imageomics-biosci  #mabrownlab   #ml4science
#SBATCH --partition=dgx_normal_q
#SBATCH --time=0-4:00:00 
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH -o ../SLURM/slurm-%j.out

# setup
module load Anaconda3/2020.11
source activate landscapesvisenv



train="yes" #"yes" "no"
wheretosave=LoadBalancing
models_path=../trajectories/LoadBalancing_same_initialization/Beta10




#training
weights="--rec_weight 10000.0"
epochs="--epochs 600000"
num_of_layers=
learning_rate="--learning_rate 5e-4"
patience_scheduler="--patience_scheduler 400000"
resume= # "--resume"
cosine="--cosine_Scheduler_patience 2000"

if [ "$train" == "yes" ]; then
    python ../train_Convection.py --model_file ../saved_models/$wheretosave/model.pt --model_folder $models_path $weights $everynth $num_of_layers $learning_rate $resume $epochs $patience_scheduler $cosine
fi



# plotting
everynth="--every_nth 1"
key_models="0 301 602 903 1204 1505" 
key_modelnames="GradNorm RLW DWA EW LRannealing CW"
beta=10.0
vlevel=30
vmin=-1
vmax=-1
x="-1.2:1.2:25"
whichlosses=("residual" "test_mse")
colorFromGridOnly="--colorFromGridOnly"

for whichloss in "${whichlosses[@]}"
do
    python ../plot_Convection.py --model_file ../saved_models/$wheretosave/model.pt --model_folder $models_path --vlevel $vlevel --vmin $vmin --vmax $vmax --x=$x --whichloss $whichloss --layers "2, 50, 50, 50, 50, 1" --activation tanh --nt 251 --N_f 10000 --beta $beta --xgrid 512 --u0_str "sin(x)" --key_models $key_models --key_modelnames $key_modelnames $everynth $colorFromGridOnly
done



exit;