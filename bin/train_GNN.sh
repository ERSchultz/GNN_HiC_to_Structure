#! /bin/bash

cd ~/GNN_HiC_to_Structure
source activate python3.9_pytorch1.9
source bin/GNN_fns.sh # $ROOT defined here

rootName='GNNModel1' # change this to run multiple bash files at once
dirname="${ROOT}/dataset_synthetic"
m=512
preTransforms='ContactDistance-MeanContactDistance-AdjPCs_10'
hiddenSizesList='16-16-16-16'
updateHiddenSizesList='1000-1000-1000-1000-128'
outputPreprocesing='none'
headArchitecture='bilinear'
headArchitecture2="fc-fill_${m}"
headHiddenSizesList='1000-1000-1000-1000-1000-1000'
rescale=2
yNorm='mean_fill'
k=10
useSignPlus='true'
batchSize=1
nEpochs=5 # TODO
milestones='3' # TODO
loss='mse_log'

id=2 # need to manually specify modelID
for lr in 1e-4
do
  train
  id=$(( $id + 1 ))
done
python3 ./scripts/clean_directories.py --data_folder $dirname --GNN_file_name $rootName --scratch $scratch
