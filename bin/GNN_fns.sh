#/bin/bash
# default argparse options for GNN

ROOT="/home/erschultz"
if ! [ -d "${ROOT}" ]
then
    echo "${ROOT} does not exist"
    ROOT=$PWD
    echo "Using ROOT: ${ROOT}"
fi


scratch="${ROOT}/scratch"
deleteRoot='false'

modelType='ContactGNNEnergy'
GNNMode='true'
outputMode='energy_sym_diag'
resumeTraining='false'
pretrainID='none'
id='none'
SEED=42
saveEarlyStop='false'

# preprocessing
yPreprocessing='log_inf'
sweepChoices='none'
yNorm='mean'
yLogTransform='none'
maxDiagonal='none'
keepZeroEdges='false'
KR='false'
rescale='none'
meanFilt='none'
outputPreprocesing='none'
plaidScoreCutoff='none'
maxSample='none'
bondedPath='none'

# architecture
m=1024
messagePassing='weighted_GAT'
useNodeFeatures='false'
k='none'
useEdgeWeights='false'
useEdgeAttr='true'
hiddenSizesList='8-8-8'
EncoderHiddenSizesList='none'
EdgeEncoderHiddenSizesList='none'
updateHiddenSizesList='100-100-16'
transforms='empty'
preTransforms='degree'
topK='none'
sparsifyThresholdUpper='none'
sparsifyThreshold='none'
loss='mse'
lossLambda1=1
lossLambda2=1
lossLambda3=1
lossK=2

act='leaky'
innerAct='leaky'
headAct='leaky'
outAct='leaky'

trainingNorm='none'
headArchitecture='bilinear'
headArchitecture2='none'
headHiddenSizesList='none'
numHeads=8
gated='false'
useSignNet='false'
useSignPlus='false'
inputLtoD='false'
inputLtoDMode='meandist'

# hyperparameters
nEpochs=100
batchSize=1
numWorkers=4
scheduler='MultiStepLR'
milestones='50'
patience=10
gamma=0.1
splitSizes='none'
splitPercents='0.9-0.1-0.0'
randomSplit='true'
lr=1e-3
minlr=1e-6
weightDecay=0.
dropout=0.
wReg='none'
regLambda=1e-1
gradClip='none'
outputClip='none'


verbose='false'
plotPredictions='true'
plot='true'
crop='none'
printParams='true'

train () {
  echo "id=${id}"
  python3 core_test_train.py --data_folder $dirname --root_name $rootName --delete_root $deleteRoot --model_type $modelType --GNN_mode $GNNMode --output_mode $outputMode --m $m --y_preprocessing $yPreprocessing --sweep_choices $sweepChoices --preprocessing_norm $yNorm --log_preprocessing $yLogTransform --kr $KR --rescale $rescale --mean_filt $meanFilt --output_preprocesing $outputPreprocesing --plaid_score_cutoff $plaidScoreCutoff --max_diagonal $maxDiagonal --keep_zero_edges $keepZeroEdges --message_passing $messagePassing --use_node_features $useNodeFeatures --k $k --use_edge_weights $useEdgeWeights --use_edge_attr $useEdgeAttr --hidden_sizes_list $hiddenSizesList  --encoder_hidden_sizes_list $EncoderHiddenSizesList --edge_encoder_hidden_sizes_list $EdgeEncoderHiddenSizesList --update_hidden_sizes_list $updateHiddenSizesList --transforms $transforms --pre_transforms $preTransforms --top_k $topK --sparsify_threshold $sparsifyThreshold --sparsify_threshold_upper $sparsifyThresholdUpper --loss $loss --lambda1 $lossLambda1 --lambda2 $lossLambda2 --loss_k $lossK --act $act --inner_act $innerAct --head_act $headAct --out_act $outAct --training_norm $trainingNorm --head_architecture $headArchitecture --head_architecture_2 $headArchitecture2 --head_hidden_sizes_list $headHiddenSizesList --num_heads $numHeads --gated $gated --use_sign_net $useSignNet --use_sign_plus $useSignPlus --n_epochs $nEpochs --lr $lr --min_lr $minlr --weight_decay $weightDecay --dropout $dropout --w_reg $wReg --reg_lambda $regLambda --grad_clip $gradClip --output_clip $outputClip --batch_size $batchSize --num_workers $numWorkers --scheduler $scheduler --milestones $milestones --patience $patience --gamma $gamma --split_sizes=$splitSizes --split_percents $splitPercents --random_split $randomSplit --verbose $verbose --scratch $scratch --plot_predictions $plotPredictions --plot $plot --crop $crop --print_params $printParams --id $id --resume_training $resumeTraining --pretrain_id $pretrainID --seed $SEED --save_early_stop $saveEarlyStop --input_L_to_D $inputLtoD --input_L_to_D_mode $inputLtoDMode --max_sample $maxSample --bonded_path $bondedPath
}
