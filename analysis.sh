set -x
DIR=$1
CPU=$2

# First run MLM pre-training
# This step is faster with a GPU
# bash pretraining.sh "$DIR"/pretraining

# Then generate negative decoys
# This step is very CPU and RAM intensive
bash negative_decoys.sh "$DIR"/negative_decoys "$CPU"

# Finally perform supervised training and evaluate the model
# This step is faster with a GPU
bash train_and_evaluate.sh "$DIR"/negative_decoys/datasets "$DIR"/pretraining/model "$DIR"/training