set -x
DATA_DIR=$1
MODEL_DIR=$2
OUT_DIR=$3

mkdir -p "$OUT_DIR"

python -m bertrand.training.train \
  --input-dir=$DATA_DIR \
  --model-ckpt=$MODEL_DIR \
  --output-dir=$OUT_DIR \
  --n-splits=21

python -m bertrand.training.evaluate \
  --datasets-dir=$DATA_DIR \
  --results-dir=$OUT_DIR \
  --out=$OUT_DIR/results.csv
