set -x
DIR=$1

mkdir -p "$DIR"

python -m bertrand.pretraining.peptide_tcr_repertoire \
  --input-peptides=data/pos.csv.gz \
   --input-tcrs=data/synthetic_tcrs_11M.csv.gz \
   --out-dir="$DIR"

python -m bertrand.pretraining.train_mlm \
  --train="$DIR"/mlm_train.csv.gz \
  --val="$DIR"/mlm_val.csv.gz \
  --out-dir="$DIR"

python -m bertrand.pretraining.evaluate_mlm \
  --input-dir="$DIR"/checkpoints \
  --out-dir="$DIR"/model