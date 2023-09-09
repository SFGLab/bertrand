set -ex
DIR=$1
CPU=$2

mkdir -p "$DIR"

python -m bertrand.negative_decoys.basic_filtering \
  --binders=data/phla_tcr_unique.csv.gz \
  --reference-dir=data/oakes_tcr_repertoires \
  --out-distances=$DIR/reference_to_positive_distance.npz \
  --out-tcrs=$DIR/positives_and_reference_negatives.csv.gz \
  --cpu=$CPU

python -m bertrand.negative_decoys.outliers_filtering \
  --tcrs=$DIR/positives_and_reference_negatives.csv.gz \
  --out-tcrs=$DIR/positives_and_reference_negatives_filtered.csv.gz \
  --out-results=$DIR/outliers_filtering_results.pkl \
  --cpu=$CPU \
  --cache-dir=$DIR/outliers_filtering_cache \
  --predict-k-folds 5 \
  --predict-n-repeats 10 \
  --score-k-folds 5 \
  --score-n-repeats 5

python -m bertrand.negative_decoys.compute_distance \
  --tcrs=$DIR/positives_and_reference_negatives_filtered.csv.gz \
  --out-distances=$DIR/pairwise_distances.npz \
  --cpu=$CPU

python -m bertrand.negative_decoys.tcr_clustering \
  --distances=$DIR/pairwise_distances.npz \
  --out-linkage=$DIR/linkage.npz

python -m bertrand.negative_decoys.assign_clusters_and_filter \
  --tcrs=$DIR/positives_and_reference_negatives_filtered.csv.gz \
  --linkage=$DIR/linkage.npz \
  --out-tcrs=$DIR/positives_and_reference_negatives_filtered_clustered.csv.gz

python -m bertrand.negative_decoys.negative_decoys_generation \
  --tcrs=$DIR/positives_and_reference_negatives_filtered_clustered.csv.gz \
  --binders=data/phla_tcr_unique.csv.gz \
  --out-dir=$DIR/datasets \
  --ratio=3
