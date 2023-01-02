import argparse
import glob
import json
import logging
import multiprocessing as mp
import os
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_validate,
    cross_val_predict,
    StratifiedKFold,
)
from tqdm.auto import tqdm
from bertrand.model.tokenization import AA_dict


def parse_args() -> argparse.Namespace:
    """
    argument parser for the script
    :return: args
    """
    parser = argparse.ArgumentParser(
        "This script performs outliers filtering.\n"
        "using an iterative easy negatives removal algorithm.\n"
        "N=`step` observations that are easily classified as negatives from TCR sequences.\n"
        "alone are removed at each iteration.\n"
        "The process is repeated until AUROC=`auroc` is reached.\n"
        "This computation is CPU-intensive."
    )
    parser.add_argument(
        "--tcrs",
        type=str,
        required=True,
        help="Path to positive&reference TCR dataset. (Result of `basic_filtering.py`",
    )
    parser.add_argument(
        "--out-tcrs",
        type=str,
        required=True,
        help="Path to output positive&reference TCR dataset with outliers removed",
    )
    parser.add_argument(
        "--out-results",
        type=str,
        required=True,
        help="Path to outliers filtering results",
    )
    parser.add_argument(
        "--predict-k-folds",
        type=int,
        default=5,
        help="Number of folds for `StratifiedKFold` in `cross-val-predict`",
    )
    parser.add_argument(
        "--predict-n-repeats",
        type=int,
        default=10,
        help="Number of repeats of `cross-val-predict`",
    )
    parser.add_argument(
        "--score-k-folds",
        type=int,
        default=5,
        help="Number of folds of `cross-val-score`",
    )
    parser.add_argument(
        "--score-n-repeats",
        type=int,
        default=5,
        help="Number of repeats for  of `cross-val-score`",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=2000,
        help="Number of observations removed at each iteration",
    )
    parser.add_argument(
        "--maxiter", type=int, default=200, help="Maximum number of iterations"
    )
    parser.add_argument(
        "--cpu", type=int, default=-1, help="Total number of CPUs to use"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--cache-dir", type=str, default=None, help="Results cache folder"
    )
    parser.add_argument("--auroc", type=float, default=0.5, help="Target AUROC")
    return parser.parse_args()


def get_features(dataset: pd.DataFrame) -> np.ndarray:
    """
    Generate one-hot representation for TCRs
    :param dataset: TCR dataset - pd.DataFrame with `CDR3b` column
    :return: flattened one-hot representation of CDR3b - np.ndarray of shape=(len(dataset, 400))
    """
    logging.info("Generating one-hot features")
    X = np.zeros(shape=(len(dataset), 20, 20), dtype="float32")
    for i, cdr3b in enumerate(tqdm(dataset.CDR3b)):
        for j, aa in enumerate(cdr3b):
            k = AA_dict[aa] - 1
            X[i, j, k] = 1
    X = X.reshape(len(X), -1)
    return X


def distribute_jobs(n_cpus: int, n_splits: int) -> Tuple[int, int]:
    """
    Distributes CPUs between `ExtraTreesClassifier` and `cross_validate`/`cross_val_predict`.
    1. cv_jobs (n_jobs in `cross_validate`),
    2. clf_jobs (n_jobs in `ExtremeTreesClassifier`),
    The CPUs are distributed so that cv_jobs * n_splits * clf_jobs <= n_cpus
    If n_cpus <= n_splits, only cross-validation in parallelized (i.e. clf_jobs==1)
    :param n_cpus: total number of CPUs (-1 for all)
    :param n_splits: number of cross-validation iterations
    :return: number of jobs for classifier, number of jobs for cross-validation
    """
    if n_cpus == -1:
        n_cpus = mp.cpu_count()
    cv_jobs = min(n_cpus, n_splits)
    clf_jobs = max(
        1, n_cpus // n_splits
    )  # if n_jobs > n_cpus, then n_cpus / n_jobs < 1
    return clf_jobs, cv_jobs


def _cross_val_score(
    X: np.ndarray, y: np.ndarray, n_splits: int = 5, n_repeats: int = 5, n_cpus: int = 1
) -> Dict:
    """
    Compute average cross-validated AUROC for the dataset of positives and reference TCRs.
    :param X: representation of the dataset - np.ndarray of shape (n_samples, n_features)
    :param y: target variable (1 - binder TCR, 0 - reference TCR)
    :param n_splits: number of splits in `RepeatedStratifiedKFold`
    :param n_repeats: number of repeats in `RepeatedStratifiedKFold`
    :param n_cpus: number of CPUs available for `cross_validate`
    :return: result of cross validation - dictionary returned by `cross_validate`
    """
    clf_jobs, cv_jobs = distribute_jobs(n_cpus, n_splits * n_repeats)
    res = cross_validate(
        ExtraTreesClassifier(n_jobs=clf_jobs),
        X,
        y,
        scoring="roc_auc",
        cv=RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=42
        ),
        verbose=3,
        n_jobs=cv_jobs,
        return_estimator=False,
    )
    return res


def _cross_val_predict(
    X: np.ndarray, y: np.ndarray, n_splits: int = 5, n_repeats: int = 10, n_cpus=1
) -> np.ndarray:
    """
    Compute average cross-validated predictions for the dataset of positives and reference TCRs.
    :param X: representation of the dataset - np.ndarray of shape (n_samples, n_features)
    :param y: target variable (1 - binder TCR, 0 - reference TCR)
    :param n_splits: number of splits for `StratifiedKFold`
    :param n_repeats: number of repetitions of `cross_val_predict`
    :param n_cpus: number of CPUs available for `cross_val_predict`
    :return: average out-of-fold predictions for the whole dataset
    """
    y_pred_list = []
    clf_jobs, cv_jobs = distribute_jobs(n_cpus, n_splits)
    for seed in range(42, 42 + n_repeats):
        y_pred = cross_val_predict(
            ExtraTreesClassifier(n_jobs=clf_jobs),
            X,
            y,
            cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed),
            method="predict_proba",
            verbose=3,
            n_jobs=cv_jobs,
        )
        y_pred_list.append(y_pred)

    y_pred = np.vstack([y[:, 1] for y in y_pred_list]).mean(axis=0)
    return y_pred


def check_args(args: argparse.Namespace) -> None:
    """
    Checks for  stored arguments (`args.json` in cache dir),
    raises an Exception if stored arguments don't match the call arguments.
    :param args: arguments of the script - Namespace
    """
    cache_dir = args.cache_dir
    args_fn = os.path.join(cache_dir, "args.json")
    call_args = dict(vars(args))
    call_args.pop("cpu")

    if os.path.isfile(args_fn):
        with open(args_fn, "r") as fp:
            stored_args = json.load(fp)

        if stored_args != call_args:
            logging.error(
                f"Stored arguments (args.json) in {cache_dir} do not match the arguments passed to "
                f"the process. Review the cache and/or remove the stored arguments is you know what you "
                f"are doing"
            )
            logging.error("Stored arguments:")
            logging.error(stored_args)
            logging.error("Call arguments:")
            logging.error(call_args)
            raise Exception("Stored args do not match call args")

    else:
        with open(args_fn, "w") as fp:
            json.dump(call_args, fp, indent=4)


def outliers_filtering(dataset: pd.DataFrame, args: argparse.Namespace) -> List[Dict]:
    """
    Performs outliers filtering. The algorithm iteratively
    checks the AUROC of positive vs reference TCRs classification.
    N=`step` observations that are easily classified as negatives
    from TCR sequences alone are removed at each iteration.
    Easy negatives are determined using out-of-fold predictions.
    The process is repeated until AUROC=`auroc` is reached.
    :param dataset: dataset of positive&reference TCRs
    :param args: script arguments, see `parse_args`
    :return: list of dictionaries with outliers filtering results
    """
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        check_args(args)

    clf_jobs, cv_jobs = distribute_jobs(
        args.cpu, args.score_k_folds * args.score_n_repeats
    )
    logging.info(f"cross-val-score: {cv_jobs} CPUs for cv, {clf_jobs} CPUs for XT")
    clf_jobs, cv_jobs = distribute_jobs(args.cpu, args.predict_k_folds)
    logging.info(f"cross-val-predict: {cv_jobs} CPUs for cv, {clf_jobs} CPUs for XT")

    X = get_features(dataset)
    y = dataset.y.values

    np.random.seed(args.seed)
    results = []
    mask = np.ones(len(X), dtype=bool)

    def load_cache(iteration):
        if args.cache_dir:
            files = glob.glob(os.path.join(args.cache_dir, f"iteration={iteration}_*"))
            if files:
                r = pd.read_pickle(files[0])
                return r
        return None

    logging.info(
        f"Outliers filtering of {len(X)} observations "
        f"for max. {args.maxiter} iterations, "
        f"until AUROC={args.auroc:.2f} is reached"
    )

    for iteration in range(len(results), args.maxiter):
        result = load_cache(iteration)
        if result:
            score = result["score"]
            mask = result["mask"]
        else:
            y_pred = _cross_val_predict(
                X[mask], y[mask], args.predict_k_folds, args.predict_n_repeats, args.cpu
            )
            thr = np.sort(y_pred[y[mask] == 0])[args.step]
            easy_neg_mask = (y[mask] == 0) & (y_pred <= thr)

            mask[mask] = mask[mask] & ~easy_neg_mask
            nneg = (y[mask] == 0).sum()
            npos = (y[mask] == 1).sum()

            cv_res = _cross_val_score(
                X[mask], y[mask], args.score_k_folds, args.score_n_repeats, args.cpu
            )

            score = cv_res["test_score"].mean()
            result = dict(
                thr=thr,
                score=score,
                nneg=nneg,
                npos=npos,
                mask=mask.copy(),
                y_pred=y_pred,
            )
            if args.cache_dir:
                fn = os.path.join(
                    args.cache_dir,
                    f"iteration={iteration}_N={mask.sum()}_AUROC={score:.2f}.pkl",
                )
                pd.to_pickle(result, fn)

        logging.info(f"Iter. {iteration}: {mask.sum()} observations, {score:.2f} AUROC")
        results.append(result)

        # Stopping criterion
        if score - args.auroc < 0.001:
            break

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    args = parse_args()

    # Read positive&reference TCRs dataset
    dataset = pd.read_csv(args.tcrs)

    if os.path.isfile(args.out_results):
        # Load the results file if it already exists
        results = pd.read_pickle(args.out_results)
    else:
        # Compute outliers filtering results
        results = outliers_filtering(dataset, args)
        # Save outliers filtering results
        pd.to_pickle(results, args.out_results)

    # The last result contains the mask of observations that remained -
    # positive and reference TCRs that aren't easily distinguishable
    mask = results[-1]["mask"]

    # Apply the mask
    dataset_filtered = dataset[mask]

    # Save the filtered dataset
    dataset_filtered.to_csv(args.out_tcrs, index=False)
