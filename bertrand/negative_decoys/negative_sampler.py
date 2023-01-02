import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster

from collections import defaultdict
from tqdm.auto import tqdm


def get_peptide_freq_flat(peptide_seqs):
    pep_freq = defaultdict(int)
    for pep_seqs, count in zip(peptide_seqs.index, peptide_seqs.values):
        for pep in pep_seqs.split("|"):
            pep_freq[pep] += count
    pep_freq = pd.Series(pep_freq)
    pep_freq = pep_freq / pep_freq.sum()
    return pep_freq


def assign_clusters(dataset, linkage, thr):
    clusters = fcluster(linkage, thr, criterion="distance")
    dataset.loc[:, "tcr_cluster"] = clusters


class Sampler(object):
    def __init__(self, dataset, desired_counts, seed, verbose=True):
        self.dataset = dataset.copy()
        self.verbose = verbose
        self.seed = seed
        self.desired_counts = desired_counts

    def sample_in_cluster_tcrs(self):
        np.random.seed(self.seed)
        # sampled_negatives = []
        dataset = self.dataset
        iterable = dataset.groupby("tcr_cluster")
        if self.verbose:
            iterable = tqdm(iterable)
        for cluster_id, cluster_df in iterable:
            y_mask = cluster_df.y == 1
            if y_mask.sum() == 0:
                continue
            if y_mask.sum() == len(y_mask):
                continue
            positives_cluster = cluster_df[y_mask]
            negatives_cluster = cluster_df[~y_mask]

            pep_freq = get_peptide_freq_flat(
                positives_cluster.peptide_seq.value_counts()
            )
            sampled_tcrs = negatives_cluster.CDR3b
            sampled_peptides = np.random.choice(
                pep_freq.index, size=len(sampled_tcrs), replace=True, p=pep_freq
            )
            # sampled_df = pd.DataFrame(data={"CDR3b": sampled_tcrs, "y": 0, "tcr_cluster": cluster_id})
            # sampled_df.loc[:, "peptide_seq"] = sampled_peptides
            dataset.loc[negatives_cluster.index, "peptide_seq"] = sampled_peptides
            # sampled_negatives.append(sampled_df)

    def sample_inx_cluster_tcrs(self):
        np.random.seed(self.seed)
        # sampled_negatives = []
        dataset = self.dataset
        iterable = dataset.groupby("tcr_cluster")
        if self.verbose:
            iterable = tqdm(iterable)
        for cluster_id, cluster_df in iterable:
            y_mask = cluster_df.y == 1
            if y_mask.sum() == 0:
                continue
            if y_mask.sum() == len(y_mask):
                continue
            positives_cluster = cluster_df[y_mask]
            negatives_cluster = cluster_df[~y_mask]

            pep_freq = get_peptide_freq_flat(
                positives_cluster.peptide_seq.value_counts()
            )
            sampled_tcrs = negatives_cluster.CDR3b
            c = self.desired_counts.copy()
            c = c[~c.index.isin(pep_freq)]
            #         print(c)
            #         raise Exception()
            #         c = np.log10(c)
            c = c / c.sum()
            sampled_peptide = np.random.choice(c.index, size=None, p=c)
            #         raise Exception()
            # sampled_df = pd.DataFrame(data={"CDR3b": sampled_tcrs, "y": 0, "tcr_cluster": cluster_id})
            # sampled_df.loc[:, "peptide_seq"] = sampled_peptides
            dataset.loc[negatives_cluster.index, "peptide_seq"] = sampled_peptide

    def sample_out_cluster_tcrs(self):
        np.random.seed(self.seed)
        dataset = self.dataset
        negative_clusters = dataset[(dataset.y == 0)].tcr_cluster.unique()
        assigned_clusters_mask = (dataset.y == 0) & (~dataset.peptide_seq.isna())
        used_clusters = dataset[assigned_clusters_mask].tcr_cluster.unique()
        negative_clusters = np.setdiff1d(negative_clusters, used_clusters)
        positive_clusters = dataset[(dataset.y == 1)].tcr_cluster.unique()
        print(len(used_clusters), len(positive_clusters))
        negative_clusters = np.setdiff1d(negative_clusters, positive_clusters)
        if self.verbose:
            print("%d clusters already assigned" % len(used_clusters))
            print(len(used_clusters))

        cluster_gb = dataset.groupby("tcr_cluster")

        negative_assigned_peptide_counts = dataset[
            assigned_clusters_mask
        ].peptide_seq.value_counts()

        peptide_desired_count = self.desired_counts
        peptide_actual_count = negative_assigned_peptide_counts.reindex(
            peptide_desired_count.index
        ).fillna(0)
        count_diff = (peptide_desired_count - peptide_actual_count).astype(int)

        negative_clusters_current = list(negative_clusters)

        for peptide, n_to_sample in zip(count_diff.index, count_diff):
            if n_to_sample <= 0:
                if self.verbose:
                    print(peptide, "too much")
                continue

            negative_clusters_current = self._sample_negative_clusters(
                peptide, n_to_sample, negative_clusters_current, cluster_gb
            )
            if self.verbose:
                print(len(negative_clusters_current), "clusters left")

    def _sample_negative_clusters(
        self, peptide, n_to_sample, negative_clusters, cluster_gb
    ):
        count_sampled = 0
        dataset = self.dataset
        if self.verbose:
            print(peptide, n_to_sample, "to sample")
        while count_sampled < n_to_sample:
            if len(negative_clusters) == 0:
                raise Exception("all sampled!")
            ii = np.random.randint(0, len(negative_clusters))
            negative_clusters_sample = negative_clusters.pop(ii)

            cluster_df = cluster_gb.get_group(negative_clusters_sample)
            assert (cluster_df.y == 0).all()

            dataset.loc[cluster_df.index, "peptide_seq"] = peptide

            # sampled_df = cluster_df[["CDR3b", "y", "peptide_seq", "tcr_cluster"]].copy()
            # sampled_df.peptide_seq = peptide
            # sampled_negatives_pep.append(sampled_df)
            count_sampled += len(cluster_df)
        return negative_clusters

    def get_sampled_dataset(self):
        return self.dataset[
            ~self.dataset.peptide_seq.isna() & (self.dataset.y == 0)
        ].copy()
