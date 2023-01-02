library(immuneSIM)
library(parallel)
library(pbmcapply)

insertion_and_deletion_lengths_df <- load_insdel_data()

starts <- 1:11260
fx <- function(n)
{
  sim_repertoire_TRB <-immuneSIM(
    number_of_seqs = 1000,
    species = "hs",
    receptor = "tr",
    chain = "b",
    insertions_and_deletion_lengths = insertions_and_deletion_lengths_df,
    verbose= F)
  return(sim_repertoire_TRB)
}


numCores <- detectCores() - 1
numCores
results <- pbmclapply(starts, fx, mc.cores = numCores)  

dt <- do.call("rbind", results)
rownames(dt) <- 1:nrow(dt)
write.csv(dt, 'data/simulated_cdr3b_11M.csv')

