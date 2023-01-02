# Data curation

## Sources

The data comes mainly from public peptide:TCR databases (VDJdb, McPAS, TBAdb),
as well as publications (Zhang et al. 2018, 10x Genomics, Pogorelyy et al. 2018, and many more). 
The full list of data sources is available in the publication. 

## Prepare datasets from raw data

Run all prepare_*.ipynb notebooks to prepare uniform datasets from the original published data
This will process all datasets in the `data/original` folder and save those to `data/processed`
## data-curation.ipynb

This notebook filters, combines and removes duplicates all previously processed datasets. 