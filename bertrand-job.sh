#!/bin/bash
#SBATCH --mail-type=ALL          # Powiadomienia mailowe. Opcje: NONE, BEGIN, END, FAIL, ALL
#SBATCH --mail-user=jacenko.vlad@gmail.com     # adres e-mail
#SBATCH --ntasks=4                  
#SBATCH --mem=32gb
#SBATCH --gpus=a100:1
#SBATCH --time=12:00:00               # maksymalny limit czasu DD-HH:MM:SS
#SBATCH --partition=short

pwd; hostname; date

source /home2/sfglab/yvladyslav/anaconda3/etc/profile.d/conda.sh
cd bertrand
conda activate bertrand
./analysis.sh "/home2/sfglab/yvladyslav/distilbert-no-pretrain/bertrand_results" 16
 
date
