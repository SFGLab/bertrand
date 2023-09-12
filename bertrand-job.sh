#!/bin/bash
#SBATCH --mail-type=ALL          # Powiadomienia mailowe. Opcje: NONE, BEGIN, END, FAIL, ALL
#SBATCH --mail-user=jacenko.vlad@gmail.com     # adres e-mail
#SBATCH --ntasks=16                  
#SBATCH --mem=128gb
#SBATCH --gpus=a100:2
#SBATCH --time=72:00:00               # maksymalny limit czasu DD-HH:MM:SS
#SBATCH --partition=long

pwd; hostname; date

source /home2/sfglab/yvladyslav/anaconda3/etc/profile.d/conda.sh
cd bertrand
conda activate bertrand
./analysis.sh "/home2/sfglab/yvladyslav/protbert/bertrand_results" 16
 
date
