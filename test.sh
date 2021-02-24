#!/bin/bash
  
#PBS -l ncpus=48
##PBS -l ngpus=2
##PBS -q gpuvolta
#PBS -q hugemem
##PBS -q express
#PBS -P yf70
#PBS -l wd
#PBS -l walltime=24:00:00
#PBS -l mem=1000GB
#PBS -l jobfs=1GB

##PBS -m abe
#PBS -M tham.nguyen@mq.edu.au

echo $PWD 
echo $WD
source /scratch/yf70/tn9582/miniconda3/bin/activate
conda activate spacy3
cd /scratch/yf70/tn9582/spacy_mi
echo $PWD 
echo $WD
