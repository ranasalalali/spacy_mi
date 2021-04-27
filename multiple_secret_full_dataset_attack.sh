#!/bin/bash

 
#$1 - N passwords
#$2 - r_space
#$3 - knowledge
#$4 - epoch
#$5 - insertions
#$6 - strength_low
#$7 - strength_high
#$8 - shape_passwords
#$9 - features
#$10 - new_password Y or N
#$11 - attach_type
#$12 - batch_size
#$13 - dataset

#DICTIONARY ATTACK
python generate_password_space_with_features.py --r_space $2 --strength $6 $7 --N $1 --S $8 --features $9 --new_password ${10} --attack_type ${11} --epoch $4 --insertions $5

filename="r_space_data/$1_passwords_$2_r_space_$4_epoch_$5_insertions_${11}_attack/$1_r_space_passwords_strength_$6-$7.txt"
n=1
while read line; do
# reading each line
declare password="$line"
declare -i password_len=${#line}
declare -i start_loc=21
declare -i end_loc=start_loc+password_len

echo "The secret phrase is "$password""
#echo "Rana's secret is "$password""

#spacy2
#qsub -v password=$password,start_loc=$start_loc,end_loc=$end_loc,run=$n,n_passwords=$1,r_space=$2,knowledge=$3,epoch=$4,insertions=$5,strength_low=$6,strength_high=$7,features=$9,features_passwords=$8 jobscript;

#spacy3
qsub -v password=$password,start_loc=$start_loc,end_loc=$end_loc,run=$n,n_passwords=$1,r_space=$2,knowledge=$3,epoch=$4,insertions=$5,strength_low=$6,strength_high=$7,features=$9,features_passwords=$8,attack_type=${11},batch_size=${12},dataset=${13} jobscript_spacy3;


n=$((n+1))
done < $filename