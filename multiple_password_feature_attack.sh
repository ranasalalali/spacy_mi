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

#DICTIONARY ATTACK
echo "You are about to gen password"
python3 generate_password_space_with_features.py --r_space $2 --strength $6 $7 --N $1 --S $8 --features $9

filename="r_space_data/$1_r_space_passwords_strength_$6-$7_features_$9.txt"
echo filename
n=1
while read line; do
# reading each line
declare password="$line"
declare -i password_len=${#line}
declare -i start_loc=17
declare -i end_loc=start_loc+password_len

echo "Rana's secret is "$password""

qsub -v password=$password,start_loc=$start_loc,end_loc=$end_loc,run=$n,n_passwords=$1,r_space=$2,knowledge=$3,epoch=$4,insertions=$5,strength_low=$6,strength_high=$7,features=$9 jobscript_spacy3;

n=$((n+1))
done < $filename


