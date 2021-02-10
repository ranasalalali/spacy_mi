#!/bin/bash

 
#$1 - N passwords
#$2 - r_space
#$3 - knowledge
#$4 - epoch
#$5 - insertions
#$6 - strengths 

#DICTIONARY ATTACK
python generate_r_space.py --r_space $2 --type password --secret qwertyui --knowledge $3 --dictionary y
python pick_N_r_space_passwords.py --r_space $2 --N $1 --strength $6 $7

filename="r_space_data/$1_r_space_passwords_strength_$6-$7.txt"
n=1
while read line; do
# reading each line
declare password="$line"
declare -i password_len=${#line}
declare -i start_loc=17
declare -i end_loc=start_loc+password_len

echo "Rana's secret is "$password""

#qsub -v password=$password,start_loc=$start_loc,end_loc=$end_loc,run=$n,n_passwords=$1,r_space=$2,knowledge=$3,epoch=$4,insertions=$5 jobscript;

n=$((n+1))
done < $filename


