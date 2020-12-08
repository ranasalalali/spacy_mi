#!/bin/bash


#DICTIONARY ATTACK
python generate_r_space.py --r_space 10000 --type password --secret qwertyui
python pick_N_r_space_passwords.py --r_space 1000 --N $1

filename="r_space_data/$1_r_space_passwords.txt"
n=1
while read line; do
# reading each line
declare password="$line"
declare -i password_len=${#line}
declare -i start_loc=17
declare -i end_loc=start_loc+password_len

qsub -v password=$password,start_loc=$start_loc,end_loc=$end_loc,run=$n,n_passwords=$1 jobscript;

n=$((n+1))
done < $filename