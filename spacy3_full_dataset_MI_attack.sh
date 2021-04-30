#!/bin/bash

 
#$1 - epoch
#$2 - attach_type
#$3 - batch_size
#$4 - dataset
#$5 - member_set
#$6 - non_member_set
#$7 - subruns
#$8 - label

qsub -v epoch=$1,attack_type=$2,batch_size=$3,dataset=$4,member_set=$5,non_member_set=$6,subruns=$7,label=$8 jobscript_MIA;

