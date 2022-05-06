#!/bin/bash

N=1
sign_begin='--------------------------- model training begin -------------------------'
sign_end='--------------------------- model training end -------------------------'
while [ $N -le 10 ]

do

  echo 'Iteration:' $N
  model_name=model_$N

  echo $sign_begin
  python script/trainer.py --model_type gscales --dataset dset --s1_orbit as --model_name $model_name --num_epoch 500
  echo $sign_end

  echo $sign_begin
  python script/trainer.py --model_type gscales --dataset dset --s1_orbit des --model_name $model_name --num_epoch 500
  echo $sign_end

  N=$( echo $N + 1 | bc )

done


