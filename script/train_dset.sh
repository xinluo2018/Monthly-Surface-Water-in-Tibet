#!/bin/bash
## author: xin luo
## create: 2022.5.11
## des:train the model based on the whole dataset

N=21
sign_begin='--------------------------- model training begin -------------------------'
sign_end='--------------------------- model training end -------------------------'
while [ $N -le 25 ]

do

  echo 'Iteration:' $N
  model_name=model_$N

  # echo $sign_begin
  # python script/trainer.py --model_type gscales --dataset dset --s1_orbit as --model_name $model_name --num_epoch 500
  # echo $sign_end

  # echo $sign_begin
  # python script/trainer.py --model_type gscales --dataset dset --s1_orbit des --model_name $model_name --num_epoch 500
  # echo $sign_end

  echo $sign_begin
  python script/trainer.py --model_type gscales --dataset dset --s1_orbit as_des --model_name $model_name --num_epoch 500
  echo $sign_end


  N=$( echo $N + 1 | bc )

done


