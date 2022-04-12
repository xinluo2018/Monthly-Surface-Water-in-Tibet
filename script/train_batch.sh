#!/bin/bash

N=1

while [ $N -le 5 ]

do

  echo 'Iteration:' $N
  model_name=model_$N
  # python script/trainer.py --model_type single --dataset traset --s1_orbit as_des --model_name $model_name
  # python script/trainer.py --model_type scales --dataset traset --s1_orbit as_des --model_name $model_name
  # python script/trainer.py --model_type gscales --dataset traset --s1_orbit as_des --model_name $model_name
  # python script/trainer.py --model_type gscales --dataset traset --s1_orbit as --model_name $model_name
  # python script/trainer.py --model_type gscales --dataset traset --s1_orbit des --model_name $model_name  
  python script/trainer.py --model_type gscales --dataset dset --s1_orbit as --model_name $model_name  
  python script/trainer.py --model_type gscales --dataset dset --s1_orbit des --model_name $model_name  
  N=$( echo $N + 1 | bc )

done


