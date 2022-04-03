#!/bin/bash

N=1

while [ $N -le 10 ]
do
  model_name=model_$N
  python trainer-tmp/trainer.py --model_type gscales --dataset traset --s1_orbit as_des --model_name $model_name
  python trainer-tmp/trainer.py --model_type scales --dataset traset --s1_orbit as_des --model_name $model_name
  python trainer-tmp/trainer.py --model_type single --dataset traset --s1_orbit as_des --model_name $model_name
  echo 'Iteration:' $N
  N=$( echo $N + 1 | bc )
done

