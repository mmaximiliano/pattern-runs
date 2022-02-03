#!/bin/bash

echo "----- Running massive pattern creator -----"

c=8

for th in 500 550 600 650 700 750 800 850
do
  for ap in "0.009125" "550" "600" "650"
  do
    for am in "0.009125" "550" "600" "650"
      do
        echo "Loop: $c/1"
        echo "Running th= $th a_minus= $am a_plus= $ap"
        python3 pattern-run.py -th $th -am $am -ap $ap
        echo
        c=$((c+1))
      done
    done
done

echo "Done."