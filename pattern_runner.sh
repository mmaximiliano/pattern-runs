#!/bin/bash

echo "----- Running massive pattern creator -----"

c=8

for th in 450 470 490 500 520 540
do
  for seed in 1 2 3 4 5 6 7 8
  do
    echo "Loop: $c/1"
    echo "Running th= $th seed= $seed"
    python3 pattern-run.py -th $th -seed $seed
    echo
    c=$((c+1))
  done
done

echo "Done."