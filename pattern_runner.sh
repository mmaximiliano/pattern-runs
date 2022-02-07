#!/bin/bash

echo "----- Running massive pattern creator -----"

c=1

for th in 450 460 470 490 500 520 540
do
  for seed in 11
  do
    echo "Loop: $c/7"
    echo "Running th= $th seed= $seed"
    python3 pattern-run.py -th $th -seed $seed
    echo
    c=$((c+1))
  done
done

echo "Done."