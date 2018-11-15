#!/bin/bash

if [[ $# -ne 1 ]]; then
  NRUNS=10
else
  NRUNS=$1
fi

START=$(date +%s)

for i in `seq 1 $NRUNS`;do
  #echo "run $i"
  ./Main
done


END=$(date +%s)
DIFF=$(echo "$END - $START" | bc)
echo "total runtime of $NRUNS jobs: $DIFF s"
