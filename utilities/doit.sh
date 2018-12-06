#!/bin/bash

file=md0.log
#1 method one, search for total # of exchanges, divide by attempts

nrep=`grep replicas $file  | tail -n 1 | awk '{print $4}'`
success=`grep Repl\ ex  $file | grep -o " x " | wc -l`
val1=`grep Repl\ ex  $file | wc -l` 
multiply=`echo "($nrep-1)/2" | bc -l`
attempts=`echo "$multiply*$val1" | bc -l | awk '{printf "%4.0f\n", $1}' `
prob=`echo "$success/($multiply*$val1)" | bc -l | awk '{printf "%4.4f\n", $1}' `
echo "method 1 success rate: $prob with $attempts attempts and $success successes" 
