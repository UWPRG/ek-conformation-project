#!/bin/bash

Tinit=300.0
Tfinal=450.0
nfinal=7

count=0

k=`echo | awk '{print log(TF/TI)/numf}' TF=$Tfinal TI=$Tinit numf=$nfinal`
echo $k 
for ((i=0;i<=$nfinal;i++)); do
  conf=$i.gro
  temp=`echo | awk '{printf "%3.3f\n",TI*exp(i*k)}' TI=$Tinit k=$k i=$i` 
  cp skel.mdp $count.mdp
  perl -pi -e "s/XXX/$temp/g" $count.mdp
  gmx grompp  -f $count.mdp -c $conf 
  mv topol.tpr topol$count.tpr
  rm -rf mdout.mdp 
  count=$(($count+1))
done;

