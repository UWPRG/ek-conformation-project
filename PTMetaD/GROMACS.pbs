#!/bin/bash
#PBS -N EK-conf-2
#PBS -l nodes=12:ppn=16
#PBS -l feature=16core
#PBS -l mem=22gb
#PBS -l walltime=999:00:00
# #PBS -W depend=afterany:2701881
## Put the output from jobs into the below directory
## Put both the stderr and stdout into a single file
#PBS -j oe
## Sepcify the working directory for this job

### You shouldn't need to change anything in this section ###
###                                                       ###
# Total Number of processors (cores) to be used by the job
HYAK_NPE=$(wc -l < $PBS_NODEFILE)
# Number of nodes used by MPICH
HYAK_NNODES=$(uniq $PBS_NODEFILE | wc -l )

### You shouldn't need to change anything in this section ###
###                                                       ###
echo "**** CPU and Node Utilization Information ****"
echo "This job will run on $HYAK_NPE total CPUs on $HYAK_NNODES different nodes"
echo ""
echo "Node:CPUs Used"
uniq -c $PBS_NODEFILE | awk '{print $2 ":" $1}'
echo "**********************************************"


module load icc_15.0.3-impi_5.0.3
export PATH=$PATH:/gscratch/pfaendtner/cdf6gc/codes/PRG_USE/plumed2/bin
export INCLUDE=$INCLUDE:/gscratch/pfaendtner/cdf6gc/codes/PRG_USE/plumed2/include
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gscratch/pfaendtner/cdf6gc/codes/PRG_USE/plumed2/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gscratch/pfaendtner/cdf6gc/codes/PRG_USE/libmatheval/lib/.libs
source /gscratch/pfaendtner/cdf6gc/codes/PRG_USE/gromacs-5.1.2/bin/bin/GMXRC

cd $PBS_O_WORKDIR

mpiexec.hydra gmx_mpi mdrun -multi 12 -cpt 1.0 -cpo restart -plumed plumed.dat -replex 250



#dir=`pwd | awk -F '/' '{print $NF}'`
#cd ..
#./convg_compare.sh $dir


### include any post processing here                      ###
###                                                       ###
#

exit 0

