#!/bin/bash
#$ -cwd
#$ -o plot.log
#$ -j y
#$ -l P4,h_data=2G,h_rt=24:00:00,exclusive
#$ -m bea

. /u/local/Modules/default/init/modules.sh
module load intel/14.cs 
module load intelmpi/5.0.0 
module load python/3.7.2

#module load python/3.6.1
#~/PYTHON/python3 
#qsub run.sh
python3 grids.py
python3 simul_dft.py
python3 stats.py
python3 train.py
python3 predict_mc.py
