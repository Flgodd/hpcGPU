#!/bin/bash

#SBATCH --job-name=d2q9-bgk
#SBATCH --gres=gpu:0
#SBATCH --partition=teach_gpu
#SBATCH --account=COMS031424
#SBATCH --output=lbm.out
#SBATCH --time=00:30:00
#SBATCH --nodes=1


#--ntasks-per-node=2
echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job ID is $SLURM_JOB_ID
echo This job runs on the following machines:
echo `echo $SLURM_JOB_NODELIST | uniq`
echo GPU number: $CUDA_VISIBLE_DEVICES

export OCL_DEVICE=1

module load libs/cuda/10.0-gcc-5.4.0-2.26
module use /software/x86/tools/nvidia/hpc_sdk/modulefiles
#module load NVIDIA/nvhpc/21.9
#export LD_LIBRARY_PATH=/usr/lib64/psm2-compat:$LD_LIBRARY_PATH
#! Run the executable
#./d2q9-bgk input_128x128.params obstacles_128x128.dat
#./d2q9-bgk input_256x256.params obstacles_256x256.dat
#./d2q9-bgk input_1024x1024.params obstacles_1024x1024.dat
nvprof ./d2q9-bgk input_2048x2048.params obstacles_2048x2048.dat
#cd /user/home/tn21145/osu-micro-benchmarks-7.4/c/mpi/pt2pt/standard/
#srun --mpi=pmi2 -N 1 --ntasks-per-node=2 ./osu_latency
