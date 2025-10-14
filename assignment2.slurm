#!/bin/bash
#SBATCH --cpus-per-task=36
#SBATCH --ntasks=8
#SBATCH --time=00:15:00
#SBATCH --mem=256G
#SBATCH --partition=cits3402
module load gcc openmpi
make
for i in $(seq 400001 5000 400001); do
    for j in $(seq 36 15 36); do
        # for k in $(seq 1 1 3); do 
            for l in $(seq 1 1 3); do
                   H=$((i ))
                    W=$((i * l))
                    echo "mpiexec -n ${SLURM_NTASKS} ass2 -r -H $H -W $W -kW $j -kH $j -sH $l -sW 1"
                    mpiexec -n ${SLURM_NTASKS} ass2 -r -H $H -W $W -kW $j -kH $j -sH $l -sW 1
                    wait
                    pkill -f orted
                    pkill -f hydra_proxy
            done
        # done
    done
done