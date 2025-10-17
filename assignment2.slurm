#!/bin/bash
#SBATCH --nodes=4
#SBATCH --partition=cits3402
#SBATCH --account=cits3402
#SBATCH --mem=0
#SBATCH --job-name=assignment2
#SBATCH --output=kayajob-%j.out
#SBATCH --error=myjob-%j.err
#SBATCH --ntasks=96
#SBATCH --time=00:15:00
module load gcc openmpi
make
export MPICH_ABORT_ON_ERROR=1
export MPICH_STACKTRACE=1
export MPICH_VERSION_DISPLAY=1
for i in $(seq 30001 500 40001); do
    for j in $(seq 36 15 36); do
        # for k in $(seq 1 1 3); do 
            for l in $(seq 1 1 1); do
                   H=$((i ))
                    W=$((i * l))
                    echo "srun ./ass2 -p -r -H $H -W $W -kW $j -kH $j -sH $l -sW 1"
                    mpiexec ./ass2 -p -r -H $H -W $W -kW $j -kH $j -sH $l -sW 1
                    # wait
                    # pkill -f orted
                    # pkill -f hydra_proxy
            done
        # done
    done
done
