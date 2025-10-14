#!/bin/bash
#SBATCH --nodes=4
#SBATCH --partition=work
#SBATCH --account=courses0101
#SBATCH --mem=0
#SBATCH --job-name=assignment2
#SBATCH --ntasks-per-node=1     # 8 MPI ranks per node
#SBATCH --cpus-per-task=128      # 16 threads per rank
#SBATCH --time=00:15:00
module load gcc openmpi
make
for i in $(seq 200001 5000 200001); do
    for j in $(seq 36 15 36); do
        # for k in $(seq 1 1 3); do 
            for l in $(seq 1 1 3); do
                   H=$((i ))
                    W=$((i * l))
                    echo "srun ./ass2 -p -r -H $H -W $W -kW $j -kH $j -sH $l -sW 1"
                    srun ./ass2 -p -r -H $H -W $W -kW $j -kH $j -sH $l -sW 1
                    wait
                    pkill -f orted
                    pkill -f hydra_proxy
            done
        # done
    done
done