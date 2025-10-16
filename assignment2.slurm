#!/bin/bash
#SBATCH --nodes=4
#SBATCH --partition=cits3402
#SBATCH --account=cits3402
#SBATCH --mem=0
#SBATCH --job-name=assignment2
#SBATCH --output=kayajob-%j.out
#SBATCH --error=myjob-%j.err
#SBATCH --ntasks=34
#SBATCH --time=00:15:00
module load gcc openmpi
make
for i in $(seq 40001 500 50001); do
    for j in $(seq 36 15 36); do
        # for k in $(seq 1 1 3); do 
            for l in $(seq 1 1 1); do
                   H=$((i ))
                    W=$((i * l))
                    echo "srun ./ass2 -p -r -H $H -W $W -kW $j -kH $j -sH $l -sW 1"
                    srun ./ass2 -p -r -H $H -W $W -kW $j -kH $j -sH $l -sW 1
                    # wait
                    # pkill -f orted
                    # pkill -f hydra_proxy
            done
        # done
    done
done
