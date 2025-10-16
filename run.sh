#!/bin/bash

# mpicc -o ass1butMPI ass1butMPI.c -lm
# mpiexec -n 8 ass1butMPI -H 11 -W 11  -kW 5 -kH 5 -o out.txt

# gcc -fopenmp 23069261_assignment1.c -o 23069261 -lm
# mpicc -fopenmp  -o ass2 ass2.c -lm
make
# ./23069261 -ff ./f.txt -fg ./g.txt -o ./out.txt
# mpiexec -n 8 ass2 -H 1001 -W 1001  -kW 51 -kH 51 
for i in $(seq 2001 100 2001); do
    for j in $(seq 501 50 501); do
        for k in $(seq 1 1 1); do 
            for l in $(seq 1 1 1); do
                for p in 8; do
                   H=$((i * k))
                    W=$((i * l))
                    echo "mpiexec -n $p ass2 -r -H $H -W $W -kW $j -kH $j -sH $l -sW $k"
                    mpiexec -n $p ass2  -H $H -W $W -kW $j -kH $j -sH $l -sW $k 
                    wait
                    pkill -f orted
                    pkill -f hydra_proxy
                done
                
            done
        done
    done
done
# mpiexec -n 8 ass2 -H 10001 -W 10001  -kW 501 -kH 501 
# mpiexec -n 8 ass2 -H 1001 -W 1001  -kW 51 -kH 51 
# mpiexec -n 8 ass2 -H 1001 -W 1001  -kW 51 -kH 51 
# mpiexec -n 8 ass2 -H 1001 -W 1001  -kW 51 -kH 51 
# mpiexec -n 8 ass2 -H 1001 -W 1001  -kW 51 -kH 51 
# mpiexec -n 8 ass2 -H 1001 -W 1001  -kW 51 -kH 51 
# mpiexec -n 8 ass2 -H 1001 -W 1001  -kW 51 -kH 51 
# mpiexec -n 8 ass2 -H 1001 -W 1001  -kW 51 -kH 51 