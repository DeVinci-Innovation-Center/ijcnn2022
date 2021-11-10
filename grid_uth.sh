#!/bin/bash

for scale in 0.01 0.025 0.05 0.075 0.1
do 
    python launcher.py -d GUM -m recall -l ${scale} -o test/grid -b 8
    mv ./test/grid save/custom_grid/${scale}_result
done

#0.01 0.25 0.5 0.75 