#!/bin/bash

for scale in 0.03 0.035 0.04 0.045
do 
    LAMB=${scale} make clean train-immediate
    mv ./test/test-ner-immediate save/custom_grid/${scale}_result
done

#0.01 0.25 0.5 0.75 