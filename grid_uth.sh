#!/bin/bash

for scale in 0.3 0.35 0.4 0.45
do 
    LAMB=${scale} make clean train-immediate
    mv ./test/test-ner-immediate save/custom_grid/${scale}_result
done

#0.01 0.25 0.5 0.75 