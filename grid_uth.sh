#!/bin/bash

for scale in 0.01 0.025 0.05 0.075 0.1
do 
    LAMB=${scale} make clean train-avuc
    mv ./test/test-ner-avuc save/avuc_grid/${scale}_result
done