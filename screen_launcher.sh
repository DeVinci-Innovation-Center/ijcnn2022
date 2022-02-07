#!/bin/bash

function term () {
    echo Terminating $1
    screen -X -S ner$1 quit
}

# python launcher.py -d cycle -m avuc -l 0.025 -b 3 -o test/avuc -s 10 || exit 1
function launch () {
    num=$1
    echo Launching ner$1 on gpu $(($((${num}-1))/2))
    screen -dmS ner$1
    num=$1 bash -c 'screen -S ner${num} -X stuff $"CUDA_VISIBLE_DEVICES=$(($((${num}-1))/2)) python3.8 launcher.py -d cycle -m noise -l 0.05 -o test/combine_avuc_${num}hidden_128width_5e-2 -b 3 -s 2 -x ${num} -w 128 -e 10\n"'
}

term 1
term 2
term 3
term 4
term 5
term 6
term 7
term 8

launch 1
launch 2
launch 3
launch 4
launch 5
launch 6
launch 7
launch 8
