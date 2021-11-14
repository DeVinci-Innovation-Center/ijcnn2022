#!/bin/bash

python launcher.py -d cycle -m raw -l 0.025 -b 5 -o test/raw || exit 1
python launcher.py -d cycle -m recall -l 0.025 -b 5 -o test/recall || exit 1
python launcher.py -d cycle -m avuc -l 0.025 -b 5 -o test/avuc || exit 1
python launcher.py -d cycle -m immediate -l 0.025 -b 5 -o test/immediate || exit 1
python launcher.py -d cycle -m recall-immediate -l 0.025 -b 5 -o test/recallimmediate || exit 1