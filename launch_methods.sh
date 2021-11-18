#!/bin/bash

python3.8 launcher.py -d cycle -m raw -l 0.025 -b 3 -o test/raw -s 10 || exit 1
python3.8 launcher.py -d cycle -m difficulty -l 0.025 -b 3 -o test/difficulty -s 10 || exit 1
python3.8 launcher.py -d cycle -m recall -l 0.025 -b 3 -o test/recall -s 10 || exit 1
python3.8 launcher.py -d cycle -m avuc -l 0.025 -b 3 -o test/avuc -s 10 || exit 1
python3.8 launcher.py -d cycle -m immediate -l 0.025 -b 3 -o test/immediate -s 10 || exit 1
