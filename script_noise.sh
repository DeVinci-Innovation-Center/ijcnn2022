#!/bin/bash
python3.8 launcher.py -d cycle -m immediate -l 0.025 -o test/immediate_noise_1hidden -b 2 -s 2 -x 1 -e 10
python3.8 launcher.py -d cycle -m immediate -l 0.025 -o test/immediate_noise_2hidden -b 2 -s 2 -x 2 -e 10
python3.8 launcher.py -d cycle -m immediate -l 0.025 -o test/immediate_noise_3hidden -b 2 -s 2 -x 3 -e 10
python3.8 launcher.py -d cycle -m immediate -l 0.025 -o test/immediate_noise_4hidden -b 2 -s 2 -x 4 -e 10
python3.8 launcher.py -d cycle -m immediate -l 0.025 -o test/immediate_noise_5hidden -b 2 -s 2 -x 5 -e 10
python3.8 launcher.py -d cycle -m immediate -l 0.025 -o test/immediate_noise_6hidden -b 2 -s 2 -x 6 -e 10
