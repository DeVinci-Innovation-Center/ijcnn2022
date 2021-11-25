#!/bin/bash
python launcher.py -d cycle -m immediate -l 0.025 -o test/immediate_noise_1hidden -b 2 -s 2 -x 1
python launcher.py -d cycle -m immediate -l 0.025 -o test/immediate_noise_2hidden -b 2 -s 2 -x 2
python launcher.py -d cycle -m immediate -l 0.025 -o test/immediate_noise_3hidden -b 2 -s 2 -x 3
python launcher.py -d cycle -m immediate -l 0.025 -o test/immediate_noise_4hidden -b 2 -s 2 -x 4
python launcher.py -d cycle -m immediate -l 0.025 -o test/immediate_noise_5hidden -b 2 -s 2 -x 5
