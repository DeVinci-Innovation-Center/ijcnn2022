#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python3.8 launcher.py -d cycle -m immediate -l 0.025 -o test/immediate_no_noise_1hidden_64width -b 2 -s 2 -x 1 -w 64  -e 10
CUDA_VISIBLE_DEVICES=3 python3.8 launcher.py -d cycle -m immediate -l 0.025 -o test/immediate_no_noise_1hidden_128width -b 2 -s 2 -x 1 -w 128 -e 10
CUDA_VISIBLE_DEVICES=3 python3.8 launcher.py -d cycle -m immediate -l 0.025 -o test/immediate_no_noise_1hidden_150width -b 2 -s 2 -x 1 -w 150 -e 10

CUDA_VISIBLE_DEVICES=3 python3.8 launcher.py -d cycle -m immediate -l 0.025 -o test/immediate_no_noise_1hidden_200width -b 2 -s 2 -x 1 -w 200 -e 10
CUDA_VISIBLE_DEVICES=3 python3.8 launcher.py -d cycle -m immediate -l 0.025 -o test/immediate_no_noise_1hidden_256width -b 2 -s 2 -x 1 -w 256 -e 10
CUDA_VISIBLE_DEVICES=3 python3.8 launcher.py -d cycle -m immediate -l 0.025 -o test/immediate_no_noise_1hidden_312width -b 2 -s 2 -x 1 -w 312 -e 10
