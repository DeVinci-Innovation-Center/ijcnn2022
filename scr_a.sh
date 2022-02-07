CUDA_VISIBLE_DEVICES=2 python3.8 launcher.py -d cycle -m noise -l 0.005 -o test/noise_1hidden_64width_run1_5e-3 -b 2 -s 2 -x 1 -w 64  -e 10
CUDA_VISIBLE_DEVICES=2 python3.8 launcher.py -d cycle -m noise -l 0.05 -o test/noise_1hidden_64width_run1_5e-2 -b 2 -s 2 -x 1 -w 64  -e 10
CUDA_VISIBLE_DEVICES=2 python3.8 launcher.py -d cycle -m noise -l 0.1 -o test/noise_1hidden_64width_run1_1e-1 -b 2 -s 2 -x 1 -w 64 -e 10
CUDA_VISIBLE_DEVICES=2 python3.8 launcher.py -d cycle -m noise -l 0.5 -o test/noise_1hidden_64width_run1_5e-1 -b 2 -s 2 -x 1 -w 64 -e 10
