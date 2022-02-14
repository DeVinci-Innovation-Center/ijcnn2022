CUDA_VISIBLE_DEVICES=3 python3.8 launcher.py -d cycle -m combine_immediate_unfrozen -l 0.025 -o test/combine_immediate_unfrozen_1hidden_256width -b 2 -s 2 -x 1 -w 256 -e 10
CUDA_VISIBLE_DEVICES=3 python3.8 launcher.py -d cycle -m combine_avuc -l 0.025 -o test/combine_avuc_1hidden_128_width -b 2 -s 2 -x 1 -w 128 -e 10


#CUDA_VISIBLE_DEVICES=3 python3.8 launcher.py -d cycle -m noise -l 1   -o test/noise_1hidden_64width_run2_1e-0  -b 2 -s 2 -x 1 -w 64 -e 10
#CUDA_VISIBLE_DEVICES=3 python3.8 launcher.py -d cycle -m noise -l 0.1 -o test/noise_1hidden_64width_run2_1e-1  -b 2 -s 2 -x 1 -w 64 -e 10
#CUDA_VISIBLE_DEVICES=3 python3.8 launcher.py -d cycle -m noise -l 0.5 -o test/noise_1hidden_64width_run2_5e-1  -b 2 -s 2 -x 1 -w 64 -e 10
#CUDA_VISIBLE_DEVICES=3 python3.8 launcher.py -d cycle -m noise -l 1.2 -o test/noise_1hidden_64width_run2_12e-1 -b 2 -s 2 -x 1 -w 64 -e 10

