CUDA_VISIBLE_DEVICES=3 python3.8 launcher.py -d cycle -m immediate -l 0.025 -o test/immediate_no_noise_2hidden_64width -b 2 -s 2 -x 2 -w 64  -e 10
CUDA_VISIBLE_DEVICES=3 python3.8 launcher.py -d cycle -m immediate -l 0.025 -o test/immediate_no_noise_2hidden_128width -b 2 -s 2 -x 2 -w 128 -e 10
CUDA_VISIBLE_DEVICES=3 python3.8 launcher.py -d cycle -m immediate -l 0.025 -o test/immediate_no_noise_2hidden_150width -b 2 -s 2 -x 2 -w 150 -e 10
