CUDA_VISIBLE_DEVICES=3 python3.8 launcher.py -d cycle -m immediate -l 0.025 -o test/immediate_no_noise_2hidden_200width -b 2 -s 2 -x 2 -w 200 -e 10
CUDA_VISIBLE_DEVICES=3 python3.8 launcher.py -d cycle -m immediate -l 0.025 -o test/immediate_no_noise_2hidden_256width -b 2 -s 2 -x 2 -w 256 -e 10
CUDA_VISIBLE_DEVICES=3 python3.8 launcher.py -d cycle -m immediate -l 0.025 -o test/immediate_no_noise_2hidden_312width -b 2 -s 2 -x 2 -w 312 -e 10
