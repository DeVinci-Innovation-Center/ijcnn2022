from argparse import ArgumentParser
from subprocess import PIPE, Popen

import datetime
import os
import sys
import shutil

DATASETS = ['conll2003', 'ncbi_disease', 'wikiann', 'GUM', 'wnut_17']


def launch(dataset, method, lamb, batch_size=8, model="bert-base-uncased", out='test/latest', freeze=False, mc=1):
#    	python3 ner_test.py --save_strategy epoch 
# --lamb=${LAMB} --per_device_train_batch_size ${BATCH_SIZE} 
# --model_name_or_path bert-base-uncased --abstention_method immediate 
# --dataset_name ${DATASET} --output_dir ./test/test-ner-immediate --do_train --do_eval

    if os.path.isdir(f'test/latest'):
        shutil.rmtree('test/latest')

    if os.path.isdir(f'datasets/{dataset}'):
        dataset_args = f'--train_file datasets/{dataset}/train.json --validation_file datasets/{dataset}/test.json'
    else:
        dataset_args = f'--dataset_name {dataset}'
        if dataset == 'wikiann':
            dataset_args += ' --dataset_config_name en'
        if dataset == 'dfki-nlp/few-nerd':
            dataset_args += ' --dataset_config_name supervised'


    meta_args  = f'--save_strategy no --save_steps 0 --no_dataloader_pin_memory --dataloader_num_workers 24 --per_device_train_batch_size {batch_size}'
    model_args = f'--model_name_or_path {model}'
    if freeze:
        model_args += ' --freeze'
    method_args= f'--lamb={lamb} --abstention_method {method} --mc {mc}'
    other_args = f'--output_dir {out} --do_train --do_eval'

    line = f'{sys.executable} ner_test.py {dataset_args} {meta_args} {method_args} {model_args} {other_args}'
    print(line)
    proc = Popen(line.split(' '), stdout=None, stderr=None, bufsize=0) # bufsize is for tqdm
    if proc.wait() != 0:
        exit(1)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-d', '--dataset', help="Dataset name, either hf or in dataset fodler")
    parser.add_argument('-m', '--method', help="Downstream as abstention method")
    parser.add_argument('-l', '--lamb', type=float, help='Passed downstream as the scaler')
    parser.add_argument('-o', '--output', help='Output folder', default="test/latest")
    parser.add_argument('-b', '--batch_size', default=8, type=int)
    parser.add_argument('-f', '--freeze', action='store_true', default=False, help="Freeze BERT model")
    parser.add_argument('-s', '--mc_samples', type=int, default=1, help="Number of MC Dropout Samples")

    args = parser.parse_args()
    if args.dataset == 'cycle':
        for dataset in DATASETS:
            #for lamb in linspace(...)
            launch(dataset, args.method, args.lamb, out=f'{args.output}/{dataset}', batch_size=args.batch_size, freeze=args.freeze, mc=args.mc_samples)
    else:
        launch(args.dataset, args.method, args.lamb, out=args.output, batch_size=args.batch_size, freeze=args.freeze, mc=args.mc_samples)
