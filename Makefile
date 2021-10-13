.PHONY: all setup
default: all

setup:
	mkdir -p test

clean:
	-rm -fr ./test

train-none: setup
	python3 ner_test.py --per_device_train_batch_size 16 --model_name_or_path bert-base-uncased --dataset_name conll2003 --output_dir ./test/test-ner-none --do_train --do_eval
	
train-immediate: setup
	python3 ner_test.py --per_device_train_batch_size 16 --model_name_or_path bert-base-uncased --abstention_method immediate --dataset_name conll2003 --output_dir ./test/test-ner-immediate --do_train --do_eval

eval-none:
	python3 ner_test.py --model_name_or_path bert-base-uncased --dataset_name conll2003 --output_dir ./test-ner-none --do_eval

train: train-none train-immediate
all: clean train
	