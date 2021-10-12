.PHONY: all setup
default: all

clean:
	-rm -fr ./test-ner-none
	-rm -fr ./test-ner-immediate

train:
	python3 ner_test.py --model_name_or_path bert-base-uncased --dataset_name conll2003 --output_dir ./test-ner-none --do_train --do_eval
	python3 ner_test.py --model_name_or_path bert-base-uncased --abstention_method immediate --dataset_name conll2003 --output_dir ./test-ner-immediate --do_train --do_eval


all: clean train
	