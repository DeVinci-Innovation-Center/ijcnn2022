.PHONY: all setup
default: all

clean:
	-rm -fr ./test-ner

train:
	python3 ner_test.py --model_name_or_path bert-base-uncased --dataset_name conll2003 --output_dir ./test-ner --do_train --do_eval

setup:
	mkdir -p models
	wget -O models/bert-base-uncased-pytorch_model.bin https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin

all: clean train
	