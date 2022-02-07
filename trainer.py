from transformers import Trainer
from transformers import TrainingArguments
from typing import Dict, Union, Any
from torch.nn.modules.module import Module
from torch.nn import DataParallel
import torch

def entropy(ten: torch.Tensor, dim: int = -1):
    return -1 * torch.sum(ten.log()*ten, dim=dim)

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_args: TrainingArguments = kwargs['args']
        self.model = kwargs['model']
        self.train_dataset = kwargs['train_dataset']
        self.eval_dataset  = kwargs['eval_dataset']
        self.tokenizer     = kwargs['tokenizer']

        if isinstance(self.model, DataParallel):
            obj = self.model.module
        else:
            obj = self.model
        self.training_step = self.training_step_normal if obj.abst_method == "raw" else self.training_step_last_only


    def set_freeze(self, model, frozen):
        for p in model.bert.parameters():
            p.requires_grad = not frozen

    def training_step_last_only(self, model: Module, inputs: Dict[str, Union[torch.Tensor, Any]]):
        if isinstance(model, DataParallel):
            obj = model.module
        else:
            obj = model

        abst_method = obj.abst_method
        model.train()
        inputs = self._prepare_inputs(inputs)


        count = 0

        # First pass
        if 'combine' not in abst_method:
            self.set_freeze(obj, False)
            obj.abst_method = 'raw' # regularizer disabled
            loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()
            loss.backward(); count+= 1


        # Second pass, only regulation on last layer
        if 'noise' not in abst_method:
            self.set_freeze(obj, 'unfrozen' not in abst_method)
            obj.abst_method = abst_method 
            loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()
            loss.backward(); count += 1


        ##! WARNING WARNING WARNING WARNING WARNING WARNING 
        ##! WARNING WARNING WARNING WARNING WARNING WARNING 
        ##! WARNING WARNING WARNING WARNING WARNING WARNING


        if 'noise' in abst_method:
            self.set_freeze(obj, True)
            batch_size, seq_len = inputs["input_ids"].shape
            noise = torch.cuda.FloatTensor(batch_size, seq_len, 768).normal_()
            probas = obj.classifier(noise).softmax(2)
            loss = torch.pow((1/probas.shape[2] - probas.max(2).values), 2).sum()
            if self.args.n_gpu > 1:
                loss = loss.mean()
            loss.backward(); count += 1

        # We just backpropagated 2 times on the final classifier but once only on the upstream language model
        # We /2 the classifier grad to break even
        for p in obj.classifier.parameters():
            p.grad /= float(count)

        #: WARN
        
        return loss.detach()

    def training_step_normal(self, model: Module, inputs: Dict[str, Union[torch.Tensor, Any]]):
        model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        loss.backward()
        return loss.detach()