from transformers import Trainer
from transformers import TrainingArguments
from typing import Dict, Union, Any
from torch.nn.modules.module import Module
from torch.nn import DataParallel
import torch

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_args: TrainingArguments = kwargs['args']
        self.model = kwargs['model']
        self.abst_method   = kwargs['abst_method']
        self.train_dataset = kwargs['train_dataset']
        self.eval_dataset  = kwargs['eval_dataset']
        self.tokenizer     = kwargs['tokenizer']

        self.training_step = self.training_step_normal if self.abst_method == "raw" else self.training_step_last_only


    def set_freeze(self, model, frozen):
        for p in model.bert.parameters():
            p.requires_grad = not frozen

    def training_step_last_only(self, model: Module, inputs: Dict[str, Union[torch.Tensor, Any]]):
        if isinstance(model, DataParallel):
            obj = model.model
        else:
            obj = model

        abst_method = obj.abst_method
        model.train()
        inputs = self._prepare_inputs(inputs)

        # First pass
        self.set_freeze(obj, False)
        obj.abst_method = 'raw' # regularizer disabled
        loss = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            loss = loss.mean()
        loss.backward()

        # Second pass, only regulation on last layer
        self.set_freeze(obj, True)
        obj.abst_method = abst_method # restore regularizer
        loss = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            loss = loss.mean()
        loss.backward()

        return loss.detach()

    def training_step_normal(self, model: Module, inputs: Dict[str, Union[torch.Tensor, Any]]):
        model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        loss.backward()
        return loss.detach()