from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.bert import BertForTokenClassification
import torch.nn as nn
import torch

class PrinterLayer(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        print(x)
        print(x.shape)
        return x

def entropy(ten: torch.Tensor, dim: int):
    return -1 * torch.sum(ten.log()*ten, dim=dim)

class AbstentionBertForTokenClassification(BertForTokenClassification):
    def __init__(self, config, abst_meth: str, lamb: float = 5e-2, mc_samples = 10, hidden_layers = 0, width = 128):
        super().__init__(config)
        self.lamb = lamb
        self.abst_method = abst_meth
        self.mc_samples = mc_samples
        self.uth = 5e-4 # FIXME too high: loss crash, moving average?
        self.register_parameter("beta", nn.parameter.Parameter(torch.tensor(1.), requires_grad=True))
        
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if hidden_layers == 0 else nn.Sequential(
            nn.Linear(config.hidden_size, width), 
            *[ nn.Sequential(nn.Linear(width, width), nn.ReLU()) for i in range(hidden_layers - 1) ], 
            nn.Linear(width, config.num_labels)
        )
        self.init_weights()

    def loss_miss_labels_recall(self, confidence, prediction, labels):
        O_LABEL = 0 # O label lebel int

        o_labels = (labels == O_LABEL)
        correctness = (prediction == labels)
        missed_labels = torch.logical_and(~o_labels, ~correctness)
        missed_confidence = torch.masked_select(confidence, missed_labels)

        # beta param: scheduler or regularisatrion
        return self.lamb * (torch.exp(missed_confidence) - 1.).sum()

    def loss_abstention_entropy(self, probas, labels):
    
        confidence = entropy(probas, dim=2)
        correctness = torch.argmax(probas, dim=2) == labels
        correct_confidence = torch.masked_select(confidence, correctness)
        wrong_confidence = torch.masked_select(confidence, ~correctness)
        regularizer = 0

        for cc in correct_confidence:
            for wc in wrong_confidence:
                regularizer += torch.clamp(cc-wc, min=0) ** 2 #torch.clamp(wc-cc, min=0) ** 2
        return self.lamb * regularizer

    def loss_abstention(self, confidence, prediction, labels):
        # batch, example, proba
        #!!! WARNING: Implicit Sum aggregator with torch.masked_select
        
        correctness = (prediction == labels)
        correct_confidence = torch.masked_select(confidence, correctness)
        wrong_confidence = torch.masked_select(confidence, ~correctness)
        
        regularizer = 0
        for cc in correct_confidence:
            for wc in wrong_confidence:
                regularizer += torch.clamp(wc-cc, min=0) ** 2
        return self.lamb * regularizer

    def loss_avuc(self, probas: torch.Tensor, confidence: torch.Tensor, prediction: torch.Tensor, labels: torch.Tensor):
        # uncertainty = 1 - confidence #? can also use other methods: entropy variance etc...
        uncertainty = entropy(probas, 2) # Probas is (B, S, P)
        self.uth = uncertainty.median() * self.lamb #1e-1

        correctness = (prediction == labels)
        certainty = (uncertainty < self.uth)

        ac_p = torch.masked_select(confidence,   torch.logical_and(correctness, certainty))
        ac_u = torch.masked_select(uncertainty,  torch.logical_and(correctness, certainty))

        au_p = torch.masked_select(confidence,   torch.logical_and(correctness, ~certainty))
        au_u = torch.masked_select(uncertainty,  torch.logical_and(correctness, ~certainty))

        ic_p = torch.masked_select(confidence,   torch.logical_and(~correctness, certainty))
        ic_u = torch.masked_select(uncertainty,  torch.logical_and(~correctness, certainty))
        
        iu_p = torch.masked_select(confidence,   torch.logical_and(~correctness, ~certainty))
        iu_u = torch.masked_select(uncertainty,  torch.logical_and(~correctness, ~certainty))

        nac = torch.sum(ac_p * (1 - torch.tanh(ac_u)))
        nau = torch.sum(au_p * torch.tanh(au_u))
        nic = torch.sum( (1 - ic_p) * (1 - torch.tanh(ic_u)))
        niu = torch.sum( (1 - iu_p) * torch.tanh(iu_u))

        return torch.log(1 + ( (nau + nic) / (nac + niu) ))

    def loss_top2(self, probas, labels):
        correctness = torch.argmax(probas, dim=2) == labels

        cert = torch.topk(probas, 2, -1).values # batch, samples, 2
        cert = 1 - cert[:,:,0] # (1 - (cert[:,:,0] - cert[:,:,1]))
        cert_false = torch.pow((1/probas.shape[2] - probas.max(2).values), 2) #probas.std(2) #2 - cert

        l    = torch.sum(torch.masked_select(cert, correctness))
        lf   = 0. #torch.sum(torch.masked_select(cert_false, ~correctness))

        # print(f'l={l}')
        # print(f'lf={lf}')
        # exit(0)

        return self.lamb * (l + lf)

    def loss_difficulty(self, difficulty, probas, labels):
        # diff: B, E
        # probas B, E, O
        entrop = entropy(probas, dim=2)

        correctness = torch.argmax(probas, dim=2) == labels

        correct_difficulty = torch.masked_select(difficulty, correctness)
        correct_entropy = torch.masked_select(entrop, correctness)

        incorrect_entropy = torch.masked_select(entrop, ~correctness)
        incorrect_difficulty = torch.masked_select(difficulty, ~correctness)

        # print(f'''
        # entrop {entrop.shape}
        # correct_entrop {correct_entropy.shape}
        # correct_diff {correct_difficulty.shape}
        # incorrect_entrop {incorrect_entropy.shape}
        # incorrect_diff {incorrect_difficulty.shape}
        # ''')

        l = torch.sum(correct_difficulty*correct_entropy) + torch.sum(1/incorrect_difficulty*incorrect_entropy)
        # print(l)
        # l = torch.sum(torch.log(1+l))

        return self.lamb * l

    def forward(self, 
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs):
        
        mc_samples = []
        for _ in range(self.mc_samples):
            output: TokenClassifierOutput = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids, 
                position_ids=position_ids, 
                head_mask=head_mask, 
                inputs_embeds=inputs_embeds, 
                labels=labels, 
                output_attentions=output_attentions, 
                output_hidden_states=output_hidden_states, 
                return_dict=return_dict)
            mc_samples.append(output)
        
        mean_logits = torch.stack([i.logits for i in mc_samples]).mean(dim=0)
        mean_loss   = sum([i.loss for i in mc_samples])/self.mc_samples
        # outputs: [Batch_norm, SequenceLength, NClasses]
        output = TokenClassifierOutput(mean_loss, mean_logits)
        
        if labels is not None:
            # difficulty = torch.stack([entropy(i.logits.softmax(dim = 2), dim=2) for i in mc_samples]).std(0)
            difficulty = torch.stack([i.logits for i in mc_samples]).std(0).sum(2)

            probas = output.logits.softmax(dim = 2)
            confidence, prediction = probas.max(dim=2)

            if "top2" in self.abst_method:
                output.loss = self.loss_top2(probas, labels)

            if "difficulty" in self.abst_method:
                output.loss = self.loss_difficulty(difficulty, probas, labels)

            if "entrop" in self.abst_method:
                output.loss = self.loss_abstention_entropy(probas, labels)

            if "recall" in self.abst_method:
                output.loss = self.loss_miss_labels_recall(confidence, prediction, labels)

            if "avuc" in self.abst_method:
                output.loss = self.loss_avuc(probas, confidence, prediction, labels)

            if "immediate" in self.abst_method:
                l = self.loss_abstention(confidence, prediction, labels)
                if l == 0:
                    output.loss = output.loss * 0 # Neutralize loss if no modification
                else:
                    output.loss = l # apply regularizer

            if self.abst_method == "history":
                 if self.training:
                    batch_size = input_ids.size()[0]
                    # here correctness is continuous in [0,1]

                    correctness = kwargs['history_record']
                    _, sorted_correctness_index = torch.sort(correctness)
                    lower_index = sorted_correctness_index[:int(0.2 * batch_size)]
                    higher_index = sorted_correctness_index[int(0.2 * batch_size):]
                    regularizer = 0
                    for li in lower_index:  # indices with lower correctness
                        for hi in higher_index:
                            if correctness[li] < correctness[hi]:
                                # only if it's strictly smaller
                                regularizer += torch.clamp(
                                    confidence[li] - confidence[hi], min=0
                                ) ** 2


            if self.abst_method == "combination":
                c = self.loss_abstention(confidence, prediction, labels) + self.loss_avuc(probas, confidence, prediction, labels)
                c *= 1e-1 # simple scaling, put in parameters
                output.loss += c

        # print(f'{self.beta} {self.lamb}')

        return output

