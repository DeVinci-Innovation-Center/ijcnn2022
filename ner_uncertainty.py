from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.bert import BertForTokenClassification
import torch

def entropy(ten: torch.Tensor, dim: int):
    return -1 * torch.sum(ten.log()*ten, dim=dim)

class AbstentionBertForTokenClassification(BertForTokenClassification):
    def __init__(self, config, abst_meth: str, lamb: float = 5e-2):
        super().__init__(config)
        self.lamb = lamb
        self.abst_method = abst_meth
        self.uth = 5e-4 # FIXME too high: loss crash, moving average?


    def loss_miss_labels_recall(self, confidence, prediction, labels):
        O_LABEL = 0 # O label lebel int

        o_labels = (labels == O_LABEL)
        correctness = (prediction == labels)
        missed_labels = torch.logical_and(~o_labels, ~correctness)
        missed_confidence = torch.masked_select(confidence, missed_labels)

        return self.lamb * torch.exp(-1 * missed_confidence).sum()

    def loss_abstention(self, confidence, prediction, labels):
        # batch, example, proba
        #!!! WARNING: Implicit Sum aggregator with torch.masked_select
        #TODO: test out mean, or others
        
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

        # outputs: [Batch_norm, SequenceLength, NClasses]
        
        if labels is not None:
            probas = output.logits.softmax(dim = 2)
            confidence, prediction = probas.max(dim=2)

            if self.abst_method == "avuc":
                output.loss += self.loss_avuc(probas, confidence, prediction, labels) + self.loss_miss_labels_recall(confidence, prediction, labels)

            if self.abst_method == "immediate":
                output.loss += self.loss_abstention(confidence, prediction, labels) + self.loss_miss_labels_recall(confidence, prediction, labels)

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


        return output

