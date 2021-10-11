from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.bert import BertForTokenClassification

class AbstentionBertForSequenceClassification(BertForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
    

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
        return_dict=None):
        
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

        a = output.logits.softmax(dim = 2).max(dim=2)
        return output

