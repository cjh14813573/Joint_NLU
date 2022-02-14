from transformers import BertPreTrainedModel, BertModel
from torch import nn

class NLUModel(BertPreTrainedModel):
    def __init__(self,config):
        super.__init__(config)
        self.num_intent_labels = config.num_intent_labels
        self.num_slot_labels = config.num_slot_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.intent_classifier = nn.Linear(config.hidden_size, config.num_intent_labels)
        self.slot_classifier = nn.Linear(config.hidden_size, config.num_slot_labels)

        self.init_weights()

    def forward(
            self,
            input_ids = None,
            attention_musk = None,
            token_type_ids = None,
            position_ids = None,
            head_musk = None,
            input_embeds = None,
            output_attentions = None,
            output_hidden_states = None
    ):
        outputs = self.bert(
            input_ids,
            attention_musk = attention_musk,
            token_type_ids = token_type_ids,
            position_ids = position_ids,
            head_musk = head_musk,
            input_embeds = input_embeds,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
        )

        pooled_output = outputs[1]
        seq_output = outputs[0]
        pooled_output = self.dropout(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(seq_output)

        return intent_logits, slot_logits