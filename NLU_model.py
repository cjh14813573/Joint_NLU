import torch
from transformers import BertPreTrainedModel, BertModel
from torch import nn
from torchcrf import CRF

class NLUModel(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.args = args
        self.num_intent_labels = config.num_intent_labels
        self.num_slot_labels = config.num_slot_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.intent_classifier = nn.Linear(config.hidden_size, config.num_intent_labels)
        #self.slot_classifier = nn.Linear(config.hidden_size, config.num_slot_labels)
        self.slot_classifiers = []
        for i in range(self.num_intent_labels):
            self.slot_classifiers.append(nn.Linear(config.hidden_size, config.num_slot_labels))

        if self.args.crf is True:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

        self.init_weights()

    def forward(
            self,
            input_ids = None,
            slot_labels = None,
            attention_mask = None,
            token_type_ids = None,
            position_ids = None,
            head_mask = None,
            inputs_embeds = None,
            output_attentions = None,
            output_hidden_states = None,
            intent_labels = None,
            training = False
    ):
        outputs = self.bert(
            input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            position_ids = position_ids,
            head_mask = head_mask,
            inputs_embeds = inputs_embeds,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
        )

        pooled_output = outputs[1]
        seq_output = outputs[0]
        pooled_output = self.dropout(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)

        #slot_logits = self.slot_classifiers[intent_logits](seq_output)
        if training == True:
            intent_slot_labels = intent_labels
        else:
            intent_slot_labels = torch.argmax(intent_logits, dim=-1)
        #slot_logits = []
        for i in range(intent_slot_labels.shape[0]):
            if i == 0:
                slot_logits = self.slot_classifiers[intent_slot_labels[i]](seq_output[i]).unsqueeze(0)
            else:
                slot_logits = torch.cat((slot_logits,
                                        self.slot_classifiers[intent_slot_labels[i]](seq_output[i]).unsqueeze(0)),0)

        #slot_logits = torch.tensor([item.cpu().numpy() for item in slot_logits])

        if self.args.crf:
            slot_loss = self.crf(slot_logits, slot_labels, mask=attention_mask.byte(), reduction='mean')
            crf_loss = -1 * slot_loss
            crf_decode = self.crf.decode(slot_logits, mask=attention_mask.byte())
            return (intent_logits, slot_logits), (crf_loss, crf_decode)
        else:
            return (intent_logits, slot_logits), (None, None)