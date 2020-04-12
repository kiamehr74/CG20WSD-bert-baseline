import torch
import torch.nn as nn
from transformers import BertModel
from architecture.custom_layers import MaskedAverageLayer


class Bert(nn.Module):

    def __init__(self, model_name, pretrained_path, class_count, freeze_bert=False):
        super().__init__()
        # Instantiating BERT model object
        self.pretrained = BertModel.from_pretrained(model_name, cache_dir=pretrained_path)

        # Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # other layers
        self.masked_average = MaskedAverageLayer()
        self.linear = nn.Linear(2 * self.pretrained.config.hidden_size, class_count)

    def forward(self, seq, attn_masks, target_mask):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
            -target_mask : target word mask
        '''
        embeddings, _ = self.pretrained(seq, attention_mask=attn_masks)
        masked_average = self.masked_average(embeddings, target_mask)
        cls = embeddings[:, 0]
        combined = torch.cat((cls, masked_average), 1)
        logits = self.linear(combined)
        return logits