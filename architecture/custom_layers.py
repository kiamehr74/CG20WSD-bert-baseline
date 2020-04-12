import torch
import torch.nn as nn


class MaskedAverageLayer(nn.Module):
    def __init__(self):
      super(MaskedAverageLayer, self).__init__()

    def forward(self, seq, mask):
        '''
        Inputs:
            -seq : Tensor of shape [B, T, E] containing embeddings of sequences
            -mask : Tensor of shape [B, T, 1] containing masks to be used to pull from seq
        '''
        output = None
        if mask is not None:
          if len(mask.shape) < len(seq.shape):
            mask = mask.unsqueeze(-1)
            mask = mask.repeat(1, 1, seq.shape[-1])

          masked_inputs = (mask.int()*seq) + (1-mask.int())*torch.zeros_like(seq)
          unmasked_counts = torch.sum(mask.float(), dim=1)
          output = torch.sum(masked_inputs, dim=1)/(unmasked_counts+1e-10)

        return output