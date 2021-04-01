import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import unsort

class b_model(nn.Module):
    def __init__(self, encoder_input_dim, args):
        super(b_model, self).__init__()
        self.output_dim = encoder_input_dim
        self.device = args.device


    def forward(self,encoded_input):
        embeded_big_tensor,sorted_lengths,sort_order,sentences_per_doc = encoded_input
        batch_size, out_dim = embeded_big_tensor.shape[0],embeded_big_tensor.shape[2]
        pooled_output = embeded_big_tensor[:,0,:].squeeze(1) # torch.zeros(batch_size, out_dim).to(self.device)
        # for i in range(batch_size):
        #     maxes[i, :] = torch.max(embeded_big_tensor[i, :sorted_lengths[i],:], 0)[0]
        unsort_order = torch.LongTensor(unsort(sort_order)).to(self.device)
        unsorted_sent_repr = pooled_output.index_select(0, unsort_order)
        return unsorted_sent_repr, sentences_per_doc

    def get_output_dim(self):
        return self.output_dim