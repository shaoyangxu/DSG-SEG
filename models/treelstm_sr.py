import torch
import torch.nn as nn
from models.Treelstm.basic import *
from models.Treelstm.utils import *
from models.Treelstm.models import *
class t_model_treelstm(nn.Module):
    def __init__(self,encoder_input_dim,args):
        super(t_model_treelstm, self).__init__()
        self.use_leaf_rnn = args.use_leaf_rnn
        self.pooling_method = args.pool_method
        assert self.pooling_method in ['None','max','mean','attention']
        self.bidirectional = args.bidirectional
        self.device = args.device
        self.hidden_dim = args.attention_hd
        self.word_dim = encoder_input_dim
        self.pooling_method = None if self.pooling_method == 'None' else self.pooling_method
        self.treelstm = RecursiveTreeLSTMEncoder(
            self.word_dim,
            self.hidden_dim,
            self.use_leaf_rnn,
            self.pooling_method,
            self.bidirectional
        )
        self.transpose = torch.nn.Linear(self.get_dim(), 512)  # same input dim with baseline:Textseg

    def forward(self,encoded_input):
        embeded_sentence, lengths, tree_masks, sentences_per_doc = encoded_input
        treelstm_output,_ = self.treelstm(inp=embeded_sentence, length=lengths,fixed_masks=tree_masks)
        return self.transpose(treelstm_output), sentences_per_doc

    def get_dim(self):
        if self.bidirectional:
            return self.word_dim * 2
        else:
            return self.word_dim

    def get_output_dim(self):
        return 512
