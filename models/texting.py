import torch.nn as nn
import torch
from utils import unsort
from models.GGNN.layers import *

class g_model_text_ing(nn.Module):
    def __init__(self,encoder_input_dim,args):
        super(g_model_text_ing,self).__init__()
        self.input_dim = encoder_input_dim
        self.hidden_dim = args.texting_hidden_dim
        self.output_dim = args.texting_output_dim
        self.device = args.device
        self.layers = []
        self.graph_layer = GraphLayer(args,
                                      input_dim=self.input_dim,
                                      output_dim=self.hidden_dim,
                                      act=nn.Tanh(),
                                      dropout_p=0.3)

        self.readout_layer = ReadoutLayer(args,
                                        input_dim=self.hidden_dim,
                                        output_dim=self.output_dim,
                                        act=nn.Tanh(),
                                        dropout_p=0.3)

    def forward(self,encoded_input):
        feature, adj, mask, sorted_lengths, sort_order, sentences_per_doc  = encoded_input
        graph_layer_output = self.graph_layer(feature, adj, mask) # bs, ml , dim
        texting_output = self.readout_layer(graph_layer_output,mask)
        unsort_order = torch.LongTensor(unsort(sort_order)).to(self.device)
        unsorted_sent_repr = texting_output.index_select(0, unsort_order)
        return unsorted_sent_repr, sentences_per_doc

    def get_output_dim(self):
        return self.output_dim