import torch.nn as nn
import torch
from models.GGNN.layers import *

class g_model_DSG_SEG(nn.Module):
    def __init__(self,encoder_input_dim,args):
        super(g_model_DSG_SEG,self).__init__()
        self.input_dim = encoder_input_dim
        self.output_dim = args.DSG_SEG_output_dim
        self.device = args.device
        self.layers = []
        self.graph_layer = GraphLayer(args,
                                      input_dim=self.input_dim,
                                      output_dim=self.output_dim,
                                      act=nn.Tanh(),
                                      dropout_p=0.3)

    def forward(self,encoded_input):
        doc_feature, adj, mask, sentences_per_doc = encoded_input
        batch_size = doc_feature.shape[0]
        graph_layer_output = self.graph_layer(doc_feature, adj, mask) # bs, max_node_size , dim
        sent_repr = []
        for i in range(batch_size):
            sent_repr.append(graph_layer_output[i, :sentences_per_doc[i], :])
        sent_repr = torch.cat(sent_repr,0)
        return sent_repr, sentences_per_doc

    def get_output_dim(self):
        return self.output_dim