import torch
import torch.nn as nn
from models.GGNN.inits import *

class GraphConvolution(nn.Module):
    def __init__( self, input_dim, \
                        output_dim, \
                        act = None, \
                        dropout_p = 0., \
                        bias=True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_p = dropout_p
        self.weight = glorot(self.input_dim, self.output_dim)
        if bias:
            self.b = uniform_bias(self.weight, self.output_dim)

        self.act = act
        self.dropout = nn.Dropout(self.dropout_p)


    def forward(self, x, adj, mask):
        x = self.dropout(x)
        pre_sup = torch.matmul(x, self.weight) +self.b# bs, node_size, input_dim x input_dim, output_dim = bs, node_size, output_dim
            # pre_sup = x.mm(self.weight)
        pre_sup = mask * pre_sup
        # out = adj.mm(pre_sup)
        out = torch.bmm(adj, pre_sup) # bs, node_size, node_size x node_size, output_dim
        out = self.act(out)
        return out

class gru_unit(nn.Module):
    def __init__(self, output_dim, act, dropout_p):
        super(gru_unit,self).__init__()
        self.output_dim = output_dim
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(self.dropout_p)
        self.act = act
        self.z0_weight = glorot(self.output_dim, self.output_dim)
        self.z1_weight = glorot(self.output_dim, self.output_dim)
        self.r0_weight = glorot(self.output_dim, self.output_dim)
        self.r1_weight = glorot(self.output_dim, self.output_dim)
        self.h0_weight = glorot(self.output_dim, self.output_dim)
        self.h1_weight = glorot(self.output_dim, self.output_dim)
        self.z0_bias = uniform_bias(self.z0_weight, self.output_dim)
        self.z1_bias = uniform_bias(self.z1_weight, self.output_dim)
        self.r0_bias = uniform_bias(self.r0_weight, self.output_dim)
        self.r1_bias = uniform_bias(self.r1_weight, self.output_dim)
        self.h0_bias = uniform_bias(self.h0_weight, self.output_dim)
        self.h1_bias = uniform_bias(self.h1_weight, self.output_dim)

    def forward(self,adj, x, mask):
        # adj: bs,ml,ml  | x: bs,ml,dim
        adj = self.dropout(adj)
        a = torch.matmul(adj, x)
        # updata gate
        z0 = torch.matmul(a, self.z0_weight) + self.z0_bias
        z1 = torch.matmul(x, self.z1_weight) + self.z1_bias
        z = torch.sigmoid(z0+z1)
        # reset gate
        r0 = torch.matmul(a, self.r0_weight) + self.r0_bias
        r1 = torch.matmul(x, self.r1_weight) + self.r1_bias
        r = torch.sigmoid(r0+r1)
        # update embeddings
        h0 = torch.matmul(a, self.h0_weight) + self.h0_bias
        h1 = torch.matmul(r*x, self.h1_weight) + self.h1_bias
        h = self.act(mask * (h0 + h1))
        return h*z + x*(1-z)


class GraphLayer(nn.Module):
    """Graph layer."""
    def __init__(self, args,
                      input_dim,
                      output_dim,
                      act=nn.Tanh(),
                      dropout_p = 0.):
        super(GraphLayer, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru_step = self.args.texting_gru_step
        self.gru_unit = gru_unit(self.output_dim, self.act, self.dropout_p)
        # self.dropout
        self.encode_weight = glorot(self.input_dim, self.output_dim)
        self.encode_bias = uniform_bias(self.encode_weight, self.output_dim)

    def forward(self, feature, adj, mask):
        feature = self.dropout(feature)
        # encode inputs
        encoded_feature = torch.matmul(feature, self.encode_weight) +self.encode_bias
        output = mask * self.act(encoded_feature)
        # convolve
        for _ in range(self.gru_step):
            output = self.gru_unit(adj, output, mask)
        return output

class ReadoutLayer(nn.Module):
    """Graph Readout Layer."""
    def __init__(self, args,
                 input_dim,
                 output_dim,
                 act=nn.ReLU(),
                 dropout_p=0.):
        super(ReadoutLayer, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(self.dropout_p)
        self.att_weight = glorot(self.input_dim, 1)
        self.emb_weight = glorot(self.input_dim, self.input_dim)
        self.mlp_weight = glorot(self.input_dim, self.output_dim)
        self.att_bias = uniform_bias(self.att_weight,1)
        self.emb_bias = uniform_bias(self.emb_weight,self.input_dim)
        self.mlp_bias = uniform_bias(self.mlp_weight,self.input_dim)

    def forward(self,x,mask):
        # soft attention
        att = torch.sigmoid(torch.matmul(x, self.att_weight)+self.att_bias)
        emb = self.act(torch.matmul(x, self.emb_weight)+self.emb_bias)
        # graph summation
        N = torch.sum(mask, dim=1)
        M = (mask - 1) * 1e9
        # classification
        g = mask * att * emb
        g = torch.sum(g, dim=1)/N + torch.max(g+M,dim=1)[0]
        # g = self.dropout(g)
        return g
