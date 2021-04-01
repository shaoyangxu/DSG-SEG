import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import unsort
def zero_state(module, batch_size, device):
    # * 2 is for the two directions
    return torch.zeros(module.num_layers * 2, batch_size, module.hidden_dim).to(device), \
           torch.zeros(module.num_layers * 2, batch_size, module.hidden_dim).to(device)

class s_model(nn.Module):
    def __init__(self, encoder_input_dim, args):
        super(s_model, self).__init__()
        self.input_dim = encoder_input_dim
        self.num_layers = args.bilstm_nl
        self.hidden_dim = args.bilstm_hd
        self.bidirectional = args.bilstm_bd
        self.device = args.device
        # self.pooling_method = args.pool_method
        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            dropout=0,
                            bidirectional=True) # batch_first=True

    # def pad(self, s, max_length):
    #     pad_lst = [self.encoder.model.pad_token_id] * (max_length - s.size(0))
    #     s = torch.LongTensor(s.tolist() + pad_lst).to(self.device)
    #     return s.unsqueeze(1)

    def forward(self,encoded_input):
        embeded_big_tensor,sorted_lengths,sort_order,sentences_per_doc=encoded_input
        packed_tensor = pack_padded_sequence(embeded_big_tensor, sorted_lengths) # batch_first=True
        batch_size = packed_tensor.batch_sizes[0]
        s = zero_state(self, batch_size, self.device)
        encoded_sentences,_ = self.lstm(packed_tensor,s)
        padded_output, lengths = pad_packed_sequence(encoded_sentences)  # (batch, max sentence len,256) ,batch_first=True
        maxes = torch.zeros(batch_size, padded_output.size(2)).to(self.device)
        for i in range(batch_size):
            maxes[i, :] = torch.max(padded_output[:lengths[i],i,:], 0)[0]
        unsort_order = torch.LongTensor(unsort(sort_order)).to(self.device)
        unsorted_sent_repr = maxes.index_select(0, unsort_order)
        return unsorted_sent_repr, sentences_per_doc


    def get_output_dim(self):
        if self.bidirectional:
            return self.hidden_dim * 2
        else:
            return self.hidden_dim