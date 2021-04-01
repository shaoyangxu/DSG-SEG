import torch.nn as nn
import torch
class f_model(nn.Module):
    def __init__(self, encoder_input_dim, args):
        super(f_model, self).__init__()
        self.args = args
        self.output_dim = encoder_input_dim
        self.device = self.args.device

    # def pad(self, s, max_length):
    #     pad_lst = [self.encoder.model.pad_token_id] * (max_length - s.size(0))
    #     s = torch.LongTensor(s.tolist() + pad_lst).to(self.device)
    #     return s.unsqueeze(0) # (1, max_len)

    def forward(self, encoded_input):
        embeded_big_tensor, sentences_per_doc = encoded_input
        sent_repr = torch.sum(embeded_big_tensor, dim=1) # max_length, dim
        sent_repr[:, 0] = 0 # pad_idx
        return sent_repr, sentences_per_doc

    def get_output_dim(self):
        return self.output_dim
