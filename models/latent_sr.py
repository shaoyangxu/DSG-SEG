import torch.nn as nn
import torch
from sklearn.decomposition import PCA

class l_model(nn.Module):
    def __init__(self, args):
        super(l_model, self).__init__()
        self.args = args
        self.device = self.args.device
        self.output_dim = 300
        self.pca = PCA(n_components = self.output_dim)

    # def pad(self, s, max_length):
    #     pad_lst = [self.encoder.model.pad_token_id] * (max_length - s.size(0))
    #     s = torch.LongTensor(s.tolist() + pad_lst).to(self.device)
    #     return s.unsqueeze(0) # (1, max_len)

    def pca_function(self, sent_repr):
        sents_num = sent_repr.size(0) # (y, dim)
        min_samples_num = self.output_dim + 1
        if sents_num < min_samples_num:
            pad_tensor = torch.zeros((min_samples_num - sents_num, sent_repr.size(1))).to(self.device) # (x,dim)
            sent_repr = torch.cat([sent_repr, pad_tensor], dim=0)
        sent_repr = sent_repr.cpu()
        pca_out = self.pca.fit_transform(sent_repr)[:sents_num]
        return torch.tensor(pca_out).to(self.device)

    def forward(self, encoded_input):
        # sentences_per_doc = []
        # all_batch_sentences = []
        # for document in batch:
        #     all_batch_sentences.extend(document)
        #     sentences_per_doc.append(len(document))
        # lengths = [s.size()[0] for s in all_batch_sentences]
        # max_length = max(lengths)
        # padded_sentences = [self.pad(s, max_length) for s in all_batch_sentences]
        # big_tensor = torch.cat(padded_sentences, 0)  # (batch size, max_length)
        # embeded_big_tensor = self.encoder(big_tensor) #  (max_length, batch size, dim)
        embeded_big_tensor,sentences_per_doc = encoded_input
        sent_repr = torch.sum(embeded_big_tensor, dim=1) # max_length, dim
        sent_repr[:, 0] = 0 # max_length, dim
        return self.pca_function(sent_repr).float(), sentences_per_doc

    def get_output_dim(self):
        return self.output_dim