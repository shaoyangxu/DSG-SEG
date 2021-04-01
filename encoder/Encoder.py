import torch.nn as nn
from encoder.one_hot_encoder import one_hot_model
from encoder.glove_encoder import glove_model
from encoder.randn_encoder import randn_model
from encoder.w2v_encoder import w2v_model
from encoder.bert_encoder import bert_model
from utils import *

class Encoder(nn.Module):
    def __init__(self, args, etype, encoder_fine_tune, glove_path=None, w2v_path=None):
        super(Encoder, self).__init__()
        self.args = args
        if etype == "one-hot":
            self.model = one_hot_model(glove_path, args, encoder_fine_tune)
        elif etype == 'glove':
            self.model = glove_model(glove_path, args, encoder_fine_tune)
        elif etype == "randn":
            self.model = randn_model(glove_path, args, encoder_fine_tune)
        elif etype == 'w2v':
            self.model = w2v_model(w2v_path, args, encoder_fine_tune)
        elif etype == 'bert':
            self.model = bert_model(args, encoder_fine_tune)
        self.etype=etype
        self.pad_token_id = self.model.pad_token_id

    def tokenize(self,sentence): # get_subword_indices=False
        if type(sentence) is list:
            sentence = ' '.join(sentence)
        return self.model.tokenize(sentence)

    def forward(self, sentence_lst,just_last_layer=True):
        return self.model(sentence_lst)

    def get_input_dim(self):
        return self.model.get_word_dim()