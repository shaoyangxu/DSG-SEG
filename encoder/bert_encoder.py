import torch.nn as nn
import torch
from transformers import BertTokenizer, BertModel
from utils import *
class bert_model(nn.Module):
    def __init__(self, args, encoder_fine_tune):
        super(bert_model, self).__init__()
        self.args = args
        self.local_rank = args.local_rank
        self.encoder_fine_tune = encoder_fine_tune
        self.model_size = args.model_size
        self.cased = args.cased
        if not self.cased:
            do_lower_case = True
            self.model_name = 'bert' + "-" + self.model_size + '-uncased'
        else:
            do_lower_case = False
            self.model_name = 'bert' + "-" + self.model_size + '-cased'

        self.embedding = self.get_bert_embedding()
        if encoder_fine_tune:
            for param in self.parameters():
                param.requires_grad = True
        else:
            for param in self.parameters():
                param.requires_grad = False

        self.device = args.device
        self.tokenizer = BertTokenizer.from_pretrained(
                    self.model_name, do_lower_case=do_lower_case)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.num_layers = self.embedding.config.num_hidden_layers + 1
        self.hidden_size = self.embedding.config.hidden_size
        self.weighing_params = nn.Parameter(torch.ones(self.num_layers))

    def get_bert_embedding(self):
        logger = get_logger(self.args)
        rank_logger_info(logger, self.local_rank, 'Reading bert embedding.')
        embedding = BertModel.from_pretrained(
                    self.model_name, output_hidden_states = True)
        return embedding

    def tokenize(self, inp):
        token_ids = self.tokenizer.encode(inp, add_special_tokens=True)
        return token_ids


    def forward(self, inp, just_last_layer=False):
        input_mask = (inp != self.pad_token_id).float().to(self.device)
        if not self.encoder_fine_tune:
            with torch.no_grad():
                output_bert = self.embedding(
                    inp, attention_mask=input_mask, output_hidden_states=True)  # B x L x E
                last_hidden_state = output_bert.last_hidden_state
                encoded_layers = output_bert.hidden_states
        else:
            output_bert = self.embedding(
                inp, attention_mask=input_mask, output_hidden_states=True)
            last_hidden_state = output_bert.last_hidden_state
            encoded_layers = output_bert.hidden_states

        if just_last_layer:
            output = last_hidden_state
        else:
            wtd_encoded_repr = 0
            soft_weight = nn.functional.softmax(self.weighing_params, dim=0)
            for i in range(self.num_layers):
                wtd_encoded_repr += soft_weight[i] * encoded_layers[i]
            output = wtd_encoded_repr

        return output


    def get_word_dim(self):
        return self.hidden_size
