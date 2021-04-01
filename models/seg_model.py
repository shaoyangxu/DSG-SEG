import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import *
from models.sequence_sr import s_model,zero_state
from models.treelstm_sr import t_model_treelstm
from models.freq_sr import f_model
from models.latent_sr import l_model
from models.bert_sr import b_model
from models.texting import g_model_text_ing
from models.DSG_SEG import g_model_DSG_SEG

class seg_model(nn.Module):
    def __init__(self, sr_choose,encoder_input_dim, args):
        super(seg_model, self).__init__()
        self.device = args.device
        self.pool_method = args.pool_method
        self.encoder_input_dim = encoder_input_dim
        self.hidden_dim=args.hidden_dim
        self.num_layers=args.num_layers
        self.sr_choose = sr_choose
        if sr_choose == "f_model":
            self.sr_model = f_model(encoder_input_dim,args)
        elif sr_choose == "l_model":
            self.sr_model = l_model(args)
        elif sr_choose == "s_model":
            self.sr_model = s_model(encoder_input_dim, args)
        elif sr_choose == "t_model":
            assert args.tr_choose in ['balanced', 'left', 'right']
            self.sr_model = t_model_treelstm(encoder_input_dim, args)
        elif sr_choose == "g_model":
            if args.gr_choose == 'texting':
                self.sr_model = g_model_text_ing(encoder_input_dim,args)
            elif args.gr_choose == 'DSG_SEG':
                self.sr_model = g_model_DSG_SEG(encoder_input_dim, args)
        elif sr_choose == 'b_model':
            self.sr_model = b_model(encoder_input_dim, args)

        if args.sr_choose != "random_baseline":
            self.sl_input_dim = self.sr_model.get_output_dim()
        else:
            self.sl_input_dim = 300
        self.sentence_lstm = nn.LSTM(input_size=self.sl_input_dim,
                                     hidden_size=self.hidden_dim,
                                     num_layers=self.num_layers,
                                     batch_first=True,
                                     dropout=0,
                                     bidirectional=True)
        self.h2s = nn.Linear(self.hidden_dim*2 , 2)

        self.criterion = nn.CrossEntropyLoss()


    def pad_document(self, d, max_document_length):
        d_length = d.size()[0]
        v = d.unsqueeze(0).unsqueeze(0)
        padded = F.pad(v, (0, 0, 0, max_document_length - d_length))  # (1, 1, max_length, 300)
        shape = padded.size()
        return padded.view(shape[2], 1, shape[3])  # (max_length, 1, 300)

    def forward(self, encoded_input):
        batch_size = encoded_input[0]
        unsorted_sent_repr, sentences_per_doc = self.sr_model(encoded_input[1:])
        index = 0
        encoded_documents = []
        for sentences_count in sentences_per_doc:
            end_index = index + sentences_count
            encoded_documents.append(unsorted_sent_repr[index: end_index, :])
            index = end_index

        doc_sizes = [doc.size()[0] for doc in encoded_documents]
        max_doc_size = np.max(doc_sizes)
        ordered_document_idx = np.argsort(doc_sizes)[::-1]
        ordered_doc_sizes = sorted(doc_sizes)[::-1]
        ordered_documents = [encoded_documents[idx] for idx in ordered_document_idx]
        padded_docs = [self.pad_document(d, max_doc_size) for d in ordered_documents]
        docs_tensor = torch.cat(padded_docs, 1)
        packed_docs = pack_padded_sequence(docs_tensor, ordered_doc_sizes)
        sentence_lstm_output, _ = self.sentence_lstm(packed_docs, zero_state(self, batch_size=batch_size, device=self.device))
        padded_x, _ = pad_packed_sequence(sentence_lstm_output)  # (max sentence len, batch, 256)

        doc_outputs = []
        for i, doc_len in enumerate(ordered_doc_sizes):
            doc_outputs.append(padded_x[0:doc_len - 1, i, :])  # -1 to remove last prediction ---> wiki_loader.py

        unsorted_doc_outputs = [doc_outputs[i] for i in unsort(ordered_document_idx)]
        sentence_outputs = torch.cat(unsorted_doc_outputs, 0)

        x = self.h2s(sentence_outputs)
        return x


