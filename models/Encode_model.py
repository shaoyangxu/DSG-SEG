import torch.nn as nn
import torch
import numpy as np
from models.Treelstm.utils import *
import torch.nn.functional as F
class Encode_model(nn.Module):
    def __init__(self, encoder, args):
        super(Encode_model , self).__init__()
        self.encoder = encoder
        self.args = args
        self.device = self.args.device
        self.sr_choose = self.args.sr_choose
        self.tr_choose = self.args.tr_choose
        self.gr_choose = self.args.gr_choose
        if self.sr_choose == "t_model":
            if args.tr_choose == 'balanced':
                self.generate_mask_function = generate_balance_masks
            elif args.tr_choose == 'left':
                self.generate_mask_function = generate_left_branch_masks
            elif args.tr_choose == 'right':
                self.generate_mask_function = generate_right_branch_masks
    def pad(self, s, max_length):
        s_length = s.size()[0]
        v = s.unsqueeze(0).unsqueeze(0)
        padded = F.pad(v, (0, 0, 0, max_length - s_length))  # (1, 1, max_length, 300)
        shape = padded.size()
        return padded.view(shape[2], 1, shape[3])  # (max_length, 1, 300)

    def pad_s_model(self, s, max_length):
        pad_lst = [self.encoder.model.pad_token_id] * (max_length - s.size(0))
        s = torch.LongTensor(s.tolist() + pad_lst).to(self.device)
        return s.unsqueeze(1)

    def pad_l_model(self, s, max_length):
        pad_lst = [self.encoder.model.pad_token_id] * (max_length - s.size(0))
        s = torch.LongTensor(s.tolist() + pad_lst).to(self.device)
        return s.unsqueeze(0) # (1, max_len)

    def preprocess_treelstm(self, sentences_lst):
        lengths = [sentence.size(0) for sentence in sentences_lst] # (length,)
        max_length = max(lengths)
        padded_sentences_lst = [sentence.tolist() + [self.encoder.model.pad_token_id] * (max_length - sentence.size(0))
                                 for sentence in sentences_lst]
        masks = self.generate_mask_function(lengths)
        padded_sentences_tensor = torch.LongTensor(padded_sentences_lst).to(self.device)
        lengths = torch.LongTensor(lengths).to(self.device)
        return padded_sentences_tensor, lengths, masks

    def preprocess_gcn_adj(self,max_length,adjs):
        mask = torch.zeros((len(adjs), max_length, 1)).to(self.device)
        for i,adj in enumerate(adjs):
            pad = max_length - adj.shape[0]
            # print("adj",adj.shape)
            # print("pad",pad)
            padded_adj = F.pad(adj, (0,pad,0,pad))
            mask[i,:adj.shape[0],:] = 1.
            adjs[i] = padded_adj.unsqueeze(0) # 1, max_length, max_length
        return adjs, mask # double

    def preprocess_gcn_feature(self, max_length,features):
        for i,feature in enumerate(features):
            # feature: node_size, dim  --> padded_feature: max_length, dim
            pad = max_length - feature.shape[0]
            padded_feature = F.pad(feature, (0,0,0,pad))
            features[i] = padded_feature.unsqueeze(0)
        return features

    def forward(self, batch):
        batch_size = len(batch)
        if self.sr_choose == "s_model":
            sentences_per_doc = []
            all_batch_sentences = []
            for document in batch:
                all_batch_sentences.extend(document)
                sentences_per_doc.append(len(document))

            lengths = [s.size()[0] for s in all_batch_sentences]
            sort_order = np.argsort(lengths)[::-1]
            sorted_sentences = [all_batch_sentences[i] for i in sort_order]
            sorted_lengths = [s.size()[0] for s in sorted_sentences]

            max_length = max(lengths)
            padded_sentences = [self.pad_s_model(s, max_length) for s in sorted_sentences]
            big_tensor = torch.cat(padded_sentences, 1)
            embeded_big_tensor = self.encoder(big_tensor)
            return batch_size,embeded_big_tensor,sorted_lengths,sort_order,sentences_per_doc
        elif self.sr_choose in ["f_model","l_model"]:
            sentences_per_doc = []
            all_batch_sentences = []
            for document in batch:
                all_batch_sentences.extend(document)
                sentences_per_doc.append(len(document))
            lengths = [s.size()[0] for s in all_batch_sentences]
            max_length = max(lengths)
            padded_sentences = [self.pad_l_model(s, max_length) for s in all_batch_sentences]
            big_tensor = torch.cat(padded_sentences, 0)  # (batch size, max_length)
            embeded_big_tensor = self.encoder(big_tensor) #  (max_length, batch size, dim)
            return batch_size,embeded_big_tensor,sentences_per_doc
        elif self.sr_choose == "t_model" and self.tr_choose in ["balanced","left","right"]:
            sentences_per_doc = []
            all_batch_sentences = []
            for document in batch:
                all_batch_sentences.extend(document)
                sentences_per_doc.append(len(document))
            sentence, lengths, tree_masks = self.preprocess_treelstm(all_batch_sentences)
            embeded_sentence = self.encoder(sentence)
            return batch_size,embeded_sentence, lengths, tree_masks, sentences_per_doc
        elif self.sr_choose == "b_model":
            sentences_per_doc = []
            all_batch_sentences = []
            for document in batch:
                all_batch_sentences.extend(document)
                sentences_per_doc.append(len(document))
            lengths = [s.size()[0] for s in all_batch_sentences]
            sort_order = np.argsort(lengths)[::-1]
            sorted_sentences = [all_batch_sentences[i] for i in sort_order]
            sorted_lengths = [s.size()[0] for s in sorted_sentences]
            max_length = max(lengths)
            padded_sentences = [self.pad_l_model(s, max_length) for s in sorted_sentences]
            big_tensor = torch.cat(padded_sentences, 0)  # (batch size,max_length)
            embeded_big_tensor = self.encoder(big_tensor)
            return batch_size,embeded_big_tensor,sorted_lengths,sort_order,sentences_per_doc
        elif self.sr_choose == "g_model" and self.gr_choose == "texting":
            # batch: [[[tensor(sent_num1,), tensor(sent_num1,sent_num1)], ...], [doc2], [doc3], []]
            sentences_per_doc = []
            all_batch_features = []
            all_batch_adjs = []
            for document in batch:
                sentences_per_doc.append(len(document))
                for sentence in document:
                    feature, adj = sentence
                    all_batch_features.append(feature)
                    all_batch_adjs.append(adj)
            lengths = [s.size()[0] for s in all_batch_features]
            sort_order = np.argsort(lengths)[::-1]
            sorted_features = [all_batch_features[i] for i in sort_order]
            sorted_adjs = [all_batch_adjs[i] for i in sort_order]
            sorted_lengths = [s.size()[0] for s in sorted_features]
            max_length = max(lengths)

            padded_features = [self.pad_l_model(s, max_length) for s in sorted_features]
            features = torch.cat(padded_features, 0) # batch_size, max_length

            padded_adjs, mask = self.preprocess_gcn_adj(max_length,sorted_adjs)
            adjs = torch.cat(padded_adjs, 0).to(self.device)

            embeded_features = self.encoder(features)
            return batch_size, embeded_features, adjs, mask, sorted_lengths, sort_order, sentences_per_doc
        elif self.sr_choose == "g_model" and self.gr_choose == "DSG_SEG":
            # batch: [[sentences, feature, adj], [doc2], [doc3], []]
            # sentences: [[feature_idx1, feature_idx2, feature_idx3, ...],[s2],[s3],...]
            sentences_per_doc = []
            words_per_doc = []
            nodes_per_doc = []
            all_batch_sentences = []
            all_batch_features = []
            all_batch_adjs = []
            for document in batch:
                sentences, feature, adj = document
                sentences_per_doc.append(len(sentences))
                words_per_doc.append(feature.size(0))
                nodes_per_doc.append(len(sentences) + feature.size(0))
                all_batch_sentences.append(sentences)
                all_batch_features.append(feature)
                all_batch_adjs.append(adj)
            max_length = max(words_per_doc)
            padded_features = [self.pad_l_model(s, max_length) for s in all_batch_features]
            features = torch.cat(padded_features, 0)  # batch_size, max_word_num
            embeded_features = self.encoder(features) # batch_size, max_word_num, dim
            doc_feature_lst = []
            for i in range(batch_size):
                doc_feature = []
                feature = embeded_features[i][:words_per_doc[i]] # word_num, dim
                for sentence in all_batch_sentences[i]:
                    doc_feature.append(torch.max(feature[sentence,:], 0)[0].unsqueeze(0))
                doc_feature.append(feature)
                doc_feature = torch.cat(doc_feature, 0)
                doc_feature_lst.append(doc_feature)
            max_node_size = max(nodes_per_doc)
            padded_doc_features = self.preprocess_gcn_feature(max_node_size, doc_feature_lst)
            doc_features = torch.cat(padded_doc_features, 0).to(self.device)  # batch_size, max_node_size, dim
            padded_adjs, mask = self.preprocess_gcn_adj(max_node_size, all_batch_adjs)
            adjs = torch.cat(padded_adjs, 0).to(self.device) # batch_size, max_node_size, max_node_size
            return batch_size, doc_features, adjs, mask, sentences_per_doc
