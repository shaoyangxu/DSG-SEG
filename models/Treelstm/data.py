import jsonlines
import numpy as np
import torch
from torch.utils.data import Dataset

class Vocab(object):

    def __init__(self, vocab_dict, add_pad, add_unk):
        self._vocab_dict = vocab_dict.copy()
        self._reverse_vocab_dict = dict()
        if add_pad:
            self.pad_word = '<pad>'
            self.pad_id = len(self._vocab_dict)
            self._vocab_dict[self.pad_word] = self.pad_id
        if add_unk:
            self.unk_word = '<unk>'
            self.unk_id = len(self._vocab_dict)
            self._vocab_dict[self.unk_word] = self.unk_id
        for w, i in self._vocab_dict.items():
            self._reverse_vocab_dict[i] = w

    @classmethod
    def from_file(cls, path, add_pad=True, add_unk=True, max_size=None):
        vocab_dict = dict()
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_size and i >= max_size:
                    break
                word = line.strip().split()[0]
                vocab_dict[word] = len(vocab_dict)
                vocab_dict[word] = len(vocab_dict)
        return cls(vocab_dict=vocab_dict, add_pad=add_pad, add_unk=add_unk)

    def word_to_id(self, word):
        if hasattr(self, 'unk_id'):
            return self._vocab_dict.get(word, self.unk_id)
        return self._vocab_dict[word]

    def id_to_word(self, id_):
        if hasattr(self, 'unk_word'):
            return self._reverse_vocab_dict.get(id_, self.unk_word)
        return self._reverse_vocab_dict[id_]

    def has_word(self, word):
        return word in self._vocab_dict

    def __len__(self):
        return len(self._vocab_dict)


class SentenceDataset(Dataset):

    def __init__(self, word_vocab, data_path, label_vocab, max_length, lower, min_length=2):
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.lower = lower
        self._max_length = max_length # 64
        self._min_length = min_length # 2
        self._data = []
        failed_to_parse = 0
        with jsonlines.open(data_path, 'r') as reader:
            for obj in reader:  # obj:{"label":1,"sentence":"..."}
                try:
                    converted = self._convert_obj(obj)
                    if converted:
                        self._data.append(converted)
                    else:
                        failed_to_parse += 1
                except ValueError:
                    failed_to_parse += 1
                except AttributeError:
                    failed_to_parse += 1
        print('Failed to parse {:d} instances'.format(failed_to_parse))

    def _convert_obj(self, obj):
        sentence = obj['sentence']
        if self.lower:
            sentence = sentence.lower()
        word_ids = [self.word_vocab.word_to_id(w) for w in sentence.split()]
        length = len(word_ids)
        if length > self._max_length or length < self._min_length:
            return None
        label = self.label_vocab.word_to_id(obj['label'])
        if 'constituency_tree_encoding' in obj:
            mask_ids = self.tree_encoding_to_mask_ids(obj['constituency_tree_encoding'])
        else:
            mask_ids = None
        if mask_ids is not None and len(mask_ids) != length - 1:
            return None
        return word_ids, sentence, length, mask_ids, label

    def pad_sentence(self, data):
        max_length = max(len(d) for d in data)
        padded = [d + [self.word_vocab.pad_id] * (max_length - len(d))
                  for d in data]
        return padded

    @staticmethod
    def pad_mask(data):
        max_length = max(len(d) for d in data)
        padded = [d + [0] * (max_length - len(d)) for d in data]
        return np.array(padded)

    @staticmethod
    def make_one_hot_gold_mask(gold_mask_ids):
        masks = list()
        for num_classes in range(gold_mask_ids.shape[1], 1, -1):
            i = gold_mask_ids.shape[1] - num_classes
            indices = torch.LongTensor(gold_mask_ids[:, i])
            mask = SentenceDataset.convert_to_one_hot(indices, num_classes).float()
            masks.append(mask)
        return masks

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item]

    def collate(self, batch):
        words_batch, raw_sentences_batch, length_batch, mask_ids_batch, label_batch = list(zip(*batch))
        sentences = torch.LongTensor(self.pad_sentence(words_batch))
        lengths = torch.LongTensor(length_batch)
        try:
            masks = self.make_one_hot_gold_mask(self.pad_mask(mask_ids_batch))
        except TypeError:
            masks = None
        labels = torch.LongTensor(label_batch)
        return {'sentences': sentences, 'lengths': lengths, 'masks': masks, 'labels': labels,
                'raw_sentences': raw_sentences_batch}

    @staticmethod
    def tree_encoding_to_mask_ids(tree_encoding):
        items = [int(x) for x in tree_encoding.strip().split(',')]
        sentence_length = len(items) // 3 + 1 # (sentence_length - 1) * 3 = len(items)
        curr_sent = [x for x in range(sentence_length)]
        mask_ids = list()
        assert 3 * (sentence_length - 1) == len(items)
        for i in range(sentence_length - 1):
            left_node = items[i * 3]
            right_node = items[i * 3 + 1]
            father_node = items[i * 3 + 2]
            left_index = curr_sent.index(left_node)
            assert curr_sent[left_index + 1] == right_node
            curr_sent = curr_sent[:left_index] + [father_node] + curr_sent[left_index + 2:]
            mask_ids.append(left_index)
        return mask_ids

    @staticmethod
    def convert_to_one_hot(indices, num_classes):
        batch_size = indices.shape[0]
        indices = indices.unsqueeze(1)
        one_hot = indices.new(batch_size, num_classes).zero_().scatter_(1, indices, 1)
        return one_hot

