from nltk.tokenize import RegexpTokenizer
from torch.utils.data import Dataset
from utils import *
from data.data_utils import *
import os
from parameters import create_parser
missing_stop_words = set(['of', 'a', 'and', 'to'])

def get_list_token():
    return "***LIST***"


def get_formula_token():
    return "***formula***"


def get_codesnipet_token():
    return "***codice***"


def get_special_tokens():
    special_tokens = []
    special_tokens.append(get_list_token())
    special_tokens.append(get_formula_token())
    special_tokens.append(get_codesnipet_token())
    return special_tokens

words_tokenizer = None
def get_words_tokenizer():
    global words_tokenizer
    if words_tokenizer:
        return words_tokenizer
    words_tokenizer = RegexpTokenizer(r'\w+')
    return words_tokenizer

def extract_sentence_words(sentence, remove_missing_emb_words = False,remove_special_tokens = False):
    if (remove_special_tokens):
        for token in get_special_tokens():
            # Can't do on sentence words because tokenizer delete '***' of tokens.
            sentence = sentence.replace(token, "")
    tokenizer = get_words_tokenizer()
    sentence_words = tokenizer.tokenize(sentence)
    if remove_missing_emb_words:
        sentence_words = [w for w in sentence_words if w not in missing_stop_words]
    return sentence_words

def get_sections(text_path, segment_path):
    text_file = open(str(text_path), "r", encoding="utf-8")
    texts = text_file.read().strip().split("\n")
    text_file.close()
    segment_file = open(str(segment_path), "r", encoding="utf-8")
    segments = segment_file.read().strip().split("\n")
    segment_file.close()
    all_sections = [] # all document
    sections = [] # one document
    one_section = [] # one section
    document_num = int(segments[0].strip().split(",")[0])
    seg_title = segments[0].strip().split(",")[2]
    for idx, text in enumerate(texts):
        text = text.split("\x01")[1]
        segment = segments[idx]
        if len(segment.strip().split(",")) == 3:
            n_doc_num,n_sent_num,n_seg_title = segment.strip().split(",")
        else:
            n_doc_num,n_sent_num,n_seg_title1,n_seg_title2 = segment.strip().split(",")
            n_seg_title = n_seg_title1 + "," + n_seg_title2
        n_doc_num,n_sent_num = int(n_doc_num),int(n_sent_num)
        if n_seg_title != seg_title and n_doc_num == document_num:
            sections.append(one_section)
            one_section = []
            seg_title = n_seg_title
        elif n_doc_num != document_num:
            sections.append(one_section)
            all_sections.append(sections)
            one_section = []
            sections = []
            document_num = n_doc_num
            seg_title = n_seg_title
        one_section.append(text)
    all_sections.append(sections)
    return all_sections


def read_wikicities_file(args, filtered_num, all_num, filtered_document_num, all_document_num,
                        text_path, segment_path,encoder=None,length_filter=10, sent_num_filter=10):
    all_sections_of_all_doc = get_sections(text_path, segment_path)
    all_return_dic = []
    for idx,all_sections in enumerate(all_sections_of_all_doc):
        path = text_path + "/document_" + str(idx)
        data = []
        adjs = []
        targets = []
        for sentences in all_sections:
            if sentences:
                targets_flag = 0
                for sentence in sentences:
                    sentence_words = extract_sentence_words(sentence)
                    if (len(sentence_words) <= 1):  # remove sentences whose length <= 1 otherwise treelstm cant run
                        continue
                    if len(sentence_words) > length_filter:
                        sentence_words = sentence_words[:length_filter]
                        filtered_num += 1
                    if args.sr_choose == 'g_model' and args.gr_choose == 'texting':
                        sentence_words, adj = build_ing_graph(sentence_words, args)
                        adjs.append(adj)
                    if encoder:
                        data.append(encoder.tokenize(sentence_words))
                    else:
                        data.append(sentence_words)
                    targets_flag += 1
                if data:
                    all_num += targets_flag
                    targets.extend([0]*(targets_flag-1))
                    if targets_flag != 0:
                        targets.append(1)  # segment
        if args.sr_choose == 'g_model' and args.gr_choose == 'texting':
            return_dict, filtered_document_num, all_document_num = generate_ing_return_dict(data, adjs, targets, path,
                                                                                            all_document_num,
                                                                                            filtered_document_num,
                                                                                            sent_num_filter)
        elif args.sr_choose == 'g_model' and args.gr_choose in ['textgcn', 'ING_GCN','S_ING_GCN']:
            return_dict, filtered_document_num, all_document_num = generate_return_dict(data, targets, path,
                                                                                        all_document_num,
                                                                                        filtered_document_num,
                                                                                        sent_num_filter)
            return_dict = generate_gcn_return_dict(return_dict, args)
        else:
            return_dict, filtered_document_num, all_document_num = generate_return_dict(data, targets, path,
                                                                                        all_document_num,
                                                                                        filtered_document_num,
                                                                                        sent_num_filter)
        if return_dict == None:
            continue
        elif type(return_dict) == dict:
            all_return_dic.append(return_dict)
        else:  # list
            all_return_dic.extend(return_dict)
    return all_return_dic, filtered_num, all_num, filtered_document_num, all_document_num

class WikiCitiesDataSet(Dataset):
    def __init__(self,
                 root = None,
                 args = None,
                 encoder = None,
                 length_filter = 40,
                 sent_num_filter = 60,
                 local_rank=-1):
        logger = get_logger(args)
        self.text_path = root + "/wikicities.text"
        self.segment_path = root + "/wikicities.segmenttitles"
        rank_logger_info(logger, local_rank, "Reading wikicities data")
        # sentences
        filtered_num = 0
        all_num = 0
        # documents
        filtered_document_num = 0
        all_document_num = 0
        self.data,filtered_num, all_num, filtered_document_num, all_document_num = read_wikicities_file(args,filtered_num,
                                                                                                        all_num,
                                                                                                        filtered_document_num,
                                                                                                        all_document_num,
                                                                                                        text_path= self.text_path,
                                                                                                        segment_path=self.segment_path,
                                                                                                        encoder=encoder,
                                                                                                        length_filter=length_filter,
                                                                                                        sent_num_filter=sent_num_filter)
        # print("one wikicities example:",self.data[0])

        if all_num != 0:
            rank_logger_info(logger, local_rank, "all sentences num:%d\tfiltered num:%d\tfiltered percent:%f"%(all_num, filtered_num,filtered_num/all_num))
        if all_document_num != 0:
            rank_logger_info(logger, local_rank, "all documents num:%d\tfiltered num:%d\tfiltered percent:%f"%(all_document_num, filtered_document_num, filtered_document_num / all_document_num))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    args = create_parser()
    print(args)
    logger = setup_logger(args.logger_name, os.path.join(args.checkpoint_dir, 'train.log'))
    wikielementsdataset = WikiCitiesDataSet(args.wikicities_path, args)
