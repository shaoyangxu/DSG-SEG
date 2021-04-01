from pathlib import Path
from torch.utils.data import Dataset
from data.data_utils import *
from utils import *
import os
import random
from parameters import create_parser

def clean_paragraph(paragraph):
    cleaned_paragraph= paragraph.replace("'' ", " ").replace(" 's", "'s").replace("``", "").strip('\n')
    return cleaned_paragraph

def read_choi_file(args,filtered_num, all_num, filtered_document_num, all_document_num,path, encoder=None,length_filter=10, sent_num_filter=10):
    seperator = '=========='
    with Path(path).open('r') as f:
        raw_text = f.read()
    paragraphs = [clean_paragraph(p) for p in raw_text.strip().split(seperator) if len(p) > 5 and p != "\n"]
    targets = []
    new_text = []
    adjs = []
    for paragraph in paragraphs:
        sentences = [s for s in paragraph.split('\n') if len(s.split()) > 0]
        if sentences:
            sentences_count =0
            # This is the number of sentences in the paragraph and where we need to split.
            for sentence in sentences:
                words = extract_sentence_words(sentence)
                if (len(words) <= 1): # remove sentences whose length <= 1 otherwise treelstm cant run
                    continue
                sentences_count +=1
                # length_filter
                if len(words) > length_filter:
                    words = words[:length_filter]
                    filtered_num += 1
                if args.sr_choose == 'g_model' and args.gr_choose == 'texting':
                    sentence_words, adj = build_ing_graph(words, args)
                    adjs.append(adj)
                if encoder:
                    new_text.append(encoder.tokenize(words))
                else:
                    new_text.append(words)
            all_num += sentences_count
            targets.extend([0] * (sentences_count - 1))
            if sentences_count != 0:
                targets.append(1)  # segment
    if args.sr_choose == 'g_model' and args.gr_choose == 'texting':
        return_dict, filtered_document_num, all_document_num = generate_ing_return_dict(new_text, adjs, targets, path,
                                                                                        all_document_num,
                                                                                        filtered_document_num,
                                                                                        sent_num_filter)
    elif args.sr_choose == 'g_model' and args.gr_choose in 'DSG_SEG':
        return_dict, filtered_document_num, all_document_num = generate_return_dict(new_text, targets, path,
                                                                                    all_document_num,
                                                                                    filtered_document_num,
                                                                                    sent_num_filter)
        return_dict = generate_gcn_return_dict(return_dict, args)
    else:
        return_dict, filtered_document_num, all_document_num = generate_return_dict(new_text, targets, path,
                                                                                    all_document_num,
                                                                                    filtered_document_num,
                                                                                    sent_num_filter)
    return return_dict,filtered_num, all_num,filtered_document_num, all_document_num

def get_cache_path(folder,split):

    if split == None:
        cache_file_path = folder / 'paths_cache'
    elif split == 'train':
        cache_file_path = folder / 'train_paths_cache'
    elif split == 'dev':
        cache_file_path = folder / 'dev_paths_cache'
    elif split == 'test':
        cache_file_path = folder / 'test_paths_cache'
    else:
        cache_file_path = None
    return cache_file_path

def cache_filenames(folder,split):
    files = Path(folder).glob('*/*/*.ref')
    files = list(files) # otherwise:TypeError: object of type 'generator' has no len()
    cache_path = get_cache_path(folder,split)
    if split == None: # write all
        with cache_path.open("w+") as f:
            for file in files:
                f.write(str(file) + u'\n')
    elif split == 'train': # write train dev and test
        all_document_num = len(files)
        all_idx = [i for i in range(all_document_num)]
        random.shuffle(all_idx)
        train_num = int(all_document_num / 10 * 8)
        valid_num = int(all_document_num / 10 * 1)
        test_num = int(all_document_num / 10 * 1)
        train_cache_path = get_cache_path(folder,"train")
        with train_cache_path.open("w+") as f:
            for i in range(train_num):
                train_idx = all_idx[i]
                f.write(str(files[train_idx]) + u'\n')
        valid_cache_path = get_cache_path(folder,"dev")
        with valid_cache_path.open("w+") as f:
            for i in range(train_num, train_num+valid_num):
                valid_idx = all_idx[i]
                f.write(str(files[valid_idx]) + u'\n')
        test_cache_path = get_cache_path(folder,"test")
        with test_cache_path.open("w+") as f:
            for i in range(train_num+valid_num, train_num+valid_num+test_num):
                test_idx = all_idx[i]
                f.write(str(files[test_idx]) + u'\n')


class ChoiDataSet(Dataset):
    def __init__(self,
                 root = None,
                 args=None,
                 encoder = None,
                 split = None, # None, train, valid, test
                 length_filter = 40000,
                 sent_num_filter = 60000,
                 local_rank=-1):
        logger = get_logger(args)
        root_path = Path(root)
        cache_path = get_cache_path(root_path,split)
        if not cache_path.exists():
            cache_filenames(root_path,split)
            self.textfiles = cache_path.read_text().splitlines()
        else:
            self.textfiles = cache_path.read_text().splitlines()
        if len(self.textfiles) == 0:
            raise RuntimeError('Found 0 images in subfolders of: {}'.format(root))
        self.data = []
        # sentences
        filtered_num = 0
        all_num = 0
        # documents
        filtered_document_num = 0
        all_document_num = 0
        if split == None:
            rank_logger_info(logger, local_rank,"Reading all choi data for test")
        else:
            rank_logger_info(logger, local_rank, "Reading choi %s data" % (split))
        for textfile in self.textfiles:
            return_dict, filtered_num, all_num, filtered_document_num, all_document_num = read_choi_file(args,filtered_num,
                                                                                                        all_num,
                                                                                                        filtered_document_num,
                                                                                                        all_document_num,
                                                                                                        textfile,
                                                                                                        encoder=encoder,
                                                                                                        length_filter=length_filter,
                                                                                                        sent_num_filter=sent_num_filter)
            if return_dict == None:
                continue
            elif type(return_dict) == dict:
                self.data.append(return_dict)
            else: # list
                self.data.extend(return_dict)
        # print("one choi example:",self.data[0])
        if all_num != 0:
            rank_logger_info(logger, local_rank, "all sentences num:%d\tfiltered num:%d\tfiltered percent:%f"%(all_num, filtered_num,filtered_num/all_num))
        if all_document_num != 0:
            rank_logger_info(logger, local_rank, "all documents num:%d\tfiltered num:%d\tfiltered percent:%f" % (all_document_num, filtered_document_num, filtered_document_num / all_document_num))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    args = create_parser()
    print(args)
    logger = setup_logger(args.logger_name, os.path.join(args.checkpoint_dir, 'train.log'))
    choidataset = ChoiDataSet(args.choi_path, args)