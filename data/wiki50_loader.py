from pathlib import Path
from torch.utils.data import Dataset
from utils import *
from data.data_utils import *
import re
import os
from parameters import create_parser
segment_seperator = "========"

def get_seperator_foramt(levels = None):
    level_format = '\d' if levels == None else '['+ str(levels[0]) + '-' + str(levels[1]) + ']'
    seperator_fromat = segment_seperator + ',' + level_format + ",.*?\."
    return seperator_fromat


def clean_section(section):
    cleaned_section = section.strip('\n')
    return cleaned_section


def get_scections_from_text(txt, high_granularity=True):
    sections_to_keep_pattern = get_seperator_foramt() if high_granularity else get_seperator_foramt((1, 2))
    if not high_granularity:
        # if low granularity required we should flatten segments within segemnt level 2
        pattern_to_ommit = get_seperator_foramt((3, 999))
        txt = re.sub(pattern_to_ommit, "", txt)
        #delete empty lines after re.sub()
        sentences = [s for s in txt.strip().split("\n") if len(s) > 0 and s != "\n"]
        txt = '\n'.join(sentences).strip('\n')
    all_sections = re.split(sections_to_keep_pattern, txt)
    non_empty_sections = [s for s in all_sections if len(s) > 0]
    return non_empty_sections


def get_sections(path, high_granularity=True):
    file = open(str(path), "r")
    raw_content = file.read()
    file.close()
    clean_txt = raw_content.encode('utf-8').decode('utf-8').strip()
    sections = [clean_section(s) for s in get_scections_from_text(clean_txt, high_granularity)]
    return sections

def read_wiki50_file(args,filtered_num, all_num, filtered_document_num, all_document_num,
                     path, logger, encoder=None, length_filter=20, sent_num_filter=10, remove_preface_segment=True, ignore_list=False, remove_special_tokens=False,
                   return_as_sentences=False, high_granularity=False, only_letters=False):
    data = []
    adjs = []
    targets = []
    all_sections = get_sections(path, high_granularity)
    required_sections = all_sections[1:] if remove_preface_segment and len(all_sections) > 0 else all_sections
    required_non_empty_sections = [section for section in required_sections if len(section) > 0 and section != "\n"]
    for section in required_non_empty_sections:
        sentences = section.split('\n')
        if sentences:
            targets_flag = 0
            for sentence in sentences:
                is_list_sentence = get_list_token() + "." == str(sentence)
                if ignore_list and is_list_sentence:
                    continue
                if not return_as_sentences:
                    sentence_words = extract_sentence_words(sentence, remove_special_tokens=remove_special_tokens)
                    if (len(sentence_words) <= 1):  # remove sentences whose length <= 1 otherwise treelstm cant run
                        continue
                    # length_filter
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
                else:  # for the annotation. keep sentence as is.
                    if (only_letters):
                        sentence = re.sub('[^a-zA-Z0-9 ]+', '', sentence)
                        data.append(sentence)
                    else:
                        data.append(sentence)
            if data:
                all_num += targets_flag
                targets.extend([0]*(targets_flag-1))
                if targets_flag != 0:
                    targets.append(1)  # segment
    if args.sr_choose == 'g_model' and args.gr_choose == 'texting':
        return_dict, filtered_document_num, all_document_num = generate_ing_return_dict(data, adjs, targets, path, all_document_num,filtered_document_num,sent_num_filter)
    elif args.sr_choose == 'g_model' and args.gr_choose in 'DSG_SEG':
        return_dict, filtered_document_num, all_document_num = generate_return_dict(data, targets, path, all_document_num,filtered_document_num,sent_num_filter)
        return_dict = generate_gcn_return_dict(return_dict,args)
    else:
        return_dict, filtered_document_num, all_document_num = generate_return_dict(data, targets, path, all_document_num,filtered_document_num,sent_num_filter)
    return return_dict, filtered_num, all_num,filtered_document_num, all_document_num

def get_cache_path(folder):
    cache_file_path = folder / 'paths_cache'
    return cache_file_path

def cache_filenames(folder):
    files = Path(folder).glob('*')
    files_lst = []
    for file in files:
        files_lst.append(file)
    cache_path = get_cache_path(folder)
    with cache_path.open("w+") as f:
        for file in files_lst:
            f.write(str(file) + u'\n')

class Wiki50DataSet(Dataset):
    def __init__(self,
                 root = None,
                 args=None,
                 encoder = None,
                 length_filter = 40,
                 sent_num_filter = 60,
                 local_rank=-1):
        logger = get_logger(args)
        root_path = Path(root)
        cache_path = get_cache_path(root_path)
        if not cache_path.exists():
            cache_filenames(root_path)
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
        rank_logger_info(logger, local_rank, "Reading wiki50 data")
        for textfile in self.textfiles:
            return_dict, filtered_num, all_num, filtered_document_num, all_document_num = read_wiki50_file(args,filtered_num,
                                                                                                        all_num,
                                                                                                        filtered_document_num,
                                                                                                        all_document_num,
                                                                                                        textfile,
                                                                                                        logger = logger,
                                                                                                        encoder=encoder,
                                                                                                        length_filter=length_filter,
                                                                                                        sent_num_filter=sent_num_filter)
            if return_dict == None:
                continue
            elif type(return_dict) == dict:
                self.data.append(return_dict)
            else: # list
                self.data.extend(return_dict)
        # print("one wiki50 example:",self.data[0])
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
    choidataset = Wiki50DataSet(args.wiki50_path, args)