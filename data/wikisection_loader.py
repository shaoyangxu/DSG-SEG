import re
from pathlib import Path
from torch.utils.data import Dataset
from utils import *
from data.data_utils import *
import os
from parameters import create_parser


segment_seperator = "=========="
def get_seperator_foramt():
    seperator_fromat = segment_seperator + ".*\n?"
    return seperator_fromat

def get_scections_from_text(txt):
    sections_to_keep_pattern = get_seperator_foramt()
    all_sections = re.split(sections_to_keep_pattern, txt)
    non_empty_sections = [s for s in all_sections if len(s) > 0]
    return non_empty_sections

def clean_section(section):
    cleaned_section = section.strip('\n')
    return cleaned_section

def get_sections(path):
    file = open(str(path), "r", encoding='utf-8')
    raw_content = file.read()
    file.close()
    clean_txt = raw_content.encode('utf-8').decode('utf-8').strip()
    sections = [clean_section(s) for s in get_scections_from_text(clean_txt)]
    return sections

def remove_last_one(data, targets):
    return data[:-1], targets[:-1]

def read_section_file(args,filtered_num, all_num, filtered_document_num, all_document_num,
                      path, encoder=None, length_filter=20, sent_num_filter=10, remove_preface_segment=True,
                      ignore_list=False, remove_special_tokens=False,return_as_sentences=False, only_letters=False):
    data = []
    adjs = []
    targets = []
    all_sections = get_sections(path)
    required_sections = all_sections[1:] if remove_preface_segment and len(all_sections) > 0 else all_sections
    required_non_empty_sections = [section for section in required_sections if len(section.split()) > 0 and section != "\n"]
    for section in required_non_empty_sections:
        sentences = section.split('\n')
        if sentences:
            targets_flag = 0
            for sentence in sentences:
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
                    targets.append(1) # segment
    if args.sr_choose == 'g_model' and args.gr_choose == 'texting':
        return_dict, filtered_document_num, all_document_num = generate_ing_return_dict(data, adjs, targets, path, all_document_num,filtered_document_num,sent_num_filter)
    elif args.sr_choose == 'g_model' and args.gr_choose in 'DSG_SEG':
        return_dict, filtered_document_num, all_document_num = generate_return_dict(data, targets, path, all_document_num,filtered_document_num,sent_num_filter)
        return_dict = generate_gcn_return_dict(return_dict,args)
    else:
        return_dict, filtered_document_num, all_document_num = generate_return_dict(data, targets, path, all_document_num,filtered_document_num,sent_num_filter)
    return return_dict,filtered_num, all_num,filtered_document_num, all_document_num

def get_cache_path(folder):
    cache_file_path = folder / 'paths_cache'
    return cache_file_path

def cache_filenames(folder):
    files = Path(folder).glob('*.ref')
    cache_path = get_cache_path(folder)
    with cache_path.open("w+") as f:
        for file in files:
            f.write(str(file) + u'\n')

class WikiSectioniDataSet(Dataset):
    def __init__(self,
                 root = None,
                 args=None,
                 encoder = None,
                 split = 'train',
                 length_filter = 40000,
                 sent_num_filter = 60000,
                 local_rank=-1):
        logger = get_logger(args)
        root_disease = Path(root+"/en_disease_"+split)
        root_city = Path(root+"/en_city_"+split)
        cache_disease_path = get_cache_path(root_disease)
        cache_city_path = get_cache_path(root_city)

        if not cache_disease_path.exists():
            cache_filenames(root_disease)
            cache_filenames(root_city)
            self.textfiles = cache_disease_path.read_text().splitlines() + cache_city_path.read_text().splitlines()
        else:
            self.textfiles = cache_disease_path.read_text().splitlines() + cache_city_path.read_text().splitlines()
        if len(self.textfiles) == 0:
            raise RuntimeError('Found 0 images in subfolders of: {}'.format('lalala'))
        self.data = []
        # sentences
        filtered_num = 0
        all_num = 0
        # documents
        filtered_document_num = 0
        all_document_num = 0
        rank_logger_info(logger, local_rank, "Reading wikisection %s data"%(split))
        for textfile in self.textfiles:
            return_dict, filtered_num, all_num, filtered_document_num, all_document_num = read_section_file(args, filtered_num,
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
        # print("one wikisection example:",self.data[0])
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
    choidataset = WikiSectioniDataSet(args.wikisection_path, args, split="validation") # train validation test