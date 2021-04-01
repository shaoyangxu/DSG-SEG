import json
from pathlib2 import Path
from data.wiki_loader import get_sections
import wiki_utils
import os
from text_manipulation import extract_sentence_words
import torch
class wiki_document:
    def __init__(self, wiki_path, bert_path, ING_GCN_path):
        self.wiki_path = wiki_path
        self.bert_path = bert_path
        self.ING_GCN_path = ING_GCN_path
        self.document_dict = {} # {"path1":{"golden":[...,"====",...,"===="],"DSG_SEG":[同],"bert":[同]}, "path2":...}
        self.get_golden_result()
        self.get_bert_result()
        self.get_ING_GCN_result()

    def get_golden_result(self):
        self.paths = Path(self.wiki_path).read_text().splitlines()
        for path in self.paths:
            data, targets = self.read_wiki_file(path)
            self.document_dict[path] = {} # {'/p300/wiki/wiki_727/test/58/46/96/5846960':{}}
            self.document_dict[path]["data"] = data
            self.document_dict[path]["targets"] = targets
            self.document_dict[path]["document_len"] = len(data)
            self.document_dict[path]["seg_num"] = sum(targets)
            self.document_dict[path]["golden"] = [] # # {'/p300/wiki/wiki_727/test/58/46/96/5846960':{'golden':[]}}
            for idx,sentence in enumerate(data):
                self.document_dict[path]["golden"].append(sentence)
                if targets[idx] == 1:
                    self.document_dict[path]["golden"].append("="*10)


    def get_bert_result(self):
        self.get_result(self.bert_path)

    def get_ING_GCN_result(self):
        self.get_result(self.ING_GCN_path)

    def get_result(self, path):
        model_name = None
        if "bert" in path:
            model_name = "bert"
        elif "DSG_SEG" in path:
            model_name = "DSG_SEG"
        with open(path, "rb") as f:
            result_dict = json.load(f)
        for path in self.paths:
            self.document_dict[path][model_name] = []
            pk = float(result_dict[path]["pk"]) # ok
            self.document_dict[path]["{}_pk".format(model_name)] = pk
            pred = result_dict[path]["pred"] # pred segments
            pred_list = pred.split(" ")
            for idx, one_pred in enumerate(pred_list[:-2]):
                if one_pred == "|":
                    self.document_dict[path][model_name].append("="*10)
                else:
                    self.document_dict[path][model_name].append(self.document_dict[path]["data"][int(one_pred)])



    def read_wiki_file(self, path,
                       high_granularity=False,
                       remove_preface_segment=True,
                       ignore_list=True,
                       remove_special_tokens=True):
        data = []
        targets = []
        all_sections = get_sections(path, high_granularity)
        required_sections = all_sections[1:] if remove_preface_segment and len(all_sections) > 0 else all_sections
        required_non_empty_sections = [section for section in required_sections if len(section) > 0 and section != "\n"]
        for section in required_non_empty_sections:
            sentences = section.split('\n')
            if sentences:
                targets_flag = 0
                for sentence in sentences:
                    is_list_sentence = wiki_utils.get_list_token() + "." == str(sentence)
                    if ignore_list and is_list_sentence:
                        continue
                    sentence_words = extract_sentence_words(sentence, remove_special_tokens=remove_special_tokens)
                    if (len(sentence_words) <= 1):
                        continue
                    data.append(" ".join(sentence_words))
                    targets_flag += 1
                if data:
                    targets.extend([0] * (targets_flag - 1))
                    if targets_flag != 0:
                        targets.append(1)  # segment
        data = data[:-1]
        targets = targets[:-1]
        return data, targets


if __name__ == "__main__":
    os.makedirs(r"/root/DSG-SEG/sampleanalysis",exist_ok=True)
    saved_path = r"/root/textsegment/sampleanalysis/WIKI_DOCUMENT.pkl"
    if os.path.exists(saved_path):
        WIKI_DOCUMENT = torch.load(saved_path)
    else:
        WIKI_DOCUMENT = wiki_document(
            wiki_path = r"/p300/wiki/wiki_727/test/random_paths",
            bert_path= r"/root/textsegment/checkpoints/bert/0_wiki_bert_b_model_9fa59240ac158d3ae21c0dcf25da8e81/wiki_best_result.json",
            ING_GCN_path = r"...",
        )
        torch.save(WIKI_DOCUMENT, saved_path)
