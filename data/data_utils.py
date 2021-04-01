import scipy.sparse as sp
import numpy as np
from math import log
from nltk.tokenize import RegexpTokenizer
import nltk

# from wiki_thresholds.py
min_section_char_count = 20
min_sentence_in_section = 2
max_section_formulas_count = 3
max_section_code_snipet_count = 2
min_valid_section_count = 3 # This includes preface section
min_words_in_sentence = 2
min_valid_section_percentage = 0.2
max_list_in_section_percentage = 0.5
max_words_in_sentence_with_backslash_n = 45

def remove_last_one(data, targets):
    return data, targets[:-1]

def remove_ing_last_one(data, adjs, targets):
    return data,adjs, targets[:-1]

def generate_return_dict(data,targets,path, all_document_num,filtered_document_num,sent_num_filter):
    data, targets = remove_last_one(data, targets)
    sent_num = len(data)
    if sent_num <= 2:
        return None, filtered_document_num, all_document_num
    assert len(targets) == len(data) - 1

    all_document_num += 1
    path_flag = -1
    if sent_num > sent_num_filter:
        filtered_document_num += 1
        return_dict = []
        start_idx = 0
        while start_idx <= sent_num - 1:
            path_flag += 1
            end_idx = start_idx + sent_num_filter
            if end_idx >= sent_num:
                end_idx = sent_num
            tmp_data = data[start_idx: end_idx] if end_idx == sent_num else data[
                                                                            start_idx: end_idx + 1]  # include one more
            tmp_targets = targets[start_idx: end_idx]
            tmp_path = path + "_|_{}".format(path_flag)
            tmp_return_dict = {
                "sentences": tmp_data,
                "targets": tmp_targets,
                "path": tmp_path
            }
            if len(tmp_return_dict["sentences"]) >= 2:  #
                return_dict.append(tmp_return_dict)
            start_idx = end_idx
    else:
        return_dict = {
            "sentences": data,
            "targets": targets,
            "path": path
        }
    return return_dict, filtered_document_num, all_document_num




def generate_ing_return_dict(data,adjs,targets,path, all_document_num,filtered_document_num,sent_num_filter):
    data, adjs,targets = remove_ing_last_one(data, adjs,targets)
    sent_num = len(data)
    if sent_num <= 2:
        return None, filtered_document_num, all_document_num
    assert len(targets) == len(data) - 1
    assert len(data) == len(adjs)
    all_document_num += 1
    path_flag = -1
    if sent_num > sent_num_filter:
        filtered_document_num += 1
        return_dict = []
        start_idx = 0
        while start_idx <= sent_num - 1:
            path_flag += 1
            end_idx = start_idx + sent_num_filter
            if end_idx >= sent_num:
                end_idx = sent_num
            tmp_data = data[start_idx: end_idx] if end_idx == sent_num else data[
                                                                            start_idx: end_idx + 1]  # include one more
            tmp_adj = adjs[start_idx: end_idx] if end_idx == sent_num else adjs[
                                                                            start_idx: end_idx + 1]  # include one more
            tmp_targets = targets[start_idx: end_idx]
            tmp_path = path + "_|_{}".format(path_flag)
            tmp_return_dict = {
                "sentences": tmp_data,
                "adjs": tmp_adj,
                "targets": tmp_targets,
                "path": tmp_path
            }
            if len(tmp_return_dict["sentences"]) >= 2:  #
                return_dict.append(tmp_return_dict)
            start_idx = end_idx
    else:
        return_dict = {
            "sentences": data,
            "adjs": adjs,
            "targets": targets,
            "path": path
        }
    return return_dict, filtered_document_num, all_document_num



def build_ing_graph(doc_words, args):
    """
    https://github.com/CRIPAC-DIG/TextING/blob/master/build_graph.py
    """
    window_size = args.gnn_window_size
    doc_len = len(doc_words)
    doc_vocab = []
    for w in doc_words:
        if w not in doc_vocab:
            doc_vocab.append(w)
    doc_nodes = len(doc_vocab)
    doc_word_id_map = {}
    for j in range(doc_nodes):
        doc_word_id_map[doc_vocab[j]] = j
    # sliding windows
    windows = []
    if doc_len <= window_size:
        windows.append(doc_words)
    else:
        for j in range(doc_len - window_size + 1):
            window = doc_words[j: j + window_size]
            windows.append(window)
    num_window = len(windows)
    word_window_freq = {}
    for window in windows:
        appeared = set()
        for i in range(len(window)):
            if window[i] in appeared:
                continue
            if window[i] in word_window_freq:
                word_window_freq[window[i]] += 1
            else:
                word_window_freq[window[i]] = 1
            appeared.add(window[i])
    word_pair_count = {}
    for window in windows:
        for p in range(1, len(window)):
            for q in range(0, p):
                word_p = window[p]
                word_p_id = doc_word_id_map[word_p]
                word_q = window[q]
                word_q_id = doc_word_id_map[word_q]
                if word_p_id == word_q_id:
                    continue
                word_pair_key = (word_p_id, word_q_id)
                # word co-occurrences as weights
                if word_pair_key in word_pair_count:
                    word_pair_count[word_pair_key] += 1.
                else:
                    word_pair_count[word_pair_key] = 1.
                # bi-direction
                word_pair_key = (word_q_id, word_p_id)
                if word_pair_key in word_pair_count:
                    word_pair_count[word_pair_key] += 1.
                else:
                    word_pair_count[word_pair_key] = 1.
    row = []
    col = []
    weight = []
    feature = []
    # preprocess of TextING, not used
    # for key in word_pair_count:
    #     p = key[0]
    #     q = key[1]
    #     row.append(p)  # p
    #     col.append(q)  # q
    #     weight.append(word_pair_count[key] if weighted_graph else 1.)
    # adj = sp.csr_matrix((weight, (row, col)), shape=(doc_nodes, doc_nodes))
    for key in word_pair_count:
        p = key[0]
        q = key[1]
        count = word_pair_count[key]
        word_freq_p = word_window_freq[doc_vocab[p]]
        word_freq_q = word_window_freq[doc_vocab[q]]
        pmi = log((1.0 * count / num_window) /
                  (1.0 * word_freq_p * word_freq_q / (num_window * num_window)))
        if pmi <= 0:
            continue
        row.append(p)
        col.append(q)
        weight.append(pmi)
    adj = sp.csr_matrix((weight, (row, col)), shape=(doc_nodes, doc_nodes))
    for k, v in sorted(doc_word_id_map.items(), key=lambda x: x[1]):
        feature.append(k)
    return feature, normalize_adj(adj)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1)) # D
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

def generate_gcn_return_dict(return_dict,args):
    if return_dict == None:
        return return_dict
    elif type(return_dict) == dict:
        sentences,targets,path = return_dict["sentences"], return_dict["targets"], return_dict["path"]
        sentences, feature, adj = build_gcn_graph(sentences,args)
        output = {"sentences":sentences,
                  "feature":feature,
                  "adj":adj,
                  "targets":targets,
                  "path":path}
        return output
    else:  # list
        output = []
        for tmp_return_dict in return_dict:
            output.append(generate_gcn_return_dict(tmp_return_dict,args))
        return output

def build_gcn_graph(sentences,args):
    window_size = args.gnn_window_size
    # weighted_graph = args.gnn_weighted_graph
    sent_num = len(sentences)
    doc_word_id_map = {}
    doc_vocab = []
    doc_len_lst = []
    for sentence in sentences:
        doc_len_lst.append(len(sentence))
        for word in sentence:
            if word not in doc_vocab:
                doc_vocab.append(word)

    doc_nodes = len(doc_vocab)

    for j in range(doc_nodes):
        doc_word_id_map[doc_vocab[j]] = j

    doc_nodes += sent_num
    new_sentences = []
    for idx, sentence in enumerate(sentences):
        new_sentences.append([doc_word_id_map[w] for w in sentence])

    windows = []
    for sentence in sentences:
        length = len(sentence)
        if length <= window_size:
            windows.append(sentence)
        else:
            for j in range(length - window_size + 1):
                window = sentence[j:j+window_size]
                windows.append(window)
    num_window = len(windows)
    word_window_freq = {}
    for window in windows:
        appeared = set()
        for i in range(len(window)):
            if window[i] in appeared:
                continue
            if window[i] in word_window_freq:
                word_window_freq[window[i]] += 1
            else:
                word_window_freq[window[i]] = 1
            appeared.add(window[i])

    word_pair_count = {}
    for window in windows:
        for p in range(1, len(window)):
            for q in range(0, p):
                word_p = window[p]
                word_p_id = doc_word_id_map[word_p]
                word_q = window[q]
                word_q_id = doc_word_id_map[word_q]
                if word_p_id == word_q_id:
                    continue
                word_pair_key = (word_p_id, word_q_id)
                # word co-occurrences as weights
                if word_pair_key in word_pair_count:
                    word_pair_count[word_pair_key] += 1.
                else:
                    word_pair_count[word_pair_key] = 1.
                # bi-direction
                word_pair_key = (word_q_id, word_p_id)
                if word_pair_key in word_pair_count:
                    word_pair_count[word_pair_key] += 1.
                else:
                    word_pair_count[word_pair_key] = 1.

    word_doc_list = {}
    word_doc_freq = {}
    for sentence_idx,sentence in enumerate(sentences):
        appeared = set()
        for word in sentence:
            if word in appeared:
                continue
            if word in word_doc_list:
                word_doc_list[word].append(sentence_idx)
            else:
                word_doc_list[word] = [sentence_idx]
            appeared.add(word)
    for word, doc_list in word_doc_list.items():
        word_doc_freq[word] = len(doc_list)

    doc_word_freq = {}
    for sentence_idx,sentence in enumerate(sentences):
        for word in sentence:
            word_idx = doc_word_id_map[word]
            doc_word_str = (sentence_idx,word_idx)
            if doc_word_str in doc_word_freq:
                doc_word_freq[doc_word_str] += 1
            else:
                doc_word_freq[doc_word_str] = 1
    row = []
    col = []
    weight = []
    feature = []
    if args.tfidf:
        for key in doc_word_freq:
            freq = doc_word_freq[key]
            sentence_idx = key[0]
            word_idx = key[1]
            row.append(sentence_idx)
            col.append(word_idx + sent_num) # ***
            idf = log(1.0 * doc_len_lst[sentence_idx] / word_doc_freq[doc_vocab[word_idx]])
            tf_idf = freq * idf
            weight.append(tf_idf)
    else: # -TF-IDF
        for key in doc_word_freq:
            sentence_idx = key[0]
            word_idx = key[1]
            row.append(sentence_idx)
            col.append(word_idx + sent_num) # ***
            weight.append(1)
    if args.pmi:
        for key in word_pair_count:
            p = key[0]
            q = key[1]
            count = word_pair_count[key]
            word_freq_p = word_window_freq[doc_vocab[p]]
            word_freq_q = word_window_freq[doc_vocab[q]]
            pmi = log((1.0 * count / num_window) /
                      (1.0 * word_freq_p * word_freq_q / (num_window * num_window)))
            if pmi <= 0:
                continue
            row.append(sent_num + p)
            col.append(sent_num + q)
            weight.append(pmi)
    else: # -PMI
        for key in word_pair_count:
            p = key[0]
            q = key[1]
            row.append(sent_num + p)
            col.append(sent_num + q)
            weight.append(1)
    adj = sp.csr_matrix((weight, (row, col)), shape=(doc_nodes, doc_nodes))
    for k, v in sorted(doc_word_id_map.items(), key=lambda x: x[1]):
        feature.append(k)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    return new_sentences, feature, sp.csr_matrix(normalize_adj(adj+sp.eye(adj.shape[0])))



segment_seperator = "========"


def get_segment_seperator(level,name):
    return segment_seperator + "," + str(level) + "," +name



def get_seperator_foramt(levels = None):
    level_format = '\d' if levels == None else '['+ str(levels[0]) + '-' + str(levels[1]) + ']'
    seperator_fromat = segment_seperator + ',' + level_format + ",.*?\."
    return seperator_fromat



def is_seperator_line(line):
    return line.startswith(segment_seperator)

def get_segment_level(seperator_line):
    return int (seperator_line.split(',')[1])

def get_segment_name(seperator_line):
    return seperator_line.split(',')[2]

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
sentence_tokenizer = None
missing_stop_words = set(['of', 'a', 'and', 'to'])
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


def get_punkt():
    global sentence_tokenizer
    if sentence_tokenizer:
        return sentence_tokenizer

    try:
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    except exceptions.LookupError:
        nltk.download('punkt')
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    sentence_tokenizer = tokenizer
    return sentence_tokenizer


def split_sentence_with_list(sentence):

    list_pattern = "\n" + get_list_token() + "."
    if sentence.endswith( list_pattern ):
        splited_sentence = [str for str in sentence.split("\n" + get_list_token() + ".") if
                            len(str) > 0]
        splited_sentence.append(get_list_token() + ".")
        return splited_sentence
    else:
        return [sentence]

def split_sentece_colon_new_line(sentence):

    splited_sentence = sentence.split(":\n")
    if (len(splited_sentence) == 1):
        return splited_sentence
    new_sentences = []
    # -1 . not to add ":" to last sentence
    for i in range(len(splited_sentence) - 1):
        if (len(splited_sentence[i]) > 0):
            new_sentences.append(splited_sentence[i] + ":")
    if (len(splited_sentence[-1]) > 0):
        new_sentences.append(splited_sentence[-1])
    return new_sentences

def split_long_sentences_with_backslash_n(max_words_in_sentence,sentences, doc_id):
    new_sentences = []
    for sentence in sentences:
        sentence_words = extract_sentence_words(sentence)
        if len(sentence_words) > max_words_in_sentence:
            splitted_sentences = sentence.split('\n')
            new_sentences.extend(splitted_sentences )
        else:
            new_sentences.append(sentence)
    return new_sentences

def split_sentences(text, doc_id):
    sentences = get_punkt().tokenize(text)
    senteces_list_fix = []
    for sentence in sentences:
        seplited_list_sentence = split_sentence_with_list(sentence)
        senteces_list_fix.extend(seplited_list_sentence)

    sentence_colon_fix = []
    for sentence in senteces_list_fix:
        splitted_colon_sentence =  split_sentece_colon_new_line(sentence)
        sentence_colon_fix.extend(splitted_colon_sentence)

    sentences_without_backslash_n = split_long_sentences_with_backslash_n(max_words_in_sentence_with_backslash_n, sentence_colon_fix, doc_id)

    ret_sentences = []
    for sentence in sentences_without_backslash_n:
        ret_sentences.append(sentence.replace('\n',' '))
    return ret_sentences


def clean_paragraph(paragraph):
    cleaned_paragraph= paragraph.replace("'' ", " ").replace(" 's", "'s").replace("``", "").strip('\n')
    return cleaned_paragraph