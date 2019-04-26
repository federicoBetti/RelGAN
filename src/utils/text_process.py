# coding=utf-8
from typing import Tuple, Dict, Any

import nltk


# text tokens to code strings
def text_to_code(tokens, dictionary, seq_len):
    code_str = ""
    eof_code = len(dictionary)  # used to filled in the blank to make up a sentence with seq_len
    for sentence in tokens:
        index = 0
        for word in sentence:
            code_str += (str(dictionary[word]) + ' ')
            index += 1
        while index < seq_len:
            code_str += (str(eof_code) + ' ')
            index += 1
        code_str += '\n'
    return code_str


# code tokens to text strings
def code_to_text(codes, dictionary):
    paras = ""
    eof_code = len(dictionary)
    for line in codes:
        numbers = map(int, line)
        for number in numbers:
            if number == eof_code:
                continue
            paras += (dictionary[str(number)] + ' ')
        paras += '\n'
    return paras


# tokenlize the file
def get_tokenized(file):
    tokenized = list()
    with open(file) as raw:
        for text in raw:
            text = nltk.word_tokenize(text.lower())
            tokenized.append(text)
    return tokenized


# get word set
def get_word_list(tokens: list) -> list:
    word_set = list()
    for sentence in tokens:
        for word in sentence:
            word_set.append(word)
    return list(set(word_set))


# get word_index_dict and index_word_dict
def get_dict(word_set: list) -> Tuple[Dict, Dict]:
    word_index_dict = dict()
    index_word_dict = dict()
    index = 0
    for word in word_set:
        assert isinstance(word, str)
        if word.startswith("cucu"):
            print(word)
        word_index_dict[word] = str(index)
        index_word_dict[str(index)] = word
        index += 1
    return word_index_dict, index_word_dict


def text_precess(train_text_loc, test_text_loc=None) -> Tuple[int, int, Any]:
    """
    Get sequence length and dict size \n
    :param train_text_loc: train file
    :param test_text_loc: test file
    :return: sequence length of the longest sentences, dict size (how many different words), dict from word to index
    """
    train_tokens = get_tokenized(train_text_loc)
    # todo stemming or lemmatization?
    if test_text_loc is None:
        test_tokens = list()
    else:
        test_tokens = get_tokenized(test_text_loc)
    word_set = get_word_list(train_tokens + test_tokens)
    [word_index_dict, index_word_dict] = get_dict(word_set)

    if test_text_loc is None:
        sequence_len = len(max(train_tokens, key=len))
    else:
        sequence_len = max(len(max(train_tokens, key=len)), len(max(test_tokens, key=len)))

    # with open(oracle_file, 'w') as outfile:
    #     outfile.write(text_to_code(tokens, word_index_dict, seq_len))

    return sequence_len, len(word_index_dict) + 1, word_index_dict