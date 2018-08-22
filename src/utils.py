import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

def soup2dict(sentence_nodes):
    """
    Input: a soup object, e.g. soup.find_all("sentence")
    Output: a list of dictionaries, contains id, text, aspect terms
    """
    sentences = []
    i = 0
    for n in sentence_nodes:
        i += 1
        sentence = {}
        aspect_term = []
        sentence['id'] = i
        sentence['text'] = n.find('text').string
        if n.find('Opinions'):
            for c in n.find('Opinions').contents:
                if c.name == 'Opinion':
                    if c['target'] not in aspect_term:
                        aspect_term.append(c['target'])

        sentence['aspect'] = aspect_term
        sentences.append(sentence)

    return sentences


def split2words(s_text):
    """Split string with white and prereserve the punctuation
    Input:
        s_text: string, a sentence, e.g. Judging from previous posts this used to be a good place, but not any longer.
    Output:
        words: a list of words, e.g. ['judging', 'from', 'previous', 'posts', 'this', 'used', 'to', 'be', 'a', 'good',
                                    'place', ',', 'but', 'not', 'any', 'longer', '.']
    """
    s_text = re.sub('([.,!?()])', r' \1 ', s_text) # match the punctuation characters and surround them by spaces,
    s_text = re.sub('\s{2,}', ' ', s_text)         # collapse multiple spaces to one space
    words = s_text.lower().split()
    return words


def tagging_IOB(s, aspects):
    """Assigning IOB tag to each word in s
    Input:
        s: sentences, a list of words, e.g. ['judging', 'from', 'previous', 'posts']
        aspects: a list of aspect term, e.g. ['a good place', 'Posts']
    Output:
        tag: a list of tag, e.g. ['O', 'O', 'O', 'B']
    """
    tags = ['O'] * len(s)

    for aspect in aspects:
        pre_index = 0
        for word in s:
            if word in aspect: # 'good' in 'a good place'
                cur_index = s.index(word)
                if cur_index - pre_index == 1: # inside an aspect term
                    tags[cur_index] = 'I'
                else:                       # beginning of an aspect term
                    tags[cur_index] = 'B'
                pre_index = cur_index
    return tags


def dict2df(sentences):
    """Convert list of dict to dataframe
    Input:
        sentences: a list of dictionaries, contains id, text, aspect terms. The output of raw2dict
    Output:
        data: a dataframe with three columns, sentence id, words, tag with IOB format
    """
    data = pd.DataFrame()
    for s in sentences:
        sentence = {}
        sentence['Sentence #'] = s['id']
        sentence['Word'] = split2words(s['text'])  # split text to words
        s_length = len(sentence['Word'])  # the length of sentence, used to generate tag
        if len(s['aspect']) == 0 or s['aspect'][0] == 'NULL':  # tagging: if no aspect term
            sentence['Tag'] = ['O'] * s_length
        else:  # IOB format tag if aspect exist
            aspect_terms = [x.lower() for x in s['aspect']]
            sentence['Tag'] = tagging_IOB(sentence['Word'], aspect_terms)

        # convert each setence to dataframe
        sentence_df = pd.DataFrame.from_dict(sentence)
        data = data.append(sentence_df, ignore_index=True)

    return data


def read_data(file_path):
    # 1 raw data to soup
    soup = None
    with file_path.open(encoding="utf-8") as f:
        soup = BeautifulSoup(f.read().strip(), "lxml-xml")
    if soup is None:
        raise Exception("Can't read xml file")
    sentence_nodes = soup.find_all("sentence")

    # 2  convert soup object to a list of dictionaries
    sentences = soup2dict(sentence_nodes)

    # 3 list to dataframe
    data = dict2df(sentences)

    return data

# Sentence class
class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["Word"].values.tolist(),
#                                                            s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None