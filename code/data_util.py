from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from defs import LBLS, NUM, UNK
from collections import OrderedDict

def get_word2embed_dict(embedding, vocab):
    """
    Load word vector mapping using @embedding, @vocab.
    Assumes each line of the vocab file matches with those of the embedding
    file.
    """  
    ret = OrderedDict()
    for key in vocab.keys(): 
        ret[key] = embedding[vocab[key]]

    return ret


def preprocess_sequence_data(dataset, embed_dict, question_max_length, context_max_length, n_features):

    ret = []
    for (question, context, answer_span) in dataset:
        # replace tokens with corresponding embedding
        question_embed = embed(question, embed_dict)
        context_embed = embed(context, embed_dict)
        # create list of labels of max_length
        answer_labels = labelize(answer_span, context_max_length)
        # pad question and context to be max_length
        # also return masks for BOTH question and context
        question_data, question_mask = pad(question_embed, n_features, question_max_length)
        context_data, context_mask = pad(context_embed, n_features, context_max_length)

        ret.append((question_data, context_data, 
                    answer_labels, 
                    question_mask, context_mask))

    return ret

def embed(tokens, embed_dict):

    ret = []
    for token in tokens:
        # normalize token to find it in embed_dict
        word = normalize(token)
        # word's embedding (UNK's embedding otherwise)
        wv = embed_dict.get(word, embed_dict[UNK])
        
        ret.append(wv)

    return ret

def normalize(word):
    """
    Normalize words that are numbers or have casing.
    """
    if word.isdigit(): 
        return NUM
    else: 
        return word.lower()

def labelize(span, max_length):
    # create negative label list
    ml = max_length
    labels = LBLS[-1] * ml
    # set appropriate labels positive
    # if span[0] < ml:
    #     start = span[0]
    #     end = min(span[1] + 1, ml)
    #     labels[start:end] = LBLS[0] * (end - start)
    if span[0]< ml: 
        labels[span[0]] = LBLS[0]
    if span[1]< ml: 
        labels[span[1]] = LBLS[1]

    assert len(labels) == max_length, "Labelize: 'labels' is of the wrong shape!"

    return labels

def pad(word_vector, n_features, max_length):
    # initialize padding variables
    zero_vector = [0] * n_features
    zero_label = 0 
    # pad word_vector to max_length
    pad_len = max(max_length - len(word_vector), 0)
    padding = zero_vector * pad_len
    word_vector = word_vector + padding
    word_vector_in = word_vector[:max_length]
    word_vector_mask = [True] * (max_length - pad_len) + [False] * pad_len

    return word_vector_in, word_vector_mask