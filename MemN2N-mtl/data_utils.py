from __future__ import absolute_import

import os
import re
import numpy as np
import tensorflow as tf

stop_words=set(["a","an","the"])


def load_candidates(data_dir, task_id):
    assert task_id > 0 and task_id < 7
    candidates=[]
    candidates_f=None
    candid_dic={}
    candidates_f='../personalized-dialog-candidates.txt'
    with open(os.path.join(data_dir,candidates_f)) as f:
        for i,line in enumerate(f):
            candid_dic[line.strip().split(' ',1)[1]] = i
            line=tokenize(line.strip())[1:]
            candidates.append(line)
    # return candidates,dict((' '.join(cand),i) for i,cand in enumerate(candidates))
    return candidates,candid_dic


def load_dialog_task(data_dir, task_id, candid_dic, isOOV):
    '''Load the nth task. There are 5 tasks in total.

    Returns a tuple containing the training and testing data for the task.
    '''
    assert task_id > 0 and task_id < 6

    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'personalized-dialog-task{}-'.format(task_id)
    train_file = [f for f in files if s in f and 'trn' in f][0]
    if isOOV:
        test_file = [f for f in files if s in f and 'tst-OOV' in f][0]
    else: 
        test_file = [f for f in files if s in f and 'tst.' in f][0]
    val_file = [f for f in files if s in f and 'dev' in f][0]
    train_data = get_dialogs(train_file,candid_dic)
    test_data = get_dialogs(test_file,candid_dic)
    val_data = get_dialogs(val_file,candid_dic)
    return train_data, test_data, val_data


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple']
    '''
    sent=sent.lower()
    if sent=='<silence>':
        return [sent]
    result=[x.strip() for x in re.split('(\W+)?', sent) if x.strip() and x.strip() not in stop_words]
    if not result:
        result=['<silence>']
    if result[-1]=='.' or result[-1]=='?' or result[-1]=='!':
        result=result[:-1]
    return result


# def parse_dialogs(lines,candid_dic):
#     '''
#         Parse dialogs provided in the babi tasks format
#     '''
#     data=[]
#     context=[]
#     u=None
#     r=None
#     for line in lines:
#         line=str.lower(line.strip())
#         if line:
#             nid, line = line.split(' ', 1)
#             nid = int(nid)
#             if '\t' in line:
#                 u, r = line.split('\t')
#                 u = tokenize(u)
#                 r = tokenize(r)
#                 # temporal encoding, and utterance/response encoding
#                 u.append('$u')
#                 u.append('#'+str(nid))
#                 r.append('$r')
#                 r.append('#'+str(nid))
#                 context.append(u)
#                 context.append(r)
#             else:
#                 r=tokenize(line)
#                 r.append('$r')
#                 r.append('#'+str(nid))
#                 context.append(r)
#         else:
#             context=[x for x in context[:-2] if x]
#             u=u[:-2]
#             r=r[:-2]
#             key=' '.join(r)
#             if key in candid_dic:
#                 r=candid_dic[key]
#                 data.append((context, u,  r))
#             context=[]
#     return data

def parse_dialogs_per_response(lines,candid_dic):
    '''
        Parse dialogs provided in the babi tasks format
    '''
    data=[]
    context=[]
    u=None
    r=None
    for line in lines:
        line=line.strip()
        if line:
            nid, line = line.split(' ', 1)
            nid = int(nid)
            if '\t' in line:
                u, r = line.split('\t')
                a = candid_dic[r]
                u = tokenize(u)
                r = tokenize(r)
                # temporal encoding, and utterance/response encoding
                # data.append((context[:],u[:],candid_dic[' '.join(r)]))
                data.append((context[:],u[:],a))
                u.append('$u')
                u.append('#'+str(nid))
                r.append('$r')
                r.append('#'+str(nid))
                context.append(u)
                context.append(r)
            else:
                r=tokenize(line)
                r.append('$r')
                r.append('#'+str(nid))
                context.append(r)
        else:
            # clear context
            context=[]
    return data



def get_dialogs(f,candid_dic):
    '''Given a file name, read the file, retrieve the dialogs, and then convert the sentences into a single dialog.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    with open(f) as f:
        return parse_dialogs_per_response(f.readlines(),candid_dic)

def vectorize_candidates_sparse(candidates,word_idx):
    shape=(len(candidates),len(word_idx)+1)
    indices=[]
    values=[]
    for i,candidate in enumerate(candidates):
        for w in candidate:
            indices.append([i,word_idx[w]])
            values.append(1.0)
    return tf.SparseTensor(indices,values,shape)

def vectorize_candidates(candidates,word_idx,sentence_size):
    shape=(len(candidates),sentence_size)
    C=[]
    for i,candidate in enumerate(candidates):
        lc=max(0,sentence_size-len(candidate))
        C.append([word_idx[w] if w in word_idx else 0 for w in candidate] + [0] * lc)
    return tf.constant(C,shape=shape)


def vectorize_data(data, word_idx, sentence_size, batch_size, candidates_size, max_memory_size, profile_mapping):
    """
    Vectorize stories and queries.

    If a sentence length < sentence_size, the sentence will be padded with 0's.

    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.

    The answer array is returned as a one-hot encoding.
    """
    P = []
    S = []
    Q = []
    A = []
    data.sort(key=lambda x:len(x[0]),reverse=True)

    for i, (story, query, answer) in enumerate(data):
        if i%batch_size==0:
            memory_size=max(1,min(max_memory_size,len(story)))

        profile_attributes = profile_mapping[tuple(story[0][:2])]

        ss = []
        for i, sentence in enumerate(story, 1):
            ls = max(0, sentence_size - len(sentence))
            ss.append([word_idx[w] if w in word_idx else 0 for w in sentence] + [0] * ls)

        # take only the most recent sentences that fit in memory
        ss = ss[::-1][:memory_size][::-1]

        # pad to memory_size
        lm = max(0, memory_size - len(ss))
        for _ in range(lm):
            ss.append([0] * sentence_size)

        lq = max(0, sentence_size - len(query))
        q = [word_idx[w] if w in word_idx else 0 for w in query] + [0] * lq

        P.append(profile_attributes)
        S.append(np.array(ss))
        Q.append(np.array(q))
        A.append(np.array(answer))

    return P, S, Q, A


def generate_profile_encoding(data):
    """
    Read the entire dataset to find all possible profiles and encode them

    Args:
        data: raw data, as the same form of `data` of vectorize_data

    Returns:
        map: Returns a map of the form `Map[(String, String), Int]` where
             `(String, String)` is of the form of a profile (ex. `("female", "young")`).
             The ID (thus the Int) is unique amongst the map.

    """

    profiles = {tuple(story[0][:2]) for (story,_,_) in data}
    profiles_sorted = sorted(profiles)
    profiles_mapping = {p: idx for (idx, p) in enumerate(profiles_sorted)}
    return profiles_mapping


class IdenticalWordIdx:
    """
    Simple testing WordIdx that simply returns the key as value

    This class mimic a dictionary used for word to index conversion,
    but does not do the conversion. It simply returns the word directly.

    It is useful in order to test the different functions.

    Attributes:
        word_idx (set|dict): Used to test if the word in known or not, but the word
                             itself will be returned by `__getitem__`.
    """
    def __init__(self, word_idx=None):
        self.word_idx = word_idx

    def __contains__(self, item):
        return item in self.word_idx if self.word_idx is not None else True

    def __getitem__(self, item):
        return item

class OnetimeCandidateDict:
    """
    Class for generating a mapping on the flow.

    Basically, if a key is not present in the dict, then it's value is
    set to a unused index. Otherwise, the index already created is returned.
    """
    def __init__(self):
        self.candidat_dict = dict()

    def __contains__(self, item):
        return True

    def __getitem__(self, item):
        if item in self.candidat_dict:
            return self.candidat_dict[item]

        value = len(self.candidat_dict)
        self.candidat_dict[item] = value

        return value

def compute_data_size(dirpath, task_id=5, oov=False):
    """Helper function to compute the number of observations in the dataset"""
    data = load_dialog_task(dirpath, task_id, OnetimeCandidateDict(), oov)
    #vectorized_p = map(lambda d: vectorize_data(d, IdenticalWordIdx(), 100, 32, -1, 20, OnetimeCandidateDict())[0], data)

    return tuple(map(len, data))


def compute_data_size_recursively(base_dir):
    """
    If base_dir has internal directories, recusrive call on them. Otherwise,
    call `compute_data_size` and print it. Does nothing if base_dir is not
    a directory.
    """

    if not os.path.isdir(base_dir):
        return

    subdirs = {p for p in map(lambda s: os.path.join(base_dir, s), os.listdir(base_dir)) if os.path.isdir(p)}

    if len(subdirs) == 0:
        try:
            dss = compute_data_size(base_dir)
        except IndexError:
            print('Ignored:', base_dir)
            return

        detail = 'train: {}, val: {}, test: {}'.format(*dss)

        print('{}: {}'.format(base_dir, detail))
    else:
        for p in subdirs:
            compute_data_size_recursively(p)

