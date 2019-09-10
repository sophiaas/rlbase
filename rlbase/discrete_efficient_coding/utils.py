import numpy as np
import itertools

"""GENERAL UTILS"""

def one_hot(length, index):
    vec = np.zeros(length)
    vec[index] = 1
    return vec

def kbits(n, k):
    result = []
    for bits in itertools.combinations(range(n), k):
        s = [0] * n
        for bit in bits:
            s[bit] = 1
        result.append(s)
    return result

def center(x):
    x -= np.mean(x)
    x /= np.std(x)
    return x

def convert_to_binary(trajectories_list, num_possible_actions):
    bin_actions = []
    for actions in trajectories_list:
        converted_sequence = [one_hot(num_possible_actions, a) for a in actions]
        bin_actions.append(np.array(converted_sequence).T)
    return bin_actions 

def data_to_matrix(bin_actions, n_actions, max_length=100):
    matrix = np.zeros((len(bin_actions), n_actions, max_length))
    for i, row in enumerate(bin_actions):
        matrix[i] = np.concatenate([row, np.zeros((n_actions, max_length - row.shape[1]))], axis=1)
    return matrix

def translate_bin_sequences(bin_sequences, action_key=None):
    translated = []
    for s in bin_sequences:
        new = s.T
        new = new[~np.all(new == 0, axis=1)]
        if action_key is not None:
            translated.append([action_key[np.argmax(x)] for x in new])
        else: 
            translated.append([np.argmax(x) for x in new])
    return translated

def pad_list(l, end_shape=100, pad_element=np.nan):
    new_l = []
    for row in l:
        new_l.append(row + [pad_element]*(end_shape-len(row)))
    return new_l

def pad_and_concat(dictionaries, max_dim=8):
    new_dics = []
    for d in dictionaries:
        d_prime = np.pad(d, ((0,0), (0,0), (0,max_dim-d.shape[2])), mode='constant')
        new_dics.append(d_prime)
    return np.array(np.concatenate(new_dics), dtype=np.float64)

def find_subsequence(subseqs, seq):
    lengths = set([len(x) for x in chunks])
    cumulative = np.zeros(len(chunks))
    n = np.zeros(len(chunks))
    k = np.zeros(len(chunks))
    for sequence_length in lengths:
        for i, t in enumerate(trajectories):
            for j in range(0, len(t)-sequence_length+1):
                candidate = t[j:j+sequence_length]
                if candidate in chunks:
                    cumulative[chunks.index(candidate)] += np.sum(reward[i][j:])
                    n[chunks.index(candidate)] += 1  
    expected_rewards = np.array([cumulative[a]/(n[a]) for a in range(len(cumulative))])
    return expected_rewards

def subsequence_index(subseq, seq):
    """Return all starting indices of a subsequence in the sequence.
    >>> index([1,2], range(5))
    [1]
    >>> index(range(1, 6), range(5))
    []
    >>> index(range(5), range(5))
    [0]
    >>> index([1,2], [0, 1, 0, 1, 2])
    [3]
    """
    i, n, m = -1, len(seq), len(subseq)
    
    try:
        idxs = []
        while True:
            if i+1+m >= n:
                return idxs
            i = seq.index(subseq[0], i + 1, n - m + 1)
            if subseq == seq[i:i + m]:
                idxs.append(i)

    except ValueError:
        return idxs
    
def batch_subsequence_index(subsequence, sequences):
    idxs = []
    for s in sequences:
        idxs.append(subsequence_index(subsequence, s))
    return idxs