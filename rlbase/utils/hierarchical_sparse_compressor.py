import numpy as np
from alphacsc import BatchCDL
import math
    
"""UTILS"""

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
def upsample_data(data, us_factor=2):
    upsampled = [np.repeat(x, 2) for x in data]
    return upsampled

def binarize_dictionary(dictionary):
    bin_d = np.zeros(dictionary.shape)
    for i, d in enumerate(dictionary):
        bin_d[i] = d
        d_mean = np.mean(d)
        bin_d[i][bin_d[i] >= d_mean] = 1.0
        bin_d[i][bin_d[i] < d_mean] = 0.0
    return bin_d

def downsample_dictionary(integer_dictionary, ds_factor=2):
    new_d = []
    for d in integer_dictionary:
        new_d.append(d[::ds_factor])
    return new_d


"""COMPRESSOR"""

class HierarchicalSparseCompressor(object):
    
    def __init__(self, params, rewards=None):
        self.n_actions = params.n_actions
        self.n_learning_stages = params.n_learning_stages
        self.max_atoms = params.max_atoms
        self.atom_length = params.atom_length
        self.sparsity = params.sparsity
        self.selection = params.selection
        self.selection_criterion = params.selection_criterion
        self.count_criterion = params.count_criterion
        self.reward_weighted = params.reward_weighted
        self.reward_coeff = params.reward_coeff
        self.rewards = rewards
        self.added_motifs = {}
        self.data_history = []
        self.dictionary_history = []
        self.split = False
        self.n_splits = None
        
    def gen_datasets(self, data, split_data=False, n_splits=2):
        data_idxs = None
        dataset = list(data)
        if split_data:
            dataset, data_idxs = self.split_data(dataset, n_splits)
        max_length = np.max([len(x) for x in dataset])
        dataset_binary = data_to_matrix(convert_to_binary(dataset, self.n_actions), self.n_actions, max_length=max_length)
        dataset_binary = center(dataset_binary)
        return dataset, dataset_binary, data_idxs

    def split_data(self, data, n_splits):
        new_d = []
        data_idxs = []
        for j,d in enumerate(data):
            chunk_length = 20
            splits = [d[i:i + chunk_length] for i in range(0, len(d), chunk_length)]
            data_idxs += [j] * len(splits)
            new_d += splits
        return new_d, data_idxs
    
    def unsplit_data(self, data, data_idxs):
        unsplit = []
        for i in list(set(data_idxs)):
            unsplit_d = []
            idxs = [j for j,x in enumerate(data_idxs) if x == i]
            data_idxd = [x for k,x in enumerate(data) if k in idxs]
            for d in data_idxd:
                unsplit_d += d
            unsplit.append(unsplit_d)
        return unsplit

    def get_sparse_code(self, dataset_binary, n_atoms, atom_length, data_idxs, D_init=None):
        csc = BatchCDL(n_atoms, atom_length, rank1=False, reg=self.sparsity, 
                           D_init=D_init, sort_atoms=True, n_iter=10,
                           uv_constraint='joint', raise_on_increase=False)
        csc.fit(dataset_binary)
        activity = self.get_dictionary_activity(csc, dataset_binary)
        D = self.convert_dictionary(csc._D_hat)
        if self.selection == 'choose_n':
            sparse_dictionary = [x for _,x in sorted(zip(activity,D)) if x not in self.added_motifs.values()][::-1][:int(self.selection_criterion)]
        elif self.selection == 'activity_threshold':
            sparse_dictionary = [d for i, d in enumerate(D) if activity[i] > self.selection_criterion]
        print('sparse dictionary: {}'.format(sparse_dictionary))
        return sparse_dictionary
        
    def convert_dictionary(self, dictionary):
        D = binarize_dictionary(dictionary)
        D = translate_bin_sequences(D, action_key=None)
        return D
    
    def get_dictionary_activity(self, model, dataset_binary):
        z = model.fit_transform(dataset_binary)
        activity = np.sum(np.sum(z, axis=0), axis=1) 
        activity /= np.sum(activity)
        return activity
    
    def sequence_compressor(self, sequence, motifs_to_add, n_primitives):
        new_motif_label = n_primitives
        added_motifs = []
        for a, s in enumerate(motifs_to_add):
            window_size = len(s)
            new_sequence = sequence.copy()
            skip = 0
            for i in range(len(sequence)-window_size+1):
                if skip == 0:
                    if sequence[i:i+window_size] == s:
                        new_sequence[i:i+window_size] = [new_motif_label] + ['placeholder'] \
                                                                * (window_size-1)
                        skip = window_size - 1
                else:
                    skip -= 1
            sequence = [x for x in new_sequence if x is not 'placeholder']
            if new_motif_label in sequence:
                added_motifs.append(new_motif_label)
                new_motif_label += 1
        return sequence, added_motifs
        
    def compress(self, data, count_threshold=2):
        original_data = list(data.copy())
        self.trajectory_lengths = [len(x) for x in original_data]
        if np.max(self.trajectory_lengths) > 20:
            self.split = True
        self.data_history.append(original_data[0])
        dataset = original_data.copy()
        for n in range(self.n_learning_stages):
            dataset, dataset_binary, data_idxs = self.gen_datasets(dataset)

            """FIND SPARSE CODE"""
            dataset, dataset_binary, data_idxs = self.gen_datasets(dataset, self.split, self.n_splits)
            D = self.get_sparse_code(dataset_binary, self.max_atoms, self.atom_length, data_idxs)


            """COMPRESS SEQUENCE"""
            if self.split:
                dataset = self.unsplit_data(dataset, data_idxs)

            new_x = []
            all_added = []
            for i, s in enumerate(dataset):
                compressed_sequence, added = self.sequence_compressor(s, D, self.n_actions)
                compressed_sequence = [x for x in compressed_sequence if x is not 'placeholder']
                new_x.append(compressed_sequence.copy())
                all_added += added

            """UPDATE DICTIONARY"""
            all_added = list(set(all_added))
            starting_n_actions = self.n_actions
            for i, m in enumerate(D):
                if starting_n_actions+i in all_added:
                    self.added_motifs[self.n_actions] = m
                    self.n_actions += 1

            dataset = new_x.copy()
            self.data_history.append(new_x[0])
            self.dictionary_history.append(self.added_motifs.copy())
            print('new dictionary : {}'.format(self.added_motifs))
