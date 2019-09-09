import numpy as np
from discrete_efficient_coding.utils import *
from alphacsc import BatchCDL
from discrete_efficient_coding.visualization import plot_grid
import math
    

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
        plot_grid(D)
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
