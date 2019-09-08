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
#             chunk_length = math.ceil(len(d) / n_splits)
            chunk_length = 20
            splits = [d[i:i + chunk_length] for i in range(0, len(d), chunk_length)]
            data_idxs += [j] * len(splits)
#             for s in range(len(splits)):
#                 data_idxs.append(j)
            new_d += splits
        return new_d, data_idxs
    
#     def unsplit_data(self, data, n_splits):
    def unsplit_data(self, data, data_idxs):
        unsplit = []
        for i in list(set(data_idxs)):
            unsplit_d = []
            idxs = [j for j,x in enumerate(data_idxs) if x == i]
            data_idxd = [x for k,x in enumerate(data) if k in idxs]
            for d in data_idxd:
                unsplit_d += d
            unsplit.append(unsplit_d)
#             unsplit.append([sum(x, []) for x in [data[j] for j,y in enumerate(data_idxs) if y == i]])
#         unsplit = [sum(x, []) for x in [data[i:i + n_splits] for i in range(0, len(data), n_splits)]]
#         unsplit =  for j in range(unique(data_idxs))]
        return unsplit
    
#     def get_basins_of_attraction(self, data, atom_length):
#         motif_extractor = HopfieldMotifExtractor([atom_length], self.n_actions)
#         motifs, counts = motif_extractor.find_motifs(data)
#         if self.count_criterion is not None:
#             motifs = [m for i, m in enumerate(motifs) if counts[i] > self.count_criterion * len(data)]
#             counts = [c for c in counts if c > self.count_criterion * len(data)]
#         return motifs, counts
    
#     def convert_basins(self, motifs):
#         motifs = np.array(convert_to_binary(motifs, self.n_actions))
#         motifs = center(motifs)    
#         return motifs
    
    def get_sparse_code(self, dataset_binary, n_atoms, atom_length, data_idxs, D_init=None):
#         if D_init is not None:
#             num_additional_atoms = n_atoms - len(D_init)
#             if num_additional_atoms > 0:
#                 D_init = np.concatenate([D_init, [np.zeros(D_init[0].shape)+np.min(D_init[0])]*num_additional_atoms])
#         else:
#             rewards = self.rewards
        csc = BatchCDL(n_atoms, atom_length, rank1=False, reg=self.sparsity, 
                           D_init=D_init, sort_atoms=True, n_iter=10,
                           uv_constraint='joint', raise_on_increase=False)
        csc.fit(dataset_binary)
#         if self.reward_weighted:
#             activity = self.get_dictionary_activity_per_trajectory(csc, dataset_binary, data_idxs)
#             activity *= self.rewards
#             activity = np.sum(activity, axis=1)
#         else:
        activity = self.get_dictionary_activity(csc, dataset_binary)
        D = self.convert_dictionary(csc._D_hat)
#         plt.figure()
#         plt.bar(range(len(D)), activity)
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
    
#     def get_dictionary_activity_per_trajectory(self, model, dataset_binary, data_idxs):
#         z = model.fit_transform(dataset_binary)
#         print('z shape {}'.format(z.shape))
#         print('bin data shape {}'.format(dataset_binary.shape))
#         activity = np.swapaxes(z, 0, 1)
#         print('activity shape {}'.format(activity.shape))
#         if self.split:
#             A = []
#             for coeff in activity:
#                 unsplit = np.array(self.unsplit_data([list(x) for x in coeff], data_idxs))
#                 print('unsplit shape  {}'.format(unsplit.shape))
#                 activity = np.sum(unsplit, axis=1)
#                 A.append(activity)
#             activity = np.array(A)
#         else:
#             activity = np.sum(activity, axis=2)
#             activity = activity.T
#         return activity
    
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
                        new_sequence[i:i+window_size] = [new_motif_label] + ['placeholder'] * (window_size-1)
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
#             self.n_splits = math.ceil(np.max(self.trajectory_lengths) / 20)
        self.data_history.append(original_data[0])
#         if self.split and self.count_criterion is not None:
#             self.count_criterion *= self.n_splits
        dataset = original_data.copy()
        print('n learning stages: {}'.format(self.n_learning_stages))
        for n in range(self.n_learning_stages):
            dataset, dataset_binary, data_idxs = self.gen_datasets(dataset)
#             plt.matshow(dataset_binary[0])

#             if type(self.atom_length) is not int:
#                 atom_length = self.atom_length[n]
#             else:
#                 atom_length = self.atom_length
#             motifs, counts = self.get_basins_of_attraction(dataset, atom_length)
#             print(list(zip(motifs, counts)))
            
            if True == True:
#             if len(motifs) > 0:
                """FIND SPARSE CODE"""
                dataset, dataset_binary, data_idxs = self.gen_datasets(dataset, self.split, self.n_splits)
    #             binary_motifs = self.convert_basins(motifs)
    #             if len(binary_motifs) >  self.max_atoms:
    #                 binary_motifs = binary_motifs[:self.max_atoms]
    #             D = self.get_sparse_code(dataset_binary, self.max_atoms, atom_length, binary_motifs)
                D = self.get_sparse_code(dataset_binary, self.max_atoms, self.atom_length, data_idxs)


                """COMPRESS SEQUENCE"""
                if self.split:
#                     dataset = self.unsplit_data(dataset, self.n_splits)
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
                max_length = np.max([len(x) for x in dataset])
#                 if self.split and len(dataset[0]) <= len(original_data[0]) / self.n_splits:
#                 if self.split and len(dataset[0]) <= len(original_data[0]) / self.n_splits:

#                     self.split = False    
#                     if self.count_criterion is not None:
#                         self.count_criterion = 2

#             else:
#                 print('NO MORE MOTIFS')
#                 print(motifs)
#                 break
#         plt.matshow(dataset_binary[0])