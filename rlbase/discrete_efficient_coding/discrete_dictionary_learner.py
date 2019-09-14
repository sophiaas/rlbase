import numpy as np
import pandas as pd
import math
import pickle
from hdnet.spikes import Spikes
from hierarchical_hdnet.hierarchical_spike_model import HierarchicalSpikeModel
from hdnet.spikes_model import SpikeModel, Spikes
import itertools
from discrete_efficient_coding.utils import convert_to_binary, data_to_matrix, translate_bin_sequences, center
import scipy as sp

class DictionaryLearner(object):
    def __init__(self, params, n_primitive_actions=5):
#         self.schedule = params['h_schedule']
        self.window_sizes = params['window_sizes']
        self.learn_every = params['learn_every']
        self.num_learning_stages = params['num_h_learning_stages']
        self.learn_vs_load = params['learn_vs_load']
        self.dictionary_path = params['h_dictionary_path']
        self.load_in_stages = params['load_in_stages']
        self.num_hl_actions = params['num_hl_actions']
        self.motif_method = params['motif_method']
        self.n_possible_actions = params['n_possible_actions']
        self.n_samples = params['n_samples']
        self.max_dictionary_size = params['max_dictionary_size']
        self.n_primitive_actions = n_primitive_actions
        self.max_length = params['max_length']
        
        if self.motif_method == 'hopfield':
            self.motif_extractor = HopfieldMotifExtractor(self.window_sizes, self.n_possible_actions)
        
        self.compressor = Compressor()
        self.reward_estimator = RewardEstimator()
        
        self.learning_stages_completed = 0
        self.last_learned_episode = 0
        self.open_dictionary_idx = n_primitive_actions
        
        self.run_data = {}
        self.dictionary = {x: [] for x in range(n_primitive_actions, n_primitive_actions+self.num_hl_actions)}
        self.load_dictionary()
            
    def load_dictionary(self):
        if self.dictionary_path is not None:
            file_name = self.dictionary_path
            if self.load_in_stages:
                file_name += '_stage_'+str(self.learning_stages_completed)   
            self.dictionary = pickle.load(open('dictionaries/'+file_name+'.p', "rb" ))
        else: 
            self.dictionary = self.dictionary
       
    def push_data(self, run_data):
        self.run_data = run_data
        
    def update_this_episode(self, episode):
        if episode > 0 and episode % self.learn_every == 0:
            return True
        
    def find_nesting(self, dictionary, n_prev_actions=5):
        dictionary = [list(item) for item in set(tuple(row) for row in dictionary)]
        lengths = [len(x) for x in dictionary]
        lengths, new_d = zip(*sorted(zip(lengths, dictionary)))
        new_d = list(new_d)
        for j in range(len(new_d)):
            for i in range(len(new_d)):
                if new_d[j] != new_d[i]:
                    new_sequence, found = self.compressor.compress(new_d[i], [new_d[j]], n_prev_actions+j)
                    if len(found) > 0:
                        new_d[i] = new_sequence    
#         new_d = [list(item) for item in set(tuple(row) for row in new_d)]
        new_d = {x: d for x,d in zip(range(n_prev_actions, n_prev_actions+len(dictionary)), new_d)}
        return new_d
        
    def get_candidate_dictionaries(self, motifs, counts, previous_dictionary):
        dictionaries = []
        dictionary_idxs = []
        probabilities = [x/np.sum(counts) for x in counts]
        for s in range(self.n_samples):
            not_passed = 0
            broken = False
            unique = False
            dictionary_size = np.random.randint(2, self.max_dictionary_size+1)
            while not unique:
                idxs = np.random.choice(range(len(motifs)), size=dictionary_size, p=probabilities)
                if set(idxs) not in dictionary_idxs:
                    dictionary_idxs.append(set(idxs))
                    unique = True
                else:
                    not_passed += 1
                if not_passed > 10:
                    broken = True
                    break
            if not broken:   
                dictionary = [x for i,x in enumerate(motifs) if i in idxs]
                dictionary = self.find_nesting(dictionary) ##NPREVACTIONS?? FIND DIC REDUNDANCY?
                dictionaries.append(dictionary)
        return dictionaries
    
    def evaluate_dictionaries(self, dictionaries, sequences, advantages):
        sequences_combined = sum(sequences, [])
        results = pd.DataFrame()
        redundancy_reduction = []
        print(len(sequences))
        for d in dictionaries:
            compressions = []
            for s in sequences:
                compressed_sequence, added = self.compressor.compress(s, list(d.values()), self.n_primitive_actions)
                compressions.append(compressed_sequence)
            motif_advantages = self.reward_estimator.estimate_motif_advantages(d, sequences, advantages)
            dictionary_advantage = np.mean(motif_advantages)
            length_diff = np.array([len(x) for x in sequences]) - np.array([len(x) for x in compressions])
            program_length = np.mean([len(x) for x in compressions]) + np.sum([len(x) for x in d.values()])
#             mean_program_length = np.mean())
            coding_reduction = np.mean(1 - (program_length / np.array([len(x) for x in sequences])))
            compressions_combined = sum(compressions, [])
            rr, original_entropy, compressed_entropy = self.compressor.compute_redundancy_reduction(sequences_combined, compressions_combined, d)
            redundancy_reduction.append(rr)
            df = pd.DataFrame({'dictionary': [d], 'compression': [compressions], 
                               'compressed_entropy': [compressed_entropy], 
                               'original_entropy': [original_entropy], 
                               'redundancy_reduction': [rr], 
                               'length_diff': [length_diff], 
                               'dictionary_advantage': [dictionary_advantage],
                               'program_length': [program_length],
                               'coding_reduction': [coding_reduction]})
            results = pd.concat([results, df])
        results = results.sort_values('redundancy_reduction', ascending=False)
        results = results.reset_index(drop=True)
        return results    
        
    def update_dictionary(self, run_data, episode):
        self.push_data(run_data)
        if self.update_this_episode(episode):
            if self.learn_vs_load == 'load' and self.load_in_stages:
                self.load_dictionary()
            elif self.learn_vs_load == 'learn':
                start_idx = self.last_learned_episode
                end_idx = self.last_learned_episode+self.learn_every
                print(start_idx, end_idx)
                training_trajectories = list(self.run_data['actions'][start_idx:end_idx])
                training_reward = list(self.run_data['reward'][start_idx:end_idx])
                training_advantages = list(self.run_data['advantages'][start_idx:end_idx])
                previous_dictionary = [x for x in self.dictionary.values() if len(x) > 0]
                motifs, counts = self.motif_extractor.find_motifs(training_trajectories, max_length=self.max_length)
                self.motifs = motifs; self.counts = counts
                candidate_dictionaries = self.get_candidate_dictionaries(motifs, counts, self.dictionary)
                evaluation = self.evaluate_dictionaries(candidate_dictionaries, training_trajectories, training_advantages)
                self.evaluation = evaluation
                new_dictionary = evaluation['dictionary'][0]
                idx_range = range(self.open_dictionary_idx, self.open_dictionary_idx+len(new_dictionary))
                for i, j in enumerate(idx_range):
                    if new_dictionary[j] not in self.dictionary.values() and j in self.dictionary.keys():
                        self.dictionary[j] = new_dictionary[j]
                self.last_learned_episode = self.last_learned_episode+self.learn_every
            self.learning_stages_completed += 1
            
        
                
             
class Compressor(object):
#     def compress(self, sequence, motifs_to_add, n_primitives=5):
#         added_motifs = []
#         new_alphabet = {}
#         for a, s in motifs_to_add.keys(), motifs_to_add.values():
#             window_size = len(s)
#             new_sequence = []
#             delete = []
#             replace = []
#             for i in range(len(sequence)-window_size+1):
#                 if sequence[i:i+window_size] == s:
#                     replace.append(i)
#                     delete += [a for a in range(i+1, i+window_size)]
#             for j, y in enumerate(sequence):
#                 if j in delete:
#                     pass
#                 elif j in replace:
#                     new_sequence.append(a)
#                 else:
#                     new_sequence.append(y)
#             if a in new_sequence:
#                 binary_motif = motifs_to_add[a]
#                 added_motifs.append(a)
#             sequence = new_sequence
#         return sequence, added_motifs
    
    def compress(self, sequence, motifs_to_add, n_primitives=5):
        new_motif_label = n_primitives
        added_motifs = []
        for a, s in enumerate(motifs_to_add):
            window_size = len(s)
            for i in range(len(sequence)-window_size+1):
                if sequence[i:i+window_size] == s:
                    sequence[i:i+window_size] = [new_motif_label] + ['placeholder'] * (window_size-1)
            if new_motif_label in sequence:
                added_motifs.append(new_motif_label)
                new_motif_label += 1
            sequence = sequence.copy()
            sequence = [x for x in sequence if x is not 'placeholder']
        sequence = sequence.copy()
        sequence = [x for x in sequence if x is not 'placeholder']
#         print(sequence)
        return sequence, added_motifs

    def compute_entropy(self, sequence, choices):
        probs = []
        for n in choices:
            probs.append(list(sequence).count(n) / len(sequence))
        H = sp.stats.entropy(probs)
        return H

    def average_motif_length(self, dictionary, n_primitives=5):
        return np.mean([len(x) for x in dictionary] + [1]*n_primitives)

    def compute_redundancy_reduction(self, original, recoded, dictionary, n_primitives=5):
        aml = self.average_motif_length(dictionary.values())
        original_entropy = self.compute_entropy(original, range(n_primitives))
        recoded_entropy = self.compute_entropy(recoded, range(n_primitives+len(dictionary))) / aml
        L = 1 - (recoded_entropy / aml) / original_entropy
        return L, original_entropy, recoded_entropy

    def uncompress_motif(self, motif, dictionary, n_primitives=5):
        primitive = False if np.any([x >= n_primitives for x in motif]) else True
        while not primitive:
            primitive_sequence = []
            for x in motif:
                if x < n_primitives:
                    primitive_sequence.append(x)
                else: 
                    primitive_sequence += dictionary[x]
            motif = primitive_sequence
            primitive = False if np.any([x >= n_primitives for x in motif]) else True
        return motif
    
class MotifExtractor(object):
    def __init__(self, motif_lengths, n_possible_values, top_n=None):
        self.motif_lengths = motif_lengths
        self.n_possible_values = n_possible_values
        self.top_n = top_n
        
    def find_motifs(self, sequences):
        return
    
                
class HopfieldMotifExtractor(MotifExtractor):
                                      
    def find_motifs(self, sequences):
        max_length = np.max([len(x) for x in sequences])
        binary_sequences = convert_to_binary(sequences, self.n_possible_values)
        data = data_to_matrix(binary_sequences, self.n_possible_values, max_length)
        spikes = Spikes(data)
        all_motifs = []
        all_counts = []
        for s in self.motif_lengths:
            self.model = HierarchicalSpikeModel(pattern_type='binary', spikes=spikes, window_size=s, nlayers=1, count_threshold=2, savefigs=False)
            self.model.fit()
            top_patterns, counts = self.model.get_top_patterns(layer=0, count_threshold=2)
            patterns = self.model.layers[0].hopfield_patterns
            motifs = np.array([patterns.pattern_for_key(patterns.patterns[x]).reshape((data.shape[1], s)) for x in top_patterns])
            motifs = translate_bin_sequences(motifs)
            counts = [x for i, x in enumerate(counts) if motifs[i] != [0] and len(motifs[i]) == s and np.sum(motifs[i])>0]
            motifs = [x for x in motifs if x != [0] and len(x) == s and np.sum(x)>0]
            if self.top_n is not None:
                motifs = motifs[:self.top_n+1]
                counts = mounts[:self.top_n+1]
#             all_motifs.append([list(x) for x in set(tuple(x) for x in motifs)])
            all_motifs.append(motifs)
            all_counts.append(counts)
        all_motifs = sum(all_motifs, [])
        all_counts = sum(all_counts, [])
        return all_motifs, all_counts
    
class RewardEstimator(object):
    def __init__(self):
        self.compressor = Compressor()
    
    def estimate_motif_advantages(self, motifs, trajectories, advantages):
        unpacked_motifs = [self.compressor.uncompress_motif(x, motifs) for x in motifs.values()]
        lengths = set([len(x) for x in unpacked_motifs])
        cumulative = np.zeros(len(unpacked_motifs))
        n = np.zeros(len(unpacked_motifs))
        k = np.zeros(len(unpacked_motifs))
        for sequence_length in lengths:
            for i, t in enumerate(trajectories):
                for j in range(0, len(t)-sequence_length+1):
                    candidate = t[j:j+sequence_length]
                    if candidate in unpacked_motifs:
                        cumulative[unpacked_motifs.index(candidate)] += np.sum(advantages[i][j:])
                        n[unpacked_motifs.index(candidate)] += 1  
                        k[unpacked_motifs.index(candidate)] += len(advantages[i][j:])
        motif_advantages = np.array([cumulative[a]/(n[a]) for a in range(len(cumulative))])
#         motif_advantages = (motif_advantages)
        return motif_advantages

# class RewardEstimator(object):
#     def __init__(self, trajectories, rewards, states, puzzle):
#         self.trajectories = trajectories
#         self.rewards = rewards
#         self.puzzle = puzzle
#         self.state_trajectories, self.state_key = self.get_states()
#         self.values = get_values()
#         self.value_trajectories = self.get_value_trajectories(self):
        
#     def get_states(self):
#         from env import LB
#         state_trajectories = []
#         unique_states = []
#         state_key = {}
#         unique_state_idx = 0
#         for t in self.trajectories:
#             lb = LB(self.puzzle, max_count=20000, testing=False, reward_fn='10,10,-10,-1', hierarchical_args={'num_h_actions':0})
#             lb.reset()
#             states = []
#             for a in t:
#                 states.append(list(lb.state))
#                 lb.step(a)
#                 if list(lb.state) not in unique_states:
#                     state_key[unique_state_idx] = list(lb.state)
#                     unique_states.append(list(lb.state))
#             state_trajectories.append(states)
#         return state_trajectories, state_key

#     def get_values(self):
#         values = {state: 0 for state in self.state_key.keys()}
#         counts = {state: 0 for state in self.state_key.keys()}
#         for state in self.state_key.keys():
#             for i, t in enumerate(self.state_trajectories):
#                 for j, s in enumerate(t):
#                     if s == self.state_key[state]:
#                         values[state] += np.sum(rewards[i][j:])
#                         counts[state] += 1
#         for i in range(len(values)):
#             values[i] /= counts[i] 
#         return values, counts
    
#     def get_value_trajectories(self):
#         value_trajectories = []
#         for t in self.state_trajectories:
#             v = []
#             for j,s in enumerate(t):
#                 idx = [x for x, y in zip(self.state_key.keys(), state_key.values()) if y == s]
#                 v.append(values[idx[0]])
#             value_traj.append(v)
#         return value_traj