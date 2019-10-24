import torch
from torch.distributions import Categorical
import numpy as np


def get_blocks(sequence, masks, max_length):
    #TODO: make sure masking is working
    blocks = {x: [] for x in range(2, max_length+1)}
    for i in range(2, max_length+1):
        m = [tuple(masks[a:a+i]) for a in range(len(masks)-i)]
        exclude = [x for x in m if (x==0).nonzero().shape[0] > 0]
        blocks[i] += [tuple(sequence[a:a+i]) for a in range(len(sequence)-i) if a not in exclude]
    return blocks

def sample_blocks(sequence, masks, max_length, n_samples):
    #TODO: make sure masking is working
    blocks = {x: [] for x in range(2, max_length+1)}
    episode_ends = (masks==0).nonzero()
    for b in range(2, max_length+1):
        for i in range(n_samples):
            nonvalid = []
            for end in episode_ends:
                nonvalid += [end-x for x in range(b+1)]
            idx_set = [x for x in range(sequence.shape[0]-max_length) if x not in nonvalid]
            idx = np.random.choice(idx_set)
            random_block = tuple(sequence[idx:idx+b])
            blocks[b].append(random_block)            
    return blocks

def block_entropy(sequence, masks, possible_values, max_length, sample=False, n_samples=None):
    if sample:
        blocks = sample_blocks(sequence, masks, max_length, n_samples)
    else:
        blocks = get_blocks(sequence, max_length)
        
    probs = {i: torch.zeros(size=[possible_values]*i) for i in range(2, max_length+1)}
    
    for d in range(2, max_length+1):
        for instance in blocks[d]:
            probs[d][instance] += 1
            
    distributions = [Categorical(x.view(-1)) for i,x in probs.items()]
    entropy = torch.tensor([x.entropy() for x in distributions])
    block_H = entropy.mean()
    return block_H