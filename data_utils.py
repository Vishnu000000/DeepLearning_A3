import torch
from torch.utils.data import Dataset
from collections import Counter
import pandas as pd
import numpy as np

class CharacterMapper:
    """Maps characters to indices and vice versa"""
    def __init__(self):
        self.char2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
        self.idx2char = {0: '<pad>', 1: '<sos>', 2: '<eos>'}
        self.vocab_size = 3
    
    def build_vocab(self, texts, min_freq=1):
        """Build vocabulary from texts"""
        # Count character frequencies
        char_freq = Counter()
        for text in texts:
            char_freq.update(text)
        
        # Add characters to vocabulary
        for char, freq in char_freq.items():
            if freq >= min_freq and char not in self.char2idx:
                idx = len(self.char2idx)
                self.char2idx[char] = idx
                self.idx2char[idx] = char
        
        self.vocab_size = len(self.char2idx)
    
    def encode(self, text):
        """Convert text to indices"""
        return [self.char2idx.get(c, 0) for c in text]
    
    def decode(self, indices):
        """Convert indices to text"""
        return ''.join([self.idx2char.get(i, '<unk>') for i in indices])

class BilingualPairDataset(Dataset):
    """Dataset for bilingual text pairs"""
    def __init__(self, data_path, source_mapper, target_mapper, max_length=100):
        self.source_mapper = source_mapper
        self.target_mapper = target_mapper
        self.max_length = max_length
        
        # Load data
        self.data = pd.read_csv(
            data_path,
            sep='\t',
            header=None,
            names=['target', 'source', 'freq']
        )
        
        # Build vocabulary if not already built
        if source_mapper.vocab_size == 3:
            source_mapper.build_vocab(self.data['source'].tolist())
        if target_mapper.vocab_size == 3:
            target_mapper.build_vocab(self.data['target'].tolist())
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get source and target texts
        source_text = self.data.iloc[idx]['source']
        target_text = self.data.iloc[idx]['target']
        
        # Encode texts
        source_indices = [self.source_mapper.char2idx['<sos>']] + \
                        self.source_mapper.encode(source_text) + \
                        [self.source_mapper.char2idx['<eos>']]
        
        target_indices = [self.target_mapper.char2idx['<sos>']] + \
                        self.target_mapper.encode(target_text) + \
                        [self.target_mapper.char2idx['<eos>']]
        
        # Pad sequences
        source_indices = self._pad_sequence(source_indices, self.max_length)
        target_indices = self._pad_sequence(target_indices, self.max_length)
        
        return torch.tensor(source_indices), torch.tensor(target_indices)
    
    def _pad_sequence(self, sequence, max_length):
        """Pad sequence to max_length"""
        if len(sequence) > max_length:
            return sequence[:max_length]
        return sequence + [0] * (max_length - len(sequence)) 