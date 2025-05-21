import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import logging
from collections import Counter, defaultdict
import re
import unicodedata
import random
from tqdm import tqdm
from dataclasses import dataclass
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CharacterProperties:
    """Character properties for advanced processing"""
    category: str
    name: str
    numeric: Optional[float]
    decomposition: str
    script: str
    bidirectional: str
    combining: int
    mirrored: bool

class DynamicCharacterProcessor:
    """Advanced character processing with dynamic mapping"""
    def __init__(self, min_freq: int = 2, max_chars: int = 1000):
        self.min_freq = min_freq
        self.max_chars = max_chars
        self.special_tokens = {
            '<pad>': 0,
            '<sos>': 1,
            '<eos>': 2,
            '<unk>': 3,
            '<mask>': 4
        }
        self.char_to_idx = {**self.special_tokens}
        self.idx_to_char = {v: k for k, v in self.special_tokens.items()}
        self.char_freq = defaultdict(int)
        self.char_properties = {}
        
    def _get_char_properties(self, char: str) -> CharacterProperties:
        """Get comprehensive Unicode properties for a character"""
        return CharacterProperties(
            category=unicodedata.category(char),
            name=unicodedata.name(char, ''),
            numeric=unicodedata.numeric(char, None),
            decomposition=unicodedata.decomposition(char),
            script=unicodedata.script(char),
            bidirectional=unicodedata.bidirectional(char),
            combining=unicodedata.combining(char),
            mirrored=unicodedata.mirrored(char)
        )
    
    def build_vocab(self, texts: List[str]):
        """Build vocabulary with advanced character analysis"""
        # Count frequencies and collect properties
        for text in texts:
            for char in text:
                self.char_freq[char] += 1
                if char not in self.char_properties:
                    self.char_properties[char] = self._get_char_properties(char)
        
        # Sort characters by multiple criteria
        sorted_chars = sorted(
            self.char_freq.items(),
            key=lambda x: (
                x[1],  # frequency
                len(self.char_properties[x[0]].name),  # name length
                bool(self.char_properties[x[0]].numeric),  # is numeric
                self.char_properties[x[0]].combining,  # combining class
                self.char_properties[x[0]].mirrored  # is mirrored
            ),
            reverse=True
        )
        
        # Add characters up to max_chars
        for char, freq in sorted_chars:
            if freq >= self.min_freq and len(self.char_to_idx) < self.max_chars:
                idx = len(self.char_to_idx)
                self.char_to_idx[char] = idx
                self.idx_to_char[idx] = char
    
    def encode(self, text: str) -> List[int]:
        """Encode text to indices with special token handling"""
        return [self.char_to_idx.get(c, self.special_tokens['<unk>']) 
                for c in text]
    
    def decode(self, indices: List[int]) -> str:
        """Decode indices to text with special token handling"""
        return ''.join(self.idx_to_char.get(idx, '<unk>') 
                      for idx in indices)
    
    def get_similar_chars(self, char: str) -> List[str]:
        """Get similar characters based on comprehensive properties"""
        if char not in self.char_properties:
            return [char]
            
        char_props = self.char_properties[char]
        similar = []
        
        for c, props in self.char_properties.items():
            # Calculate similarity score
            score = 0
            if props.category == char_props.category:
                score += 2
            if props.script == char_props.script:
                score += 2
            if props.bidirectional == char_props.bidirectional:
                score += 1
            if props.combining == char_props.combining:
                score += 1
            if props.mirrored == char_props.mirrored:
                score += 1
            if props.name.startswith(char_props.name.split()[0]):
                score += 2
                
            if score >= 3:  # Threshold for similarity
                similar.append(c)
                
        return similar or [char]

class AdaptiveTextTransformer:
    """Advanced text transformation with dynamic character manipulation"""
    def __init__(self, 
                 noise_level: float = 0.1,
                 max_perturbations: int = 3,
                 char_processor: Optional[DynamicCharacterProcessor] = None):
        self.noise_level = noise_level
        self.max_perturbations = max_perturbations
        self.char_processor = char_processor
        self.perturbation_types = [
            self._swap_chars,
            self._delete_char,
            self._duplicate_char,
            self._insert_similar_char,
            self._reverse_substring,
            self._shuffle_chars
        ]
        
    def _reverse_substring(self, text: str, pos: int) -> str:
        """Reverse a random substring"""
        if len(text) < 3:
            return text
        length = random.randint(2, min(4, len(text) - pos))
        substring = text[pos:pos + length]
        return text[:pos] + substring[::-1] + text[pos + length:]
    
    def _shuffle_chars(self, text: str, pos: int) -> str:
        """Shuffle characters in a random substring"""
        if len(text) < 3:
            return text
        length = random.randint(2, min(4, len(text) - pos))
        chars = list(text[pos:pos + length])
        random.shuffle(chars)
        return text[:pos] + ''.join(chars) + text[pos + length:]
    
    def _get_similar_chars(self, char: str) -> List[str]:
        """Get similar characters using the character processor"""
        if self.char_processor:
            return self.char_processor.get_similar_chars(char)
        return [char]
    
    def _swap_chars(self, text: str, pos: int) -> str:
        """Swap adjacent characters with probability based on similarity"""
        if pos < len(text) - 1:
            chars = list(text)
            if random.random() < 0.7:  # 70% chance to swap similar characters
                similar = self._get_similar_chars(chars[pos])
                chars[pos] = random.choice(similar)
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
            return ''.join(chars)
        return text
    
    def _delete_char(self, text: str, pos: int) -> str:
        """Delete character with probability based on frequency"""
        if pos < len(text):
            char = text[pos]
            if self.char_processor and char in self.char_processor.char_freq:
                freq = self.char_processor.char_freq[char]
                if random.random() < 0.1 * (1 / (freq + 1)):  # Higher chance to delete rare characters
                    return text[:pos] + text[pos + 1:]
        return text
    
    def _duplicate_char(self, text: str, pos: int) -> str:
        """Duplicate character with probability based on position"""
        if pos < len(text):
            # Higher chance to duplicate characters in the middle
            middle_pos = len(text) / 2
            distance = abs(pos - middle_pos)
            prob = 0.3 * (1 - distance / len(text))
            if random.random() < prob:
                return text[:pos] + text[pos] + text[pos:]
        return text
    
    def _insert_similar_char(self, text: str, pos: int) -> str:
        """Insert similar character with probability based on context"""
        if pos < len(text):
            # Get context (previous and next characters)
            prev_char = text[pos - 1] if pos > 0 else ''
            next_char = text[pos] if pos < len(text) else ''
            
            # Get similar characters for context
            similar_chars = []
            if prev_char:
                similar_chars.extend(self._get_similar_chars(prev_char))
            if next_char:
                similar_chars.extend(self._get_similar_chars(next_char))
            
            if similar_chars:
                return text[:pos] + random.choice(similar_chars) + text[pos:]
        return text
    
    def transform(self, text: str) -> str:
        """Apply random transformations to text"""
        if random.random() > self.noise_level:
            return text
        
        num_perturbations = random.randint(1, self.max_perturbations)
        for _ in range(num_perturbations):
            pos = random.randint(0, len(text) - 1)
            perturbation = random.choice(self.perturbation_types)
            text = perturbation(text, pos)
        
        return text

class TransliterationDataset(Dataset):
    """Dataset for transliteration with advanced processing"""
    def __init__(self, 
                 data: List[Tuple[str, str]],
                 source_processor: DynamicCharacterProcessor,
                 target_processor: DynamicCharacterProcessor,
                 transformer: Optional[AdaptiveTextTransformer] = None,
                 max_length: int = 100):
        self.data = data
        self.source_processor = source_processor
        self.target_processor = target_processor
        self.transformer = transformer
        self.max_length = max_length
        
        # Calculate sample weights
        self.weights = self._calculate_weights()
    
    def _calculate_weights(self) -> np.ndarray:
        """Calculate sample weights based on multiple criteria"""
        weights = []
        for src, tgt in self.data:
            # Length-based weight
            length_weight = 1.0 / (len(src) + len(tgt))
            
            # Complexity weight
            src_complexity = len(set(src)) / len(src)
            tgt_complexity = len(set(tgt)) / len(tgt)
            complexity_weight = (src_complexity + tgt_complexity) / 2
            
            # Character property weight
            src_props = [self.source_processor.char_properties.get(c, None) for c in src]
            tgt_props = [self.target_processor.char_properties.get(c, None) for c in tgt]
            
            src_prop_weight = sum(1 for p in src_props if p and p.combining > 0) / len(src)
            tgt_prop_weight = sum(1 for p in tgt_props if p and p.combining > 0) / len(tgt)
            prop_weight = (src_prop_weight + tgt_prop_weight) / 2
            
            # Combined weight
            weight = length_weight * complexity_weight * (1 + prop_weight)
            weights.append(weight)
        
        return np.array(weights) / sum(weights)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src, tgt = self.data[idx]
        
        # Apply transformation if available
        if self.transformer:
            src = self.transformer.transform(src)
        
        # Encode sequences
        src_tensor = torch.tensor(self.source_processor.encode(src))
        tgt_tensor = torch.tensor(self.target_processor.encode(tgt))
        
        # Truncate if necessary
        if len(src_tensor) > self.max_length:
            src_tensor = src_tensor[:self.max_length]
        if len(tgt_tensor) > self.max_length:
            tgt_tensor = tgt_tensor[:self.max_length]
        
        return src_tensor, tgt_tensor
    
    def get_loader(self, 
                   batch_size: int,
                   shuffle: bool = True,
                   num_workers: int = 4) -> DataLoader:
        """Get DataLoader with weighted sampling"""
        sampler = torch.utils.data.WeightedRandomSampler(
            self.weights, len(self), replacement=True)
        
        return DataLoader(
            self,
            batch_size=batch_size,
            sampler=sampler if shuffle else None,
            shuffle=shuffle if sampler is None else False,
            num_workers=num_workers,
            pin_memory=True
        )

def create_dataloaders(train_file: str,
                      val_file: str,
                      test_file: str,
                      batch_size: int,
                      max_len: int,
                      min_freq: int = 2,
                      max_vocab_size: Optional[int] = None,
                      num_workers: int = 4,
                      preprocessor: Optional[DynamicCharacterProcessor] = None,
                      augmenter: Optional[AdaptiveTextTransformer] = None) -> Tuple[DataLoader, DataLoader, DataLoader, DynamicCharacterProcessor, DynamicCharacterProcessor]:
    """Create data loaders for training, validation, and testing."""
    # Load data
    train_data = pd.read_csv(train_file)
    val_data = pd.read_csv(val_file)
    test_data = pd.read_csv(test_file)
    
    # Initialize preprocessor if not provided
    if preprocessor is None:
        preprocessor = DynamicCharacterProcessor(min_freq=min_freq, max_chars=max_vocab_size)
    
    # Build vocabularies
    preprocessor.build_vocab(train_data['source'].tolist())
    preprocessor.build_vocab(train_data['target'].tolist())
    
    # Create datasets
    train_dataset = TransliterationDataset(
        list(zip(train_data['source'], train_data['target'])),
        preprocessor, preprocessor, augmenter, max_len
    )
    
    val_dataset = TransliterationDataset(
        list(zip(val_data['source'], val_data['target'])),
        preprocessor, preprocessor, None, max_len
    )
    
    test_dataset = TransliterationDataset(
        list(zip(test_data['source'], test_data['target'])),
        preprocessor, preprocessor, None, max_len
    )
    
    # Create data loaders
    train_loader = train_dataset.get_loader(batch_size, True, num_workers)
    val_loader = val_dataset.get_loader(batch_size, False, num_workers)
    test_loader = test_dataset.get_loader(batch_size, False, num_workers)
    
    return train_loader, val_loader, test_loader, preprocessor, preprocessor