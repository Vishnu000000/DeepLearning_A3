import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any
import logging
from collections import Counter, defaultdict
import re
import unicodedata
import random
from tqdm import tqdm
from dataclasses import dataclass
from functools import lru_cache
import json
import hashlib
import emoji

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CharacterProperties:
    """Advanced character properties for processing"""
    category: str
    name: str
    numeric: Optional[float]
    decomposition: str
    script: str
    bidirectional: str
    combining: int
    mirrored: bool
    case: str
    width: str
    east_asian_width: str
    age: str
    block: str
    script_extensions: List[str]
    is_emoji: bool
    is_math: bool
    is_currency: bool
    is_punctuation: bool
    is_whitespace: bool

class DynamicCharacterMapper:
    """Advanced character mapping with dynamic processing"""
    def __init__(self, 
                 min_freq: int = 2,
                 max_chars: int = 1000,
                 cache_size: int = 1000):
        self.min_freq = min_freq
        self.max_chars = max_chars
        self.cache_size = cache_size
        
        # Special tokens with unique identifiers
        self.special_tokens = {
            '<pad>': 0,
            '<sos>': 1,
            '<eos>': 2,
            '<unk>': 3,
            '<mask>': 4,
            '<sep>': 5,
            '<cls>': 6,
            '<num>': 7,
            '<punct>': 8,
            '<space>': 9
        }
        
        # Character mappings
        self.char_to_idx = {**self.special_tokens}
        self.idx_to_char = {v: k for k, v in self.special_tokens.items()}
        
        # Character statistics
        self.char_freq = defaultdict(int)
        self.char_properties = {}
        
        # Caching
        self._similar_chars_cache = {}
        self._property_cache = {}
        
        # Character clusters
        self.char_clusters = defaultdict(set)
    
    def _get_char_properties(self, char: str) -> CharacterProperties:
        """Get comprehensive Unicode properties for a character"""
        if char in self._property_cache:
            return self._property_cache[char]
        
        props = CharacterProperties(
            category=unicodedata.category(char),
            name=unicodedata.name(char, ''),
            numeric=unicodedata.numeric(char, None),
            decomposition=unicodedata.decomposition(char),
            script=unicodedata.script(char),
            bidirectional=unicodedata.bidirectional(char),
            combining=unicodedata.combining(char),
            mirrored=unicodedata.mirrored(char),
            case=unicodedata.casefold(char),
            width=unicodedata.east_asian_width(char),
            east_asian_width=unicodedata.east_asian_width(char),
            age=unicodedata.age(char),
            block=unicodedata.block(char),
            script_extensions=unicodedata.script_extensions(char),
            is_emoji=char in emoji.UNICODE_EMOJI,
            is_math=unicodedata.category(char).startswith('Sm'),
            is_currency=unicodedata.category(char).startswith('Sc'),
            is_punctuation=unicodedata.category(char).startswith('P'),
            is_whitespace=char.isspace()
        )
        
        self._property_cache[char] = props
        return props
    
    def _compute_char_similarity(self, char1: str, char2: str) -> float:
        """Compute similarity between two characters"""
        props1 = self._get_char_properties(char1)
        props2 = self._get_char_properties(char2)
        
        # Compute similarity score
        score = 0.0
        
        # Category similarity
        if props1.category == props2.category:
            score += 2.0
        
        # Script similarity
        if props1.script == props2.script:
            score += 2.0
        
        # Name similarity
        name1_words = set(props1.name.lower().split())
        name2_words = set(props2.name.lower().split())
        name_similarity = len(name1_words & name2_words) / len(name1_words | name2_words)
        score += name_similarity * 2.0
        
        # Block similarity
        if props1.block == props2.block:
            score += 1.0
        
        # Case similarity
        if props1.case == props2.case:
            score += 0.5
        
        # Width similarity
        if props1.width == props2.width:
            score += 0.5
        
        # Special property similarity
        if props1.is_emoji == props2.is_emoji:
            score += 1.0
        if props1.is_math == props2.is_math:
            score += 1.0
        if props1.is_currency == props2.is_currency:
            score += 1.0
        if props1.is_punctuation == props2.is_punctuation:
            score += 1.0
        if props1.is_whitespace == props2.is_whitespace:
            score += 1.0
        
        return score / 10.0  # Normalize to [0, 1]
    
    def _build_char_clusters(self):
        """Build character clusters based on properties"""
        for char, props in self.char_properties.items():
            # Add to appropriate clusters
            self.char_clusters['category_' + props.category].add(char)
            self.char_clusters['script_' + props.script].add(char)
            self.char_clusters['block_' + props.block].add(char)
            
            if props.is_emoji:
                self.char_clusters['emoji'].add(char)
            if props.is_math:
                self.char_clusters['math'].add(char)
            if props.is_currency:
                self.char_clusters['currency'].add(char)
            if props.is_punctuation:
                self.char_clusters['punctuation'].add(char)
            if props.is_whitespace:
                self.char_clusters['whitespace'].add(char)
    
    def build_vocab(self, texts: List[str]):
        """Build vocabulary with advanced character analysis"""
        # Count frequencies
        for text in texts:
            for char in text:
                self.char_freq[char] += 1
                if char not in self.char_properties:
                    self.char_properties[char] = self._get_char_properties(char)
        
        # Build character clusters
        self._build_char_clusters()
        
        # Sort characters by multiple criteria
        sorted_chars = sorted(
            self.char_freq.items(),
            key=lambda x: (
                x[1],  # frequency
                len(self.char_properties[x[0]].name),  # name length
                bool(self.char_properties[x[0]].numeric),  # is numeric
                self.char_properties[x[0]].combining,  # combining class
                self.char_properties[x[0]].mirrored,  # is mirrored
                len(self.char_properties[x[0]].script_extensions),  # script extensions
                len(self.char_clusters.get('category_' + self.char_properties[x[0]].category, set())),  # category size
                len(self.char_clusters.get('script_' + self.char_properties[x[0]].script, set()))  # script size
            ),
            reverse=True
        )
        
        # Add characters up to max_chars
        for char, freq in sorted_chars:
            if freq >= self.min_freq and len(self.char_to_idx) < self.max_chars:
                idx = len(self.char_to_idx)
                self.char_to_idx[char] = idx
                self.idx_to_char[idx] = char
    
    def get_similar_chars(self, char: str, threshold: float = 0.5) -> List[str]:
        """Get similar characters based on comprehensive properties"""
        if char in self._similar_chars_cache:
            return self._similar_chars_cache[char]
        
        if char not in self.char_properties:
            return [char]
        
        # Get character clusters
        props = self.char_properties[char]
        clusters = [
            self.char_clusters.get('category_' + props.category, set()),
            self.char_clusters.get('script_' + props.script, set()),
            self.char_clusters.get('block_' + props.block, set())
        ]
        
        if props.is_emoji:
            clusters.append(self.char_clusters['emoji'])
        if props.is_math:
            clusters.append(self.char_clusters['math'])
        if props.is_currency:
            clusters.append(self.char_clusters['currency'])
        if props.is_punctuation:
            clusters.append(self.char_clusters['punctuation'])
        if props.is_whitespace:
            clusters.append(self.char_clusters['whitespace'])
        
        # Get similar characters from clusters
        similar = []
        for cluster in clusters:
            for c in cluster:
                if c in self.special_tokens:
                    continue
                
                similarity = self._compute_char_similarity(char, c)
                if similarity >= threshold:
                    similar.append((c, similarity))
        
        # Sort by similarity
        similar.sort(key=lambda x: x[1], reverse=True)
        result = [c for c, _ in similar]
        
        # Cache result
        if len(self._similar_chars_cache) >= self.cache_size:
            self._similar_chars_cache.pop(next(iter(self._similar_chars_cache)))
        self._similar_chars_cache[char] = result
        
        return result
    
    def encode(self, text: str) -> List[int]:
        """Encode text to indices with special token handling"""
        indices = []
        for char in text:
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
            else:
                props = self._get_char_properties(char)
                if props.is_emoji:
                    indices.append(self.special_tokens['<unk>'])
                elif props.is_math:
                    indices.append(self.special_tokens['<unk>'])
                elif props.is_currency:
                    indices.append(self.special_tokens['<unk>'])
                elif props.is_punctuation:
                    indices.append(self.special_tokens['<punct>'])
                elif props.is_whitespace:
                    indices.append(self.special_tokens['<space>'])
                elif props.numeric is not None:
                    indices.append(self.special_tokens['<num>'])
                else:
                    indices.append(self.special_tokens['<unk>'])
        return indices
    
    def decode(self, indices: List[int]) -> str:
        """Decode indices to text with special token handling"""
        return ''.join(self.idx_to_char.get(idx, '<unk>') 
                      for idx in indices)

class AdaptiveTextTransformer:
    """Advanced text transformation with dynamic character manipulation"""
    def __init__(self, 
                 noise_level: float = 0.1,
                 max_perturbations: int = 3,
                 char_mapper: Optional[DynamicCharacterMapper] = None):
        self.noise_level = noise_level
        self.max_perturbations = max_perturbations
        self.char_mapper = char_mapper
        
        # Define perturbation types with weights
        self.perturbation_types = [
            (self._swap_chars, 0.3),
            (self._delete_char, 0.2),
            (self._duplicate_char, 0.15),
            (self._insert_similar_char, 0.2),
            (self._reverse_substring, 0.1),
            (self._shuffle_chars, 0.05)
        ]
    
    def _get_perturbation_type(self) -> Tuple[callable, float]:
        """Get random perturbation type based on weights"""
        weights = [w for _, w in self.perturbation_types]
        return random.choices(
            self.perturbation_types,
            weights=weights,
            k=1
        )[0]
    
    def _swap_chars(self, text: str, pos: int) -> str:
        """Swap characters with probability based on similarity"""
        if pos < len(text) - 1:
            chars = list(text)
            if self.char_mapper and random.random() < 0.7:
                similar = self.char_mapper.get_similar_chars(chars[pos])
                chars[pos] = random.choice(similar)
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
            return ''.join(chars)
        return text
    
    def _delete_char(self, text: str, pos: int) -> str:
        """Delete character with probability based on frequency"""
        if pos < len(text):
            char = text[pos]
            if self.char_mapper and char in self.char_mapper.char_freq:
                freq = self.char_mapper.char_freq[char]
                if random.random() < 0.1 * (1 / (freq + 1)):
                    return text[:pos] + text[pos + 1:]
        return text
    
    def _duplicate_char(self, text: str, pos: int) -> str:
        """Duplicate character with probability based on position"""
        if pos < len(text):
            middle_pos = len(text) / 2
            distance = abs(pos - middle_pos)
            prob = 0.3 * (1 - distance / len(text))
            if random.random() < prob:
                return text[:pos] + text[pos] + text[pos:]
        return text
    
    def _insert_similar_char(self, text: str, pos: int) -> str:
        """Insert similar character with probability based on context"""
        if pos < len(text) and self.char_mapper:
            # Get context
            prev_char = text[pos - 1] if pos > 0 else ''
            next_char = text[pos] if pos < len(text) else ''
            
            # Get similar characters
            similar_chars = []
            if prev_char:
                similar_chars.extend(self.char_mapper.get_similar_chars(prev_char))
            if next_char:
                similar_chars.extend(self.char_mapper.get_similar_chars(next_char))
            
            if similar_chars:
                return text[:pos] + random.choice(similar_chars) + text[pos:]
        return text
    
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
    
    def transform(self, text: str) -> str:
        """Apply random transformations to text"""
        if random.random() > self.noise_level:
            return text
        
        num_perturbations = random.randint(1, self.max_perturbations)
        for _ in range(num_perturbations):
            pos = random.randint(0, len(text) - 1)
            perturbation, _ = self._get_perturbation_type()
            text = perturbation(text, pos)
        
        return text

class TransliterationDataset(Dataset):
    """Dataset for transliteration with advanced processing"""
    def __init__(self, 
                 data: List[Tuple[str, str]],
                 source_mapper: DynamicCharacterMapper,
                 target_mapper: DynamicCharacterMapper,
                 transformer: Optional[AdaptiveTextTransformer] = None,
                 max_length: int = 100):
        self.data = data
        self.source_mapper = source_mapper
        self.target_mapper = target_mapper
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
            src_props = [self.source_mapper.char_properties.get(c, None) for c in src]
            tgt_props = [self.target_mapper.char_properties.get(c, None) for c in tgt]
            
            src_prop_weight = sum(1 for p in src_props if p and p.combining > 0) / len(src)
            tgt_prop_weight = sum(1 for p in tgt_props if p and p.combining > 0) / len(tgt)
            prop_weight = (src_prop_weight + tgt_prop_weight) / 2
            
            # Cluster weight
            src_clusters = sum(len(self.source_mapper.char_clusters.get('category_' + p.category, set()))
                             for p in src_props if p) / len(src)
            tgt_clusters = sum(len(self.target_mapper.char_clusters.get('category_' + p.category, set()))
                             for p in tgt_props if p) / len(tgt)
            cluster_weight = (src_clusters + tgt_clusters) / 2
            
            # Combined weight
            weight = length_weight * complexity_weight * (1 + prop_weight) * (1 + cluster_weight)
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
        src_tensor = torch.tensor(self.source_mapper.encode(src))
        tgt_tensor = torch.tensor(self.target_mapper.encode(tgt))
        
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
                      mapper: Optional[DynamicCharacterMapper] = None,
                      augmenter: Optional[AdaptiveTextTransformer] = None) -> Tuple[DataLoader, DataLoader, DataLoader, DynamicCharacterMapper, DynamicCharacterMapper]:
    """Create data loaders for training, validation, and testing"""
    # Load data
    train_data = pd.read_csv(train_file)
    val_data = pd.read_csv(val_file)
    test_data = pd.read_csv(test_file)
    
    # Initialize mapper if not provided
    if mapper is None:
        mapper = DynamicCharacterMapper(min_freq=min_freq, max_chars=max_vocab_size)
    
    # Build vocabularies
    mapper.build_vocab(train_data['source'].tolist())
    mapper.build_vocab(train_data['target'].tolist())
    
    # Create datasets
    train_dataset = TransliterationDataset(
        list(zip(train_data['source'], train_data['target'])),
        mapper, mapper, augmenter, max_len
    )
    
    val_dataset = TransliterationDataset(
        list(zip(val_data['source'], val_data['target'])),
        mapper, mapper, None, max_len
    )
    
    test_dataset = TransliterationDataset(
        list(zip(test_data['source'], test_data['target'])),
        mapper, mapper, None, max_len
    )
    
    # Create data loaders
    train_loader = train_dataset.get_loader(batch_size, True, num_workers)
    val_loader = val_dataset.get_loader(batch_size, False, num_workers)
    test_loader = test_dataset.get_loader(batch_size, False, num_workers)
    
    return train_loader, val_loader, test_loader, mapper, mapper