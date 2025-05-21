import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import wandb
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
import re
import unicodedata
from collections import Counter, defaultdict
import json
from IPython.display import HTML, display
import random
import os
from typing import Dict, List, Tuple, Optional, Union
import string
from dataclasses import dataclass
from functools import lru_cache
import math

class DynamicCharacterMapper:
    """Dynamic character mapping with frequency-based filtering and Unicode properties"""
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
        
    def _get_char_properties(self, char: str) -> Dict[str, str]:
        """Get Unicode properties for a character"""
        return {
            'category': unicodedata.category(char),
            'name': unicodedata.name(char, ''),
            'numeric': unicodedata.numeric(char, None),
            'decomposition': unicodedata.decomposition(char)
        }
        
    def build_vocab(self, texts: List[str]):
        # Count character frequencies and collect properties
        for text in texts:
            for char in text:
                self.char_freq[char] += 1
                if char not in self.char_properties:
                    self.char_properties[char] = self._get_char_properties(char)
        
        # Sort characters by frequency and properties
        sorted_chars = sorted(
            self.char_freq.items(),
            key=lambda x: (
                x[1],  # frequency
                len(self.char_properties[x[0]]['name']),  # name length
                bool(self.char_properties[x[0]]['numeric'])  # is numeric
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
        return [self.char_to_idx.get(c, self.special_tokens['<unk>']) 
                for c in text]
    
    def decode(self, indices: List[int]) -> str:
        return ''.join(self.idx_to_char.get(idx, '<unk>') 
                      for idx in indices)
    
    def get_char_properties(self, char: str) -> Dict[str, str]:
        """Get properties for a character"""
        return self.char_properties.get(char, {})
    
    def get_similar_chars(self, char: str) -> List[str]:
        """Get similar characters based on Unicode properties"""
        if char not in self.char_properties:
            return [char]
            
        char_props = self.char_properties[char]
        similar = []
        
        for c, props in self.char_properties.items():
            if (props['category'] == char_props['category'] and
                props['name'].startswith(char_props['name'].split()[0])):
                similar.append(c)
                
        return similar or [char]

class AdaptiveTextTransformer:
    """Advanced text transformation with dynamic character manipulation"""
    def __init__(self, 
                 noise_level: float = 0.1,
                 max_perturbations: int = 3,
                 char_mapper: Optional[DynamicCharacterMapper] = None):
        self.noise_level = noise_level
        self.max_perturbations = max_perturbations
        self.char_mapper = char_mapper
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
        """Get similar characters using the character mapper"""
        if self.char_mapper:
            return self.char_mapper.get_similar_chars(char)
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
            if self.char_mapper and char in self.char_mapper.char_freq:
                freq = self.char_mapper.char_freq[char]
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
    """Dataset for transliteration with character-level processing"""
    def __init__(self, 
                 data: List[Tuple[str, str]],
                 source_processor: DynamicCharacterMapper,
                 target_processor: DynamicCharacterMapper,
                 augmenter: Optional[AdaptiveTextTransformer] = None,
                 max_length: int = 100):
        self.data = data
        self.source_processor = source_processor
        self.target_processor = target_processor
        self.augmenter = augmenter
        self.max_length = max_length
        
        # Calculate sample weights
        self.weights = self._calculate_weights()
    
    def _calculate_weights(self) -> np.ndarray:
        weights = []
        for src, tgt in self.data:
            # Length-based weight
            length_weight = 1.0 / (len(src) + len(tgt))
            
            # Complexity weight
            src_complexity = len(set(src)) / len(src)
            tgt_complexity = len(set(tgt)) / len(tgt)
            complexity_weight = (src_complexity + tgt_complexity) / 2
            
            # Combined weight
            weight = length_weight * complexity_weight
            weights.append(weight)
        
        return np.array(weights) / sum(weights)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src, tgt = self.data[idx]
        
        # Apply augmentation if available
        if self.augmenter:
            src = self.augmenter.transform(src)
        
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

class DualPathEncoder(nn.Module):
    """Dual-path encoder with character-level and subword-level processing"""
    def __init__(self, 
                 vocab_size: int,
                 embed_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.char_encoder = nn.GRU(
            embed_dim, hidden_dim, num_layers=num_layers,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )
        self.subword_encoder = nn.GRU(
            embed_dim, hidden_dim, num_layers=num_layers,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Character-level encoding
        embedded = self.dropout(self.embedding(x))
        char_output, char_hidden = self.char_encoder(embedded)
        
        # Subword-level encoding
        subword_output, subword_hidden = self.subword_encoder(embedded)
        
        # Combine outputs
        combined_output = char_output + subword_output
        combined_hidden = char_hidden + subword_hidden
        
        return combined_output, combined_hidden

class HierarchicalAttention(nn.Module):
    """Hierarchical attention mechanism"""
    def __init__(self, 
                 hidden_dim: int,
                 num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)
        
        # Linear projections
        q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Transpose for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, v)
        
        # Reshape and combine heads
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.num_heads * self.head_dim)
        
        return output, attention_weights

class DualPathDecoder(nn.Module):
    """Dual-path decoder with hierarchical attention"""
    def __init__(self, 
                 vocab_size: int,
                 embed_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.char_decoder = nn.GRU(
            embed_dim + hidden_dim, hidden_dim, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0
        )
        self.subword_decoder = nn.GRU(
            embed_dim + hidden_dim, hidden_dim, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = HierarchicalAttention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, 
                x: torch.Tensor,
                encoder_output: torch.Tensor,
                encoder_hidden: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Embedding
        embedded = self.dropout(self.embedding(x))
        
        # Character-level decoding
        char_output, _ = self.char_decoder(
            torch.cat([embedded, encoder_hidden[-1].unsqueeze(1)], dim=-1)
        )
        
        # Subword-level decoding
        subword_output, _ = self.subword_decoder(
            torch.cat([embedded, encoder_hidden[-1].unsqueeze(1)], dim=-1)
        )
        
        # Attention
        context, attention_weights = self.attention(
            char_output + subword_output,
            encoder_output,
            encoder_output,
            mask
        )
        
        # Output projection
        output = self.output(context)
        
        return output, attention_weights

class HierarchicalSeq2Seq(nn.Module):
    """Hierarchical sequence-to-sequence model"""
    def __init__(self, 
                 source_vocab_size: int,
                 target_vocab_size: int,
                 embed_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder = DualPathEncoder(
            source_vocab_size, embed_dim, hidden_dim, num_layers, dropout
        )
        self.decoder = DualPathDecoder(
            target_vocab_size, embed_dim, hidden_dim, num_layers, dropout
        )
        
    def forward(self, 
                source: torch.Tensor,
                target: torch.Tensor,
                source_mask: Optional[torch.Tensor] = None,
                target_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode
        encoder_output, encoder_hidden = self.encoder(source)
        
        # Decode
        decoder_output, attention_weights = self.decoder(
            target, encoder_output, encoder_hidden, source_mask
        )
        
        return decoder_output, attention_weights
    
    def generate(self, 
                source: torch.Tensor,
                max_length: int,
                device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = source.size(0)
        
        # Encode
        encoder_output, encoder_hidden = self.encoder(source)
        
        # Initialize decoder input
        decoder_input = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        
        # Generate
        outputs = []
        attention_weights = []
        
        for _ in range(max_length):
            decoder_output, attention = self.decoder(
                decoder_input, encoder_output, encoder_hidden
            )
            
            # Get most likely token
            decoder_input = decoder_output.argmax(dim=-1)
            
            outputs.append(decoder_output)
            attention_weights.append(attention)
        
        return torch.cat(outputs, dim=1), torch.cat(attention_weights, dim=1)

def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                num_epochs: int,
                learning_rate: float,
                device: torch.device,
                wandb_project: Optional[str] = None):
    """Train the model with logging"""
    if wandb_project:
        wandb.init(project=wandb_project)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_steps = 0
        
        for batch_idx, (source, target) in enumerate(tqdm(train_loader)):
            source = source.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            
            output, _ = model(source, target[:, :-1])
            loss = criterion(
                output.reshape(-1, output.size(-1)),
                target[:, 1:].reshape(-1)
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
            
            if wandb_project:
                wandb.log({
                    'train_loss': loss.item(),
                    'learning_rate': optimizer.param_groups[0]['lr']
                })
        
        avg_train_loss = train_loss / train_steps
        
        # Validation
        model.eval()
        val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            for source, target in val_loader:
                source = source.to(device)
                target = target.to(device)
                
                output, _ = model(source, target[:, :-1])
                loss = criterion(
                    output.reshape(-1, output.size(-1)),
                    target[:, 1:].reshape(-1)
                )
                
                val_loss += loss.item()
                val_steps += 1
        
        avg_val_loss = val_loss / val_steps
        scheduler.step(avg_val_loss)
        
        if wandb_project:
            wandb.log({
                'epoch': epoch,
                'avg_train_loss': avg_train_loss,
                'avg_val_loss': avg_val_loss
            })
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pt')
        
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Average Training Loss: {avg_train_loss:.4f}')
        print(f'Average Validation Loss: {avg_val_loss:.4f}')
        print(f'Best Validation Loss: {best_val_loss:.4f}')
        print('-' * 50)

def create_interactive_connectivity(attention_weights: np.ndarray,
                                  source_chars: List[str],
                                  target_chars: List[str],
                                  threshold: float = 0.1) -> str:
    """Create interactive connectivity visualization"""
    html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            .container { display: flex; flex-direction: column; align-items: center; }
            .source, .target { display: flex; justify-content: center; margin: 20px; }
            .char { 
                width: 30px; height: 30px; 
                border: 1px solid #ccc; 
                margin: 0 5px; 
                display: flex; 
                align-items: center; 
                justify-content: center;
                position: relative;
            }
            .connection {
                position: absolute;
                background: rgba(0, 255, 0, 0.3);
                pointer-events: none;
            }
            .controls {
                margin: 20px;
                display: flex;
                align-items: center;
            }
            .char:hover { background: #f0f0f0; }
            .char.selected { background: #e0e0e0; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="target" id="target"></div>
            <div class="controls">
                <label>Threshold: </label>
                <input type="range" min="0" max="100" value="10" id="threshold">
                <span id="thresholdValue">0.1</span>
            </div>
            <div class="source" id="source"></div>
        </div>
        <script>
            const attentionWeights = {attention_weights};
            const sourceChars = {source_chars};
            const targetChars = {target_chars};
            
            function createCharElement(char, isSource) {
                const div = document.createElement('div');
                div.className = 'char';
                div.textContent = char;
                div.dataset.index = isSource ? sourceChars.indexOf(char) : targetChars.indexOf(char);
                return div;
            }
            
            function updateConnections() {
                const threshold = parseFloat(document.getElementById('threshold').value) / 100;
                document.getElementById('thresholdValue').textContent = threshold.toFixed(2);
                
                // Remove existing connections
                document.querySelectorAll('.connection').forEach(c => c.remove());
                
                // Get selected characters
                const selectedSource = document.querySelector('.source .char.selected');
                const selectedTarget = document.querySelector('.target .char.selected');
                
                if (selectedSource && selectedTarget) {
                    const sourceIdx = parseInt(selectedSource.dataset.index);
                    const targetIdx = parseInt(selectedTarget.dataset.index);
                    
                    // Create connection
                    const connection = document.createElement('div');
                    connection.className = 'connection';
                    
                    const sourceRect = selectedSource.getBoundingClientRect();
                    const targetRect = selectedTarget.getBoundingClientRect();
                    
                    const x1 = sourceRect.left + sourceRect.width / 2;
                    const y1 = sourceRect.top;
                    const x2 = targetRect.left + targetRect.width / 2;
                    const y2 = targetRect.bottom;
                    
                    const length = Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
                    const angle = Math.atan2(y2 - y1, x2 - x1) * 180 / Math.PI;
                    
                    connection.style.width = `${length}px`;
                    connection.style.left = `${x1}px`;
                    connection.style.top = `${y1}px`;
                    connection.style.transform = `rotate(${angle}deg)`;
                    connection.style.transformOrigin = '0 0';
                    
                    document.body.appendChild(connection);
                }
            }
            
            // Initialize visualization
            const sourceContainer = document.getElementById('source');
            const targetContainer = document.getElementById('target');
            
            sourceChars.forEach(char => {
                const charElement = createCharElement(char, true);
                charElement.addEventListener('click', () => {
                    document.querySelectorAll('.source .char').forEach(c => c.classList.remove('selected'));
                    charElement.classList.add('selected');
                    updateConnections();
                });
                sourceContainer.appendChild(charElement);
            });
            
            targetChars.forEach(char => {
                const charElement = createCharElement(char, false);
                charElement.addEventListener('click', () => {
                    document.querySelectorAll('.target .char').forEach(c => c.classList.remove('selected'));
                    charElement.classList.add('selected');
                    updateConnections();
                });
                targetContainer.appendChild(charElement);
            });
            
            document.getElementById('threshold').addEventListener('input', updateConnections);
            
            // Initial update
            updateConnections();
        </script>
    </body>
    </html>
    '''
    
    return html_template.format(
        attention_weights=json.dumps(attention_weights.tolist()),
        source_chars=json.dumps(source_chars),
        target_chars=json.dumps(target_chars)
    )

def visualize_attention(model: nn.Module,
                       source_text: str,
                       target_text: str,
                       source_processor: DynamicCharacterMapper,
                       target_processor: DynamicCharacterMapper,
                       device: torch.device,
                       wandb_project: Optional[str] = None):
    """Visualize attention weights for a given source-target pair"""
    # Prepare input
    source_tensor = torch.tensor(
        source_processor.encode(source_text)
    ).unsqueeze(0).to(device)
    
    target_tensor = torch.tensor(
        target_processor.encode(target_text)
    ).unsqueeze(0).to(device)
    
    # Get attention weights
    with torch.no_grad():
        _, attention_weights = model(source_tensor, target_tensor)
    
    # Convert to numpy
    attention_weights = attention_weights.squeeze().cpu().numpy()
    
    # Create visualization
    html_content = create_interactive_connectivity(
        attention_weights,
        list(source_text),
        list(target_text)
    )
    
    # Save to file
    with open('attention_visualization.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # Display in notebook
    display(HTML(html_content))
    
    # Log to wandb if available
    if wandb_project:
        wandb.log({
            'attention_visualization': wandb.Html(html_content)
        }) 